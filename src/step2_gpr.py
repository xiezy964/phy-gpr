from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .step1_gpr import predict_d_phi_with_uncertainty
from .utils import save_pickle


def _mean_gaussian_nll(y_true: np.ndarray, y_mean: np.ndarray, y_var: np.ndarray) -> float:
    var = np.maximum(np.asarray(y_var, dtype=float), 1e-12)
    diff2 = (np.asarray(y_true, dtype=float) - np.asarray(y_mean, dtype=float)) ** 2
    nll = 0.5 * (np.log(2.0 * np.pi * var) + diff2 / var)
    return float(np.mean(nll))


def _baseline_obs_noise_var_raw(base_gp: GaussianProcessRegressor, y_scaler: StandardScaler) -> float:
    # WhiteKernel noise is learned/fixed in standardized y-space.
    noise_s = None
    try:
        if hasattr(base_gp, "kernel_") and hasattr(base_gp.kernel_, "k2"):
            noise_s = float(base_gp.kernel_.k2.noise_level)
    except Exception:
        noise_s = None
    if noise_s is None:
        noise_s = 1e-12
    y_scale = float(y_scaler.scale_[0])
    return float(max(noise_s * (y_scale**2), 1e-12))


def _mean_gaussian_crps(y_true: np.ndarray, y_mean: np.ndarray, y_var: np.ndarray) -> float:
    sigma = np.sqrt(np.maximum(np.asarray(y_var, dtype=float), 1e-12))
    z = (np.asarray(y_true, dtype=float) - np.asarray(y_mean, dtype=float)) / sigma
    crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def _safe_cholesky(a: np.ndarray, jitter: float = 1e-8, max_tries: int = 8) -> np.ndarray:
    eye = np.eye(a.shape[0])
    cur = jitter
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(a + cur * eye)
        except np.linalg.LinAlgError:
            cur *= 10.0
    raise np.linalg.LinAlgError("Failed Cholesky decomposition even after jitter escalation.")


def _hall_petch_moments(d_mean: np.ndarray, d_var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Delta-method moments for h(d)=d^(-1/2), with clipping for physical positivity.
    mu = np.clip(d_mean, 1e-6, None)
    var = np.clip(d_var, 1e-12, None)
    h_mean = mu ** (-0.5) + 0.375 * var * mu ** (-2.5)
    h_var = (0.25 * mu**-3) * var
    return h_mean, np.clip(h_var, 1e-12, None)


def _porosity_moments(phi_mean_pct: np.ndarray, phi_var_pct: np.ndarray, n: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    # Convert percent to ratio.
    mu = phi_mean_pct / 100.0
    var = np.clip(phi_var_pct / (100.0**2), 1e-12, None)
    base = np.clip(1.0 - mu, 1e-6, None)
    p_mean = base**n + 0.5 * n * (n - 1.0) * base ** (n - 2.0) * var
    p_var = (n**2) * base ** (2.0 * n - 2.0) * var
    return np.clip(p_mean, 1e-12, None), np.clip(p_var, 1e-12, None)


def _expected_uncertain_rbf(
    mu_a: np.ndarray,
    var_a: np.ndarray,
    mu_b: np.ndarray,
    var_b: np.ndarray,
    length_scale: float,
) -> np.ndarray:
    # k = sqrt(l^2 / (l^2 + sigma^2)) * exp(-(mu_i-mu_j)^2 / (2*(l^2 + sigma^2)))
    mu_diff = mu_a[:, None] - mu_b[None, :]
    sigma2 = var_a[:, None] + var_b[None, :]
    denom = (length_scale**2) + sigma2
    prefactor = np.sqrt((length_scale**2) / np.maximum(denom, 1e-12))
    return prefactor * np.exp(-(mu_diff**2) / (2.0 * np.maximum(denom, 1e-12)))


@dataclass
class PhysicsInformedGP:
    x_train_scaled: np.ndarray
    y_train_scaled: np.ndarray
    h_mean_train: np.ndarray
    h_var_train: np.ndarray
    p_mean_train: np.ndarray
    p_var_train: np.ndarray
    params: np.ndarray | None = None
    alpha: np.ndarray | None = None
    l_train: np.ndarray | None = None

    @staticmethod
    def _kernel(
        x_a: np.ndarray,
        x_b: np.ndarray,
        h_mean_a: np.ndarray,
        h_var_a: np.ndarray,
        p_mean_a: np.ndarray,
        p_var_a: np.ndarray,
        h_mean_b: np.ndarray,
        h_var_b: np.ndarray,
        p_mean_b: np.ndarray,
        p_var_b: np.ndarray,
        params: np.ndarray,
        add_noise_diag: bool = True,
    ) -> np.ndarray:
        log_sigma_f, log_l_proc, log_l_h, log_l_p, log_sigma_n = params
        sigma_f = np.exp(log_sigma_f)
        l_proc = np.exp(log_l_proc)
        l_h = np.exp(log_l_h)
        l_p = np.exp(log_l_p)
        sigma_n2 = np.exp(log_sigma_n) ** 2

        sqdist = cdist(x_a, x_b, metric="sqeuclidean")
        k_proc = np.exp(-sqdist / (2.0 * l_proc**2))
        k_h = _expected_uncertain_rbf(h_mean_a, h_var_a, h_mean_b, h_var_b, l_h)
        k_p = _expected_uncertain_rbf(p_mean_a, p_var_a, p_mean_b, p_var_b, l_p)

        k = (sigma_f**2) * (k_proc + k_h * k_p)
        if add_noise_diag and x_a.shape[0] == x_b.shape[0] and np.allclose(x_a, x_b):
            k = k + (sigma_n2 + 1e-8) * np.eye(x_a.shape[0])
        return k

    def _nll(self, params: np.ndarray) -> float:
        k = self._kernel(
            self.x_train_scaled,
            self.x_train_scaled,
            self.h_mean_train,
            self.h_var_train,
            self.p_mean_train,
            self.p_var_train,
            self.h_mean_train,
            self.h_var_train,
            self.p_mean_train,
            self.p_var_train,
            params=params,
            add_noise_diag=True,
        )
        l = _safe_cholesky(k)
        alpha = np.linalg.solve(l.T, np.linalg.solve(l, self.y_train_scaled))
        n = self.x_train_scaled.shape[0]
        nll = 0.5 * self.y_train_scaled @ alpha + np.sum(np.log(np.diag(l))) + 0.5 * n * np.log(2.0 * np.pi)
        return float(nll)

    def fit(self, tune_hyperparams: bool = False, optimize_restarts: int = 1) -> "PhysicsInformedGP":
        if tune_hyperparams:
            x0 = np.log(np.array([1.0, 1.0, 1.0, 1.0, 0.1]))
            bounds = [(-4, 4), (-4, 4), (-4, 4), (-4, 4), (-7, 1)]
            n_runs = int(max(1, optimize_restarts))
            rng = np.random.default_rng(42)
            best_x = x0
            best_fun = np.inf
            for ridx in range(n_runs):
                if ridx == 0:
                    init = x0
                else:
                    init = np.array([rng.uniform(low=b[0], high=b[1]) for b in bounds], dtype=float)
                res = minimize(self._nll, x0=init, method="L-BFGS-B", bounds=bounds)
                if float(res.fun) < best_fun:
                    best_fun = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
            self.params = best_x
        else:
            # Primary comparison: no tuning.
            self.params = np.log(np.array([1.0, 1.0, 1.0, 1.0, 0.1]))

        k = self._kernel(
            self.x_train_scaled,
            self.x_train_scaled,
            self.h_mean_train,
            self.h_var_train,
            self.p_mean_train,
            self.p_var_train,
            self.h_mean_train,
            self.h_var_train,
            self.p_mean_train,
            self.p_var_train,
            params=self.params,
            add_noise_diag=True,
        )
        self.l_train = _safe_cholesky(k)
        self.alpha = np.linalg.solve(self.l_train.T, np.linalg.solve(self.l_train, self.y_train_scaled))
        return self

    def predict(
        self,
        x_scaled: np.ndarray,
        h_mean: np.ndarray,
        h_var: np.ndarray,
        p_mean: np.ndarray,
        p_var: np.ndarray,
        return_std: bool = True,
        include_observation_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.params is None or self.alpha is None or self.l_train is None:
            raise RuntimeError("Model not fitted.")
        k_star = self._kernel(
            x_scaled,
            self.x_train_scaled,
            h_mean,
            h_var,
            p_mean,
            p_var,
            self.h_mean_train,
            self.h_var_train,
            self.p_mean_train,
            self.p_var_train,
            params=self.params,
            add_noise_diag=False,
        )
        mean = k_star @ self.alpha
        if not return_std:
            return mean, np.zeros_like(mean)
        v = np.linalg.solve(self.l_train, k_star.T)
        k_ss = self._kernel(
            x_scaled,
            x_scaled,
            h_mean,
            h_var,
            p_mean,
            p_var,
            h_mean,
            h_var,
            p_mean,
            p_var,
            params=self.params,
            add_noise_diag=False,
        )
        var = np.maximum(np.diag(k_ss) - np.sum(v**2, axis=0), 1e-12)
        if include_observation_noise:
            sigma_n2 = float(np.exp(self.params[4]) ** 2)
            var = var + sigma_n2
        return mean, np.sqrt(var)


def _build_phys_features(s1_models: dict, x_raw: np.ndarray, n_porosity: float = 2.0) -> dict:
    out = predict_d_phi_with_uncertainty(s1_models, x_raw)
    h_mean, h_var = _hall_petch_moments(out["d_mean"], out["d_var"])
    p_mean, p_var = _porosity_moments(out["phi_mean"], out["phi_var"], n=n_porosity)
    return {
        "d_mean": out["d_mean"],
        "d_var": out["d_var"],
        "phi_mean": out["phi_mean"],
        "phi_var": out["phi_var"],
        "h_mean": h_mean,
        "h_var": h_var,
        "p_mean": p_mean,
        "p_var": p_var,
    }


def _split_eval_step2(
    split_df: pd.DataFrame,
    raw_rows: pd.DataFrame,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    phys_gp: PhysicsInformedGP,
    base_gp: GaussianProcessRegressor,
    s1_models: dict,
) -> pd.DataFrame:
    group_ids = sorted(split_df["group_id"].astype(int).unique().tolist())
    group_df = raw_rows[raw_rows["group_id"].isin(group_ids)].copy().sort_values("group_id").reset_index(drop=True)

    x_raw = group_df[["P", "v"]].to_numpy(dtype=float)
    x_scaled = x_scaler.transform(x_raw)
    feats = _build_phys_features(s1_models, x_raw)
    phys_mean_s, phys_std_s = phys_gp.predict(
        x_scaled,
        feats["h_mean"],
        feats["h_var"],
        feats["p_mean"],
        feats["p_var"],
        return_std=True,
        include_observation_noise=True,
    )
    base_mean_s, base_std_s = base_gp.predict(x_scaled, return_std=True)
    phys_mean = y_scaler.inverse_transform(phys_mean_s.reshape(-1, 1)).ravel()
    base_mean = y_scaler.inverse_transform(base_mean_s.reshape(-1, 1)).ravel()
    y_scale = float(y_scaler.scale_[0])
    phys_var = np.maximum((phys_std_s * y_scale) ** 2, 1e-12)
    base_var = np.maximum((base_std_s * y_scale) ** 2, 1e-12)
    base_var = base_var + _baseline_obs_noise_var_raw(base_gp, y_scaler)

    out = pd.DataFrame(
        {
            "group_id": group_df["group_id"].to_numpy(dtype=int),
            "P": group_df["P"].to_numpy(dtype=float),
            "v": group_df["v"].to_numpy(dtype=float),
            "UTS_true": group_df["uts_mean"].to_numpy(dtype=float),
            "UTS_pred_physics": phys_mean,
            "UTS_pred_baseline": base_mean,
            "UTS_var_physics": phys_var,
            "UTS_var_baseline": base_var,
        }
    )
    return out


def _compute_metrics(pred_df: pd.DataFrame) -> dict:
    y = pred_df["UTS_true"].to_numpy()
    yp = pred_df["UTS_pred_physics"].to_numpy()
    yb = pred_df["UTS_pred_baseline"].to_numpy()
    vp = pred_df["UTS_var_physics"].to_numpy()
    vb = pred_df["UTS_var_baseline"].to_numpy()
    return {
        "physics_mae": float(mean_absolute_error(y, yp)),
        "physics_rmse": float(np.sqrt(mean_squared_error(y, yp))),
        "physics_nll": _mean_gaussian_nll(y, yp, vp),
        "physics_crps": _mean_gaussian_crps(y, yp, vp),
        "baseline_mae": float(mean_absolute_error(y, yb)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y, yb))),
        "baseline_nll": _mean_gaussian_nll(y, yb, vb),
        "baseline_crps": _mean_gaussian_crps(y, yb, vb),
    }


def _predict_step2_posterior(
    x_raw: np.ndarray,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    phys_gp: PhysicsInformedGP,
    base_gp: GaussianProcessRegressor,
    s1_models: dict,
) -> dict:
    x_scaled = x_scaler.transform(x_raw)
    feats = _build_phys_features(s1_models, x_raw)

    phys_mean_s, phys_std_s = phys_gp.predict(
        x_scaled,
        feats["h_mean"],
        feats["h_var"],
        feats["p_mean"],
        feats["p_var"],
        return_std=True,
    )
    base_mean_s, base_std_s = base_gp.predict(x_scaled, return_std=True)

    y_scale = float(y_scaler.scale_[0])
    phys_mean = y_scaler.inverse_transform(phys_mean_s.reshape(-1, 1)).ravel()
    base_mean = y_scaler.inverse_transform(base_mean_s.reshape(-1, 1)).ravel()
    phys_var = np.maximum((phys_std_s * y_scale) ** 2, 1e-12)
    base_var = np.maximum((base_std_s * y_scale) ** 2, 1e-12)

    return {
        "physics_mean": phys_mean,
        "physics_var": phys_var,
        "baseline_mean": base_mean,
        "baseline_var": base_var,
    }


def _plot_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model_col, title in [
        (axes[0], "UTS_pred_physics", "Physics-informed"),
        (axes[1], "UTS_pred_baseline", "Baseline RBF"),
    ]:
        train_true = train_df["UTS_true"].to_numpy(dtype=float)
        train_pred = train_df[model_col].to_numpy(dtype=float)
        test_true = test_df["UTS_true"].to_numpy(dtype=float)
        test_pred = test_df[model_col].to_numpy(dtype=float)

        x_train = np.arange(len(train_true))
        x_test = np.arange(len(test_true)) + len(train_true)

        ax.plot(x_train, train_true, color="tab:blue", linewidth=1.6, label="Train True")
        ax.plot(x_train, train_pred, color="tab:orange", linewidth=1.6, label="Train Pred")
        ax.plot(x_test, test_true, color="tab:green", linewidth=1.6, linestyle="--", label="Test True")
        ax.plot(x_test, test_pred, color="tab:red", linewidth=1.6, linestyle="--", label="Test Pred")
        ax.axvline(len(train_true) - 0.5, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Group Index (train then test)")
        ax.set_ylabel("UTS")
        ax.set_title(f"{title}: true vs predicted (line plot)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pv_maps(
    raw_rows: pd.DataFrame,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    phys_gp: PhysicsInformedGP,
    base_gp: GaussianProcessRegressor,
    s1_models: dict,
    out_path: str,
    variance_clip: float = 4.0,
    grid_size: int = 120,
) -> None:
    import matplotlib.pyplot as plt

    p_vals = np.linspace(float(raw_rows["P"].min()), float(raw_rows["P"].max()), grid_size)
    v_vals = np.linspace(float(raw_rows["v"].min()), float(raw_rows["v"].max()), grid_size)
    p_grid, v_grid = np.meshgrid(p_vals, v_vals)
    x_grid = np.column_stack([p_grid.ravel(), v_grid.ravel()])

    post = _predict_step2_posterior(x_grid, x_scaler, y_scaler, phys_gp, base_gp, s1_models)
    physics_mean = post["physics_mean"].reshape(grid_size, grid_size)
    physics_var = np.minimum(post["physics_var"], variance_clip).reshape(grid_size, grid_size)
    mean_vmin = float(physics_mean.min())
    mean_vmax = float(physics_mean.max())

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    mean_levels = np.linspace(mean_vmin, mean_vmax, 18)
    var_levels = np.linspace(0.0, variance_clip, 18)

    marker_face = "#bfbfbf"
    marker_edge = "#333333"
    scatter_x = raw_rows["P"].to_numpy(dtype=float)
    scatter_y = raw_rows["v"].to_numpy(dtype=float)

    mean_cf = axes[0].contourf(
        p_grid,
        v_grid,
        physics_mean,
        levels=mean_levels,
        cmap="viridis",
        extend="both",
    )
    axes[0].scatter(
        scatter_x,
        scatter_y,
        s=16,
        c=marker_face,
        edgecolors=marker_edge,
        linewidths=0.6,
        alpha=0.95,
        zorder=3,
    )
    axes[0].set_title("Step2 Physics-Informed UTS Mean on P-v Grid")
    axes[0].set_xlabel("P (W)")
    axes[0].set_ylabel("V (mm/s)")
    cbar_mean = fig.colorbar(mean_cf, ax=axes[0])
    cbar_mean.set_label("UTS mean (MPa)")

    var_cf = axes[1].contourf(
        p_grid,
        v_grid,
        physics_var,
        levels=var_levels,
        cmap="magma",
        extend="max",
    )
    axes[1].scatter(
        scatter_x,
        scatter_y,
        s=16,
        c=marker_face,
        edgecolors=marker_edge,
        linewidths=0.6,
        alpha=0.95,
        zorder=3,
    )
    axes[1].set_title("Step2 Physics-Informed UTS Variance on P-v Grid")
    axes[1].set_xlabel("P (W)")
    axes[1].set_ylabel("V (mm/s)")
    cbar_var = fig.colorbar(var_cf, ax=axes[1])
    ticks = np.linspace(0.0, variance_clip, 6)
    cbar_var.set_ticks(ticks)
    tick_labels = [f"{tick:.2f}" for tick in ticks]
    tick_labels[-1] = f">={variance_clip:.2f}"
    cbar_var.set_ticklabels(tick_labels)
    cbar_var.set_label("UTS variance (MPa$^2$)")

    for ax in axes:
        ax.set_xlim(float(raw_rows["P"].min()), float(raw_rows["P"].max()))
        ax.set_ylim(float(raw_rows["v"].min()), float(raw_rows["v"].max()))

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_baseline_pv_maps(
    raw_rows: pd.DataFrame,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    phys_gp: PhysicsInformedGP,
    base_gp: GaussianProcessRegressor,
    s1_models: dict,
    out_path: str,
    variance_clip: float = 10.0,
    grid_size: int = 120,
) -> None:
    import matplotlib.pyplot as plt

    p_vals = np.linspace(float(raw_rows["P"].min()), float(raw_rows["P"].max()), grid_size)
    v_vals = np.linspace(float(raw_rows["v"].min()), float(raw_rows["v"].max()), grid_size)
    p_grid, v_grid = np.meshgrid(p_vals, v_vals)
    x_grid = np.column_stack([p_grid.ravel(), v_grid.ravel()])

    post = _predict_step2_posterior(x_grid, x_scaler, y_scaler, phys_gp, base_gp, s1_models)
    baseline_mean = post["baseline_mean"].reshape(grid_size, grid_size)
    baseline_var = np.minimum(post["baseline_var"], variance_clip).reshape(grid_size, grid_size)
    mean_vmin = float(baseline_mean.min())
    mean_vmax = float(baseline_mean.max())

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    mean_levels = np.linspace(mean_vmin, mean_vmax, 18)
    var_levels = np.linspace(0.0, variance_clip, 18)

    marker_face = "#bfbfbf"
    marker_edge = "#333333"
    scatter_x = raw_rows["P"].to_numpy(dtype=float)
    scatter_y = raw_rows["v"].to_numpy(dtype=float)

    mean_cf = axes[0].contourf(
        p_grid,
        v_grid,
        baseline_mean,
        levels=mean_levels,
        cmap="viridis",
        extend="both",
    )
    axes[0].scatter(
        scatter_x,
        scatter_y,
        s=16,
        c=marker_face,
        edgecolors=marker_edge,
        linewidths=0.6,
        alpha=0.95,
        zorder=3,
    )
    axes[0].set_title("Step2 Baseline UTS Mean on P-v Grid")
    axes[0].set_xlabel("P (W)")
    axes[0].set_ylabel("V (mm/s)")
    cbar_mean = fig.colorbar(mean_cf, ax=axes[0])
    cbar_mean.set_label("UTS mean (MPa)")

    var_cf = axes[1].contourf(
        p_grid,
        v_grid,
        baseline_var,
        levels=var_levels,
        cmap="magma",
        extend="max",
    )
    axes[1].scatter(
        scatter_x,
        scatter_y,
        s=16,
        c=marker_face,
        edgecolors=marker_edge,
        linewidths=0.6,
        alpha=0.95,
        zorder=3,
    )
    axes[1].set_title("Step2 Baseline UTS Variance on P-v Grid")
    axes[1].set_xlabel("P (W)")
    axes[1].set_ylabel("V (mm/s)")
    cbar_var = fig.colorbar(var_cf, ax=axes[1])
    ticks = np.linspace(0.0, variance_clip, 6)
    cbar_var.set_ticks(ticks)
    tick_labels = [f"{tick:.2f}" for tick in ticks]
    tick_labels[-1] = f">={variance_clip:.2f}"
    cbar_var.set_ticklabels(tick_labels)
    cbar_var.set_label("UTS variance (MPa$^2$)")

    for ax in axes:
        ax.set_xlim(float(raw_rows["P"].min()), float(raw_rows["P"].max()))
        ax.set_ylim(float(raw_rows["v"].min()), float(raw_rows["v"].max()))

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _export_pv_map_data(
    raw_rows: pd.DataFrame,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    phys_gp: PhysicsInformedGP,
    base_gp: GaussianProcessRegressor,
    s1_models: dict,
    out_path: str,
    grid_size: int = 120,
) -> None:
    p_vals = np.linspace(float(raw_rows["P"].min()), float(raw_rows["P"].max()), grid_size)
    v_vals = np.linspace(float(raw_rows["v"].min()), float(raw_rows["v"].max()), grid_size)
    p_grid, v_grid = np.meshgrid(p_vals, v_vals)
    x_grid = np.column_stack([p_grid.ravel(), v_grid.ravel()])

    post = _predict_step2_posterior(x_grid, x_scaler, y_scaler, phys_gp, base_gp, s1_models)
    out_df = pd.DataFrame(
        {
            "P": x_grid[:, 0],
            "v": x_grid[:, 1],
            "physics_mean": post["physics_mean"],
            "physics_var": post["physics_var"],
            "baseline_mean": post["baseline_mean"],
            "baseline_var": post["baseline_var"],
        }
    )
    out_df.to_csv(out_path, index=False)


def _export_baseline_pv_map_data(
    raw_rows: pd.DataFrame,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    phys_gp: PhysicsInformedGP,
    base_gp: GaussianProcessRegressor,
    s1_models: dict,
    out_path: str,
    grid_size: int = 120,
) -> None:
    p_vals = np.linspace(float(raw_rows["P"].min()), float(raw_rows["P"].max()), grid_size)
    v_vals = np.linspace(float(raw_rows["v"].min()), float(raw_rows["v"].max()), grid_size)
    p_grid, v_grid = np.meshgrid(p_vals, v_vals)
    x_grid = np.column_stack([p_grid.ravel(), v_grid.ravel()])

    post = _predict_step2_posterior(x_grid, x_scaler, y_scaler, phys_gp, base_gp, s1_models)
    out_df = pd.DataFrame(
        {
            "P": x_grid[:, 0],
            "v": x_grid[:, 1],
            "baseline_mean": post["baseline_mean"],
            "baseline_var": post["baseline_var"],
        }
    )
    out_df.to_csv(out_path, index=False)


def _plot_errors(train_metrics: dict, test_metrics: dict, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(2)
    width = 0.35
    labels = ["physics", "baseline"]

    mae_train = [train_metrics["physics_mae"], train_metrics["baseline_mae"]]
    mae_test = [test_metrics["physics_mae"], test_metrics["baseline_mae"]]
    rmse_train = [train_metrics["physics_rmse"], train_metrics["baseline_rmse"]]
    rmse_test = [test_metrics["physics_rmse"], test_metrics["baseline_rmse"]]

    axes[0].bar(x - width / 2, mae_train, width=width, label="Train")
    axes[0].bar(x + width / 2, mae_test, width=width, label="Test")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Step2 Overall MAE")
    axes[0].legend()

    axes[1].bar(x - width / 2, rmse_train, width=width, label="Train")
    axes[1].bar(x + width / 2, rmse_test, width=width, label="Test")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Step2 Overall RMSE")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(data: dict, s1_models: dict, seed: int = 42) -> tuple[dict, dict]:
    train_df = data["train"]
    x_train_raw = train_df[["P", "v"]].to_numpy(dtype=float)
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train_raw)

    y_train = train_df["UTS"].to_numpy(dtype=float)
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    feats_train = _build_phys_features(s1_models, x_train_raw)
    phys_gp = PhysicsInformedGP(
        x_train_scaled=x_train,
        y_train_scaled=y_train_s,
        h_mean_train=feats_train["h_mean"],
        h_var_train=feats_train["h_var"],
        p_mean_train=feats_train["p_mean"],
        p_var_train=feats_train["p_var"],
    ).fit(tune_hyperparams=False)

    base_kernel = 1.0 * RBF(length_scale=np.array([1.0, 1.0])) + WhiteKernel(noise_level=0.1)
    baseline_gp = GaussianProcessRegressor(
        kernel=base_kernel,
        optimizer=None,
        normalize_y=False,
        random_state=seed,
    )
    baseline_gp.fit(x_train, y_train_s)

    train_pred_df = _split_eval_step2(
        data["train"], data["raw_rows"], x_scaler, y_scaler, phys_gp, baseline_gp, s1_models
    )
    test_pred_df = _split_eval_step2(
        data["test"], data["raw_rows"], x_scaler, y_scaler, phys_gp, baseline_gp, s1_models
    )

    train_metrics = _compute_metrics(train_pred_df)
    test_metrics = _compute_metrics(test_pred_df)
    metrics_df = pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "test", **test_metrics},
        ]
    )
    metrics_df.to_csv("outputs/step2_metrics.csv", index=False)

    _plot_predictions(train_pred_df, test_pred_df, "outputs/step2_predictions.png")
    _plot_pv_maps(
        data["raw_rows"],
        x_scaler,
        y_scaler,
        phys_gp,
        baseline_gp,
        s1_models,
        "outputs/step2_pv_maps.png",
    )
    _plot_baseline_pv_maps(
        data["raw_rows"],
        x_scaler,
        y_scaler,
        phys_gp,
        baseline_gp,
        s1_models,
        "outputs/step2_baseline_pv_maps.png",
    )
    _export_pv_map_data(
        data["raw_rows"],
        x_scaler,
        y_scaler,
        phys_gp,
        baseline_gp,
        s1_models,
        "outputs/step2_pv_map_data.csv",
    )
    _export_baseline_pv_map_data(
        data["raw_rows"],
        x_scaler,
        y_scaler,
        phys_gp,
        baseline_gp,
        s1_models,
        "outputs/step2_baseline_pv_map_data.csv",
    )
    _plot_errors(train_metrics, test_metrics, "outputs/step2_errors.png")

    save_pickle("outputs/step2_scalers.pkl", {"x_scaler": x_scaler, "y_scaler": y_scaler})
    save_pickle(
        "outputs/step2_model.pkl",
        {
            "physics_model": phys_gp,
            "baseline_model": baseline_gp,
            "n_porosity": 2.0,
        },
    )

    print(
        "  Step2 metrics | "
        f"physics test MAE={test_metrics['physics_mae']:.4f}, baseline test MAE={test_metrics['baseline_mae']:.4f}; "
        f"physics test RMSE={test_metrics['physics_rmse']:.4f}, baseline test RMSE={test_metrics['baseline_rmse']:.4f}; "
        f"physics test NLL={test_metrics['physics_nll']:.4f}, baseline test NLL={test_metrics['baseline_nll']:.4f}; "
        f"physics test CRPS={test_metrics['physics_crps']:.4f}, baseline test CRPS={test_metrics['baseline_crps']:.4f}"
    )

    models = {
        "physics_model": phys_gp,
        "baseline_model": baseline_gp,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "n_porosity": 2.0,
    }
    results = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_pred_df": train_pred_df,
        "test_pred_df": test_pred_df,
        "metrics_df": metrics_df,
    }
    return models, results


def predict_uts_mean(
    s2_models: dict,
    s1_models: dict,
    x_raw: np.ndarray,
    model_kind: str,
) -> np.ndarray:
    x_scaled = s2_models["x_scaler"].transform(x_raw)
    if model_kind == "physics":
        feats = _build_phys_features(s1_models, x_raw, n_porosity=s2_models.get("n_porosity", 2.0))
        mean_s, _ = s2_models["physics_model"].predict(
            x_scaled,
            feats["h_mean"],
            feats["h_var"],
            feats["p_mean"],
            feats["p_var"],
            return_std=True,
        )
    elif model_kind == "baseline":
        mean_s = s2_models["baseline_model"].predict(x_scaled)
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")
    return s2_models["y_scaler"].inverse_transform(mean_s.reshape(-1, 1)).ravel()
