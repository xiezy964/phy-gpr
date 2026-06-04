from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .utils import save_pickle


@dataclass
class TargetModel:
    gp: GaussianProcessRegressor
    y_scaler: StandardScaler


def _build_gp(
    random_state: int,
    alpha: float,
    length_scale: float,
    noise_level: float,
    optimize: bool,
) -> GaussianProcessRegressor:
    kernel = RBF(length_scale=float(length_scale), length_scale_bounds=(1e-1, 1e1)) + WhiteKernel(
        noise_level=float(noise_level),
        noise_level_bounds=(1e-8, 1e1),
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(alpha),
        normalize_y=False,
        optimizer="fmin_l_bfgs_b" if optimize else None,
        n_restarts_optimizer=0,
        random_state=int(random_state),
    )


def _fit_single_gp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    optimize: bool,
    n_restarts_optimizer: int,
    alpha: float,
) -> TargetModel:
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    rng = np.random.default_rng(seed)
    n_runs = int(max(1, n_restarts_optimizer)) if optimize else 1
    best_gp = None
    best_nll = np.inf

    for restart_idx in range(n_runs):
        if restart_idx == 0:
            init_length_scale = 1.0
            init_noise = 1e-2
        else:
            init_length_scale = float(np.exp(rng.uniform(np.log(1e-1), np.log(1e1))))
            init_noise = float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-1))))

        gp = _build_gp(
            random_state=seed + restart_idx,
            alpha=alpha,
            length_scale=init_length_scale,
            noise_level=init_noise,
            optimize=optimize,
        )
        gp.fit(x_train, y_train_scaled)
        nll = float(-gp.log_marginal_likelihood_value_)
        if nll < best_nll:
            best_nll = nll
            best_gp = gp

    return TargetModel(gp=best_gp, y_scaler=y_scaler)


def _predict_target(model: TargetModel, x_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_s, std_s = model.gp.predict(x_scaled, return_std=True)
    mean = model.y_scaler.inverse_transform(mean_s.reshape(-1, 1)).ravel()
    std = std_s * float(model.y_scaler.scale_[0])
    var = np.maximum(std**2, 1e-12)
    return mean, var


def _split_eval_group_aggregated(
    split_df: pd.DataFrame,
    raw_rows: pd.DataFrame,
    model_d: TargetModel,
    model_phi: TargetModel,
    x_scaler: StandardScaler,
) -> dict:
    x = split_df[["P", "v"]].to_numpy(dtype=float)
    x_s = x_scaler.transform(x)

    d_pred, d_var = _predict_target(model_d, x_s)
    phi_pred, phi_var = _predict_target(model_phi, x_s)
    pred_df = pd.DataFrame(
        {
            "group_id": split_df["group_id"].to_numpy(dtype=int),
            "d_pred": d_pred,
            "phi_pred": phi_pred,
        }
    )
    grouped_pred = pred_df.groupby("group_id", as_index=False).mean()
    truth = raw_rows[["group_id", "d_mean", "phi_mean"]].copy()
    merged = grouped_pred.merge(truth, on="group_id", how="inner")

    y_d = merged["d_mean"].to_numpy(dtype=float)
    y_phi = merged["phi_mean"].to_numpy(dtype=float)
    d_pred_g = merged["d_pred"].to_numpy(dtype=float)
    phi_pred_g = merged["phi_pred"].to_numpy(dtype=float)

    metrics = {
        "d_mae": float(mean_absolute_error(y_d, d_pred_g)),
        "d_rmse": float(np.sqrt(mean_squared_error(y_d, d_pred_g))),
        "phi_mae": float(mean_absolute_error(y_phi, phi_pred_g)),
        "phi_rmse": float(np.sqrt(mean_squared_error(y_phi, phi_pred_g))),
    }
    pred_df = {
        "d_true": y_d,
        "d_pred": d_pred_g,
        "d_var": np.zeros_like(d_pred_g),
        "phi_true": y_phi,
        "phi_pred": phi_pred_g,
        "phi_var": np.zeros_like(phi_pred_g),
    }
    return {"metrics": metrics, "pred_df": pred_df}


def _plot_predictions(train_df: dict, test_df: dict, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = axes.ravel()
    for ax, target in zip(axes, ["d", "phi"]):
        train_true = np.asarray(train_df[f"{target}_true"])
        train_pred = np.asarray(train_df[f"{target}_pred"])
        test_true = np.asarray(test_df[f"{target}_true"])
        test_pred = np.asarray(test_df[f"{target}_pred"])

        x_train = np.arange(len(train_true))
        x_test = np.arange(len(test_true)) + len(train_true)

        ax.plot(x_train, train_true, color="tab:blue", linewidth=1.6, label="Train True")
        ax.plot(x_train, train_pred, color="tab:orange", linewidth=1.6, label="Train Pred")
        ax.plot(x_test, test_true, color="tab:green", linewidth=1.6, linestyle="--", label="Test True")
        ax.plot(x_test, test_pred, color="tab:red", linewidth=1.6, linestyle="--", label="Test Pred")

        ax.axvline(len(train_true) - 0.5, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Sample Index (train then test)")
        ax.set_ylabel(target)
        ax.set_title(f"Step1 {target}: true vs predicted (line plot)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_errors(train_metrics: dict, test_metrics: dict, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(2)
    width = 0.35
    labels = ["d", "phi"]

    mae_train = [train_metrics["d_mae"], train_metrics["phi_mae"]]
    mae_test = [test_metrics["d_mae"], test_metrics["phi_mae"]]
    rmse_train = [train_metrics["d_rmse"], train_metrics["phi_rmse"]]
    rmse_test = [test_metrics["d_rmse"], test_metrics["phi_rmse"]]

    axes[0].bar(x - width / 2, mae_train, width=width, label="Train")
    axes[0].bar(x + width / 2, mae_test, width=width, label="Test")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Step1 Overall MAE")
    axes[0].legend()

    axes[1].bar(x - width / 2, rmse_train, width=width, label="Train")
    axes[1].bar(x + width / 2, rmse_test, width=width, label="Test")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Step1 Overall RMSE")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(
    data: dict,
    seed: int = 42,
    optimize: bool = False,
    n_restarts_optimizer: int = 1,
    alpha: float = 1e-6,
) -> tuple[dict, dict]:
    train_df = data["train"]
    x_train = train_df[["P", "v"]].to_numpy(dtype=float)
    x_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)

    d_train = train_df["d"].to_numpy(dtype=float)
    phi_train = train_df["phi"].to_numpy(dtype=float)
    model_d = _fit_single_gp(
        x_train_scaled,
        d_train,
        seed=seed,
        optimize=optimize,
        n_restarts_optimizer=n_restarts_optimizer,
        alpha=alpha,
    )
    model_phi = _fit_single_gp(
        x_train_scaled,
        phi_train,
        seed=seed + 1,
        optimize=optimize,
        n_restarts_optimizer=n_restarts_optimizer,
        alpha=alpha,
    )

    train_eval = _split_eval_group_aggregated(data["train"], data["raw_rows"], model_d, model_phi, x_scaler)
    test_eval = _split_eval_group_aggregated(data["test"], data["raw_rows"], model_d, model_phi, x_scaler)

    _plot_predictions(train_eval["pred_df"], test_eval["pred_df"], "outputs/step1_predictions.png")
    _plot_errors(train_eval["metrics"], test_eval["metrics"], "outputs/step1_errors.png")
    metrics_df = pd.DataFrame(
        [
            {"split": "train", **train_eval["metrics"]},
            {"split": "test", **test_eval["metrics"]},
        ]
    )
    metrics_df.to_csv("outputs/step1_metrics.csv", index=False)

    save_pickle(
        "outputs/step1_scalers.pkl",
        {
            "x_scaler": x_scaler,
            "y_scaler_d": model_d.y_scaler,
            "y_scaler_phi": model_phi.y_scaler,
        },
    )
    # Extension requested in demand; contents are pickle-serializable sklearn objects.
    save_pickle(
        "outputs/step1_model.pth",
        {
            "model_d": model_d.gp,
            "model_phi": model_phi.gp,
        },
    )

    results = {
        "train_metrics": train_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "train_pred_df": train_eval["pred_df"],
        "test_pred_df": test_eval["pred_df"],
        "metrics_df": metrics_df,
    }
    models = {
        "x_scaler": x_scaler,
        "d_model": model_d,
        "phi_model": model_phi,
    }

    print(
        "  Step1 metrics | "
        f"train: d_MAE={results['train_metrics']['d_mae']:.4f}, d_RMSE={results['train_metrics']['d_rmse']:.4f}, "
        f"phi_MAE={results['train_metrics']['phi_mae']:.4f}, phi_RMSE={results['train_metrics']['phi_rmse']:.4f} | "
        f"test: d_MAE={results['test_metrics']['d_mae']:.4f}, d_RMSE={results['test_metrics']['d_rmse']:.4f}, "
        f"phi_MAE={results['test_metrics']['phi_mae']:.4f}, phi_RMSE={results['test_metrics']['phi_rmse']:.4f}"
    )
    return models, results


def predict_d_phi_with_uncertainty(s1_models: dict, x_raw: np.ndarray) -> dict:
    x_s = s1_models["x_scaler"].transform(x_raw)
    d_mean, d_var = _predict_target(s1_models["d_model"], x_s)
    phi_mean, phi_var = _predict_target(s1_models["phi_model"], x_s)
    return {
        "d_mean": d_mean,
        "d_var": d_var,
        "phi_mean": phi_mean,
        "phi_var": phi_var,
    }
