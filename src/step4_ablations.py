from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from . import step1_gpr, step2_gpr


def _group_metrics(
    split_df: pd.DataFrame,
    raw_rows: pd.DataFrame,
    pred_values: np.ndarray,
    pred_col: str = "pred",
) -> dict:
    pred_df = pd.DataFrame(
        {
            "group_id": split_df["group_id"].to_numpy(dtype=int),
            pred_col: np.asarray(pred_values, dtype=float),
        }
    )
    grouped = pred_df.groupby("group_id", as_index=False).mean()
    truth = raw_rows[["group_id", "uts_mean"]].rename(columns={"uts_mean": "UTS_true"})
    merged = grouped.merge(truth, on="group_id", how="inner")
    y_true = merged["UTS_true"].to_numpy(dtype=float)
    y_pred = merged[pred_col].to_numpy(dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _fit_baseline_gp(
    x_train: np.ndarray,
    y_train_s: np.ndarray,
    seed: int,
    tuned: bool,
    tuned_restarts: int = 1,
) -> GaussianProcessRegressor:
    kernel = 1.0 * RBF(length_scale=np.array([1.0, 1.0])) + WhiteKernel(noise_level=0.1)
    if tuned:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=int(max(0, tuned_restarts)),
            normalize_y=False,
            random_state=seed,
        )
    else:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=None,
            normalize_y=False,
            random_state=seed,
        )
    gp.fit(x_train, y_train_s)
    return gp


def _fit_physics_variant(
    x_train: np.ndarray,
    y_train_s: np.ndarray,
    feats_train: dict,
    variant: str,
    tune_hyperparams: bool = False,
    tune_restarts: int = 1,
) -> step2_gpr.PhysicsInformedGP:
    h_mean = feats_train["h_mean"].copy()
    h_var = feats_train["h_var"].copy()
    p_mean = feats_train["p_mean"].copy()
    p_var = feats_train["p_var"].copy()

    if variant == "h_only":
        p_mean = np.ones_like(p_mean)
        p_var = np.full_like(p_var, 1e-12)
    elif variant == "p_only":
        h_mean = np.ones_like(h_mean)
        h_var = np.full_like(h_var, 1e-12)
    elif variant == "full":
        pass
    else:
        raise ValueError(f"Unknown physics variant: {variant}")

    model = step2_gpr.PhysicsInformedGP(
        x_train_scaled=x_train,
        y_train_scaled=y_train_s,
        h_mean_train=h_mean,
        h_var_train=h_var,
        p_mean_train=p_mean,
        p_var_train=p_var,
    ).fit(tune_hyperparams=tune_hyperparams, optimize_restarts=tune_restarts)
    return model


def _predict_physics_variant(
    model: step2_gpr.PhysicsInformedGP,
    x_scaled: np.ndarray,
    feats: dict,
    variant: str,
) -> np.ndarray:
    h_mean = feats["h_mean"].copy()
    h_var = feats["h_var"].copy()
    p_mean = feats["p_mean"].copy()
    p_var = feats["p_var"].copy()
    if variant == "h_only":
        p_mean = np.ones_like(p_mean)
        p_var = np.full_like(p_var, 1e-12)
    elif variant == "p_only":
        h_mean = np.ones_like(h_mean)
        h_var = np.full_like(h_var, 1e-12)
    mean_s, _ = model.predict(x_scaled, h_mean, h_var, p_mean, p_var, return_std=False)
    return mean_s


def _eval_suite(
    data: dict,
    s1_models: dict,
    seed: int,
    run_tuned_baseline: bool = True,
    run_tuned_physics: bool = True,
    tuned_baseline_restarts: int = 1,
    tuned_physics_restarts: int = 1,
) -> dict:
    train_df = data["train"]
    test_df = data["test"]
    raw_rows = data["raw_rows"]

    x_train_raw = train_df[["P", "v"]].to_numpy(dtype=float)
    x_test_raw = test_df[["P", "v"]].to_numpy(dtype=float)

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train_raw)
    x_test = x_scaler.transform(x_test_raw)

    y_train = train_df["UTS"].to_numpy(dtype=float)
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    feats_train = step2_gpr._build_phys_features(s1_models, x_train_raw)
    feats_test = step2_gpr._build_phys_features(s1_models, x_test_raw)

    # Full physics model
    phys_full = _fit_physics_variant(x_train, y_train_s, feats_train, variant="full")
    train_full_s = _predict_physics_variant(phys_full, x_train, feats_train, variant="full")
    test_full_s = _predict_physics_variant(phys_full, x_test, feats_test, variant="full")
    train_full = y_scaler.inverse_transform(train_full_s.reshape(-1, 1)).ravel()
    test_full = y_scaler.inverse_transform(test_full_s.reshape(-1, 1)).ravel()
    m_full_train = _group_metrics(train_df, raw_rows, train_full, pred_col="pred")
    m_full_test = _group_metrics(test_df, raw_rows, test_full, pred_col="pred")

    # Untuned baseline
    base_untuned = _fit_baseline_gp(x_train, y_train_s, seed=seed, tuned=False, tuned_restarts=0)
    train_bu = y_scaler.inverse_transform(base_untuned.predict(x_train).reshape(-1, 1)).ravel()
    test_bu = y_scaler.inverse_transform(base_untuned.predict(x_test).reshape(-1, 1)).ravel()
    m_bu_train = _group_metrics(train_df, raw_rows, train_bu, pred_col="pred")
    m_bu_test = _group_metrics(test_df, raw_rows, test_bu, pred_col="pred")

    results = {
        "physics_full": {"train": m_full_train, "test": m_full_test},
        "baseline_untuned": {"train": m_bu_train, "test": m_bu_test},
    }

    # Tuned baseline (single restart) and tuned physics (single optimizer start)
    if run_tuned_baseline:
        base_tuned = _fit_baseline_gp(
            x_train,
            y_train_s,
            seed=seed,
            tuned=True,
            tuned_restarts=tuned_baseline_restarts,
        )
        train_bt = y_scaler.inverse_transform(base_tuned.predict(x_train).reshape(-1, 1)).ravel()
        test_bt = y_scaler.inverse_transform(base_tuned.predict(x_test).reshape(-1, 1)).ravel()
        m_bt_train = _group_metrics(train_df, raw_rows, train_bt, pred_col="pred")
        m_bt_test = _group_metrics(test_df, raw_rows, test_bt, pred_col="pred")
        results["baseline_tuned"] = {"train": m_bt_train, "test": m_bt_test}
    if run_tuned_physics:
        phys_tuned = _fit_physics_variant(
            x_train,
            y_train_s,
            feats_train,
            variant="full",
            tune_hyperparams=True,
            tune_restarts=tuned_physics_restarts,
        )
        train_pt_s = _predict_physics_variant(phys_tuned, x_train, feats_train, variant="full")
        test_pt_s = _predict_physics_variant(phys_tuned, x_test, feats_test, variant="full")
        train_pt = y_scaler.inverse_transform(train_pt_s.reshape(-1, 1)).ravel()
        test_pt = y_scaler.inverse_transform(test_pt_s.reshape(-1, 1)).ravel()
        m_pt_train = _group_metrics(train_df, raw_rows, train_pt, pred_col="pred")
        m_pt_test = _group_metrics(test_df, raw_rows, test_pt, pred_col="pred")
        results["physics_tuned"] = {"train": m_pt_train, "test": m_pt_test}

    # h-only / p-only
    for v in ["h_only", "p_only"]:
        model = _fit_physics_variant(x_train, y_train_s, feats_train, variant=v)
        train_s = _predict_physics_variant(model, x_train, feats_train, variant=v)
        test_s = _predict_physics_variant(model, x_test, feats_test, variant=v)
        train_pred = y_scaler.inverse_transform(train_s.reshape(-1, 1)).ravel()
        test_pred = y_scaler.inverse_transform(test_s.reshape(-1, 1)).ravel()
        results[f"physics_{v}"] = {
            "train": _group_metrics(train_df, raw_rows, train_pred, pred_col="pred"),
            "test": _group_metrics(test_df, raw_rows, test_pred, pred_col="pred"),
        }

    return results


def _pack_metric_rows(tag: str, suite: dict) -> list[dict]:
    rows = []
    for model_name, split_data in suite.items():
        rows.append(
            {
                "scenario": tag,
                "model": model_name,
                "test_mae": float(split_data["test"]["mae"]),
                "test_rmse": float(split_data["test"]["rmse"]),
            }
        )
    return rows


def _expectation_rows(split_suite: dict, all_suite: dict) -> list[dict]:
    rows = []

    # 1) Step1 all-data improves full physics Step2
    a = split_suite["physics_full"]["test"]
    b = all_suite["physics_full"]["test"]
    cond1 = (b["mae"] < a["mae"]) or (b["rmse"] < a["rmse"])
    rows.append(
        {
            "scenario": "expectation_check",
            "model": "A1_step1_all_improves_full_physics",
            "test_mae": float(b["mae"] - a["mae"]),
            "test_rmse": float(b["rmse"] - a["rmse"]),
            "expectation_met": bool(cond1),
            "note": "delta = all_data_step1 - split_step1 (negative is better)",
        }
    )

    # 2) tuned baseline better than untuned, tuned physics better than tuned baseline on at least one axis
    bu = split_suite["baseline_untuned"]["test"]
    bt = split_suite["baseline_tuned"]["test"]
    pt = split_suite["physics_tuned"]["test"]
    cond2a = (bt["mae"] < bu["mae"]) or (bt["rmse"] < bu["rmse"])
    cond2b = (pt["mae"] < bt["mae"]) or (pt["rmse"] < bt["rmse"])
    rows.append(
        {
            "scenario": "expectation_check",
            "model": "A2_tuned_baseline_vs_untuned_and_tuned_physics",
            "test_mae": float(pt["mae"]),
            "test_rmse": float(pt["rmse"]),
            "expectation_met": bool(cond2a and cond2b),
            "note": "needs tuned_baseline<untuned_baseline and tuned_physics<tuned_baseline on at least one axis",
        }
    )

    # 3) h-only and p-only underperform full
    pf = split_suite["physics_full"]["test"]
    ph = split_suite["physics_h_only"]["test"]
    pp = split_suite["physics_p_only"]["test"]
    cond3_h = (ph["mae"] > pf["mae"]) or (ph["rmse"] > pf["rmse"])
    cond3_p = (pp["mae"] > pf["mae"]) or (pp["rmse"] > pf["rmse"])
    rows.append(
        {
            "scenario": "expectation_check",
            "model": "A3_h_only_under_full",
            "test_mae": float(ph["mae"] - pf["mae"]),
            "test_rmse": float(ph["rmse"] - pf["rmse"]),
            "expectation_met": bool(cond3_h),
            "note": "positive delta means h-only worse than full",
        }
    )
    rows.append(
        {
            "scenario": "expectation_check",
            "model": "A3_p_only_under_full",
            "test_mae": float(pp["mae"] - pf["mae"]),
            "test_rmse": float(pp["rmse"] - pf["rmse"]),
            "expectation_met": bool(cond3_p),
            "note": "positive delta means p-only worse than full",
        }
    )

    return rows


def run(
    data: dict,
    seed: int = 42,
    tuned_baseline_restarts: int = 1,
    tuned_physics_restarts: int = 1,
) -> pd.DataFrame:
    # Split-trained Step1 (normal path)
    s1_split, _ = step1_gpr.run(data, seed=seed, optimize=True, n_restarts_optimizer=1)

    # Step1 trained on all 3200 points
    data_all = dict(data)
    data_all["train"] = data["augmented"]
    data_all["train_groups"] = sorted(data["raw_rows"]["group_id"].astype(int).tolist())
    s1_all, _ = step1_gpr.run(data_all, seed=seed, optimize=True, n_restarts_optimizer=1)

    split_suite = _eval_suite(
        data,
        s1_split,
        seed=seed,
        run_tuned_baseline=True,
        run_tuned_physics=True,
        tuned_baseline_restarts=tuned_baseline_restarts,
        tuned_physics_restarts=tuned_physics_restarts,
    )
    all_suite = _eval_suite(
        data,
        s1_all,
        seed=seed,
        run_tuned_baseline=False,
        run_tuned_physics=False,
        tuned_baseline_restarts=tuned_baseline_restarts,
        tuned_physics_restarts=tuned_physics_restarts,
    )

    rows = []
    rows.extend(_pack_metric_rows("split_step1", split_suite))
    rows.extend(_pack_metric_rows("all_data_step1", all_suite))
    rows.extend(_expectation_rows(split_suite, all_suite))

    out_df = pd.DataFrame(rows)
    out_df.to_csv("outputs/step4_ablation_metrics.csv", index=False)

    checks = out_df[out_df["scenario"] == "expectation_check"].copy()
    if "expectation_met" in checks.columns and len(checks) > 0:
        met_n = int(checks["expectation_met"].astype(bool).sum())
        print(f"  Step4 expectations met: {met_n}/{len(checks)}")
    return out_df
