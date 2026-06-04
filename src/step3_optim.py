from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .step2_gpr import predict_uts_mean


def run(s2_models: dict, s1_models: dict, data: dict) -> pd.DataFrame:
    raw = data["raw_rows"]
    # Step 3 targets must come from the original Excel group means for the held-out test groups.
    test_group_ids = {int(g) for g in data["test_groups"]}
    test_rows = raw[raw["group_id"].isin(test_group_ids)].copy().reset_index(drop=True)
    if set(test_rows["group_id"].astype(int).tolist()) != test_group_ids:
        raise ValueError("Step3 targets must be selected only from the held-out test groups.")

    train_group_ids = {int(g) for g in data["train_groups"]}
    train_rows = raw[raw["group_id"].isin(train_group_ids)].copy().reset_index(drop=True)

    p_bounds = (float(raw["P"].min()), float(raw["P"].max()))
    v_bounds = (float(raw["v"].min()), float(raw["v"].max()))
    bounds = [p_bounds, v_bounds]
    x0 = np.array([float(train_rows["P"].mean()), float(train_rows["v"].mean())], dtype=float)

    summary = []
    histories = {"physics": {}, "baseline": {}}
    convergence_rows: list[dict] = []

    for _, row in test_rows.iterrows():
        target = float(row["uts_mean"])
        start_point = np.array([float(row["P"]), float(row["v"])], dtype=float)

        for model_kind in ["physics", "baseline"]:
            path: list[float] = []
            cache: dict[tuple[float, float], tuple[float, float]] = {}

            def obj(x: np.ndarray) -> float:
                key = tuple(np.asarray(x, dtype=float).tolist())
                if key in cache:
                    pred, loss = cache[key]
                else:
                    xx = np.asarray(x, dtype=float).reshape(1, 2)
                    pred = float(predict_uts_mean(s2_models, s1_models, xx, model_kind=model_kind)[0])
                    loss = (pred - target) ** 2
                    cache[key] = (pred, loss)
                path.append(loss)
                return loss

            res = minimize(
                obj,
                x0=x0.copy(),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 50, "maxfun": 20, "ftol": 1e-7},
            )
            x_star = np.asarray(res.x, dtype=float)
            pred_star = float(
                predict_uts_mean(s2_models, s1_models, x_star.reshape(1, 2), model_kind=model_kind)[0]
            )
            summary.append(
                {
                    "group_id": int(row["group_id"]),
                    "model": model_kind,
                    "target_UTS": target,
                    "P0": float(x0[0]),
                    "v0": float(x0[1]),
                    "P_star": float(x_star[0]),
                    "v_star": float(x_star[1]),
                    "pred_UTS_star": pred_star,
                    "abs_dev_target": abs(pred_star - target),
                    "input_shift_from_x0": float(np.linalg.norm(x_star - x0)),
                    "input_shift_from_group_point": float(np.linalg.norm(x_star - start_point)),
                    "n_iters": int(res.nit),
                    "success": bool(res.success),
                    "final_loss": float(res.fun),
                }
            )
            histories[model_kind][int(row["group_id"])] = path if path else [float(res.fun)]
            path_vals = histories[model_kind][int(row["group_id"])]
            for eval_idx, loss_val in enumerate(path_vals, start=1):
                convergence_rows.append(
                    {
                        "group_id": int(row["group_id"]),
                        "model": model_kind,
                        "target_UTS": target,
                        "eval_idx": int(eval_idx),
                        "loss": float(loss_val),
                    }
                )

    out_df = pd.DataFrame(summary).sort_values(by=["group_id", "model"]).reset_index(drop=True)
    out_df.to_csv("outputs/step3_metrics_summary.csv", index=False)
    pd.DataFrame(convergence_rows).to_csv("outputs/step3_convergence_data.csv", index=False)

    # Convergence plot by target group: one subplot per test group,
    # physics and baseline overlaid for direct comparison.
    import matplotlib.pyplot as plt

    group_ids = sorted(test_rows["group_id"].astype(int).tolist())
    group_target_map = {int(r["group_id"]): float(r["uts_mean"]) for _, r in test_rows.iterrows()}

    n_groups = len(group_ids)
    ncols = min(3, max(1, n_groups))
    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, gid in enumerate(group_ids):
        ax = axes[i]
        p_path = np.asarray(histories["physics"][gid], dtype=float)
        b_path = np.asarray(histories["baseline"][gid], dtype=float)

        ax.plot(np.arange(1, len(p_path) + 1), p_path, color="tab:blue", linewidth=1.8, label="physics")
        ax.plot(np.arange(1, len(b_path) + 1), b_path, color="tab:orange", linewidth=1.8, label="baseline")
        ax.set_yscale("log")
        ax.set_xlabel("Objective evaluations")
        ax.set_title(f"group {gid} | target={group_target_map[gid]:.1f}")
        ax.legend(fontsize=8)

    for j in range(n_groups, len(axes)):
        fig.delaxes(axes[j])

    if n_groups > 0:
        axes[0].set_ylabel("Loss = (pred UTS - target)^2")
    fig.tight_layout()
    fig.savefig("outputs/step3_convergence.png", dpi=180)
    plt.close(fig)

    # Console summary.
    grp = out_df.groupby("model")[["abs_dev_target", "input_shift_from_x0", "final_loss"]].mean()
    print(
        "  Step3 mean metrics | "
        f"physics abs_dev={grp.loc['physics', 'abs_dev_target']:.4f}, "
        f"baseline abs_dev={grp.loc['baseline', 'abs_dev_target']:.4f}"
    )
    return out_df
