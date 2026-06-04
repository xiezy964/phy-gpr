from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import RANDOM_SEED, ensure_output_dirs

EXCEL_PATH = "data/AlSi10Mg PSP feature table.xlsx"
N_ROWS = 32
N_SAMPLES_PER_ROW = 100


def _load_raw_rows(path: str = EXCEL_PATH, n_rows: int = N_ROWS) -> pd.DataFrame:
    # This source sheet has four header rows; row index 4 starts data.
    raw = pd.read_excel(path, header=None)
    data = raw.iloc[4 : 4 + n_rows].copy()

    # Column indices discovered from sheet header blocks.
    # 2: Power, 3: Speed, 14/16: EBSD grain mean/std, 32/33: X-ray porosity mean/std,
    # 38/39: UTS mean/std.
    mapped = pd.DataFrame(
        {
            "group_id": np.arange(n_rows, dtype=int),
            "P": pd.to_numeric(data.iloc[:, 2], errors="coerce"),
            "v": pd.to_numeric(data.iloc[:, 3], errors="coerce"),
            "d_mean": pd.to_numeric(data.iloc[:, 14], errors="coerce"),
            "d_std": pd.to_numeric(data.iloc[:, 16], errors="coerce"),
            "phi_mean": pd.to_numeric(data.iloc[:, 32], errors="coerce"),
            "phi_std": pd.to_numeric(data.iloc[:, 33], errors="coerce"),
            "uts_mean": pd.to_numeric(data.iloc[:, 40], errors="coerce"),
            "uts_std": pd.to_numeric(data.iloc[:, 41], errors="coerce"),
        }
    )

    mapped = mapped.dropna().reset_index(drop=True)
    mapped["group_id"] = np.arange(len(mapped), dtype=int)
    if len(mapped) < n_rows:
        raise ValueError(f"Expected {n_rows} valid rows, found {len(mapped)} after cleaning.")
    return mapped.iloc[:n_rows].copy()


def _augment(df_rows: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for _, row in df_rows.iterrows():
        d_std = max(float(row["d_std"]), 1e-6)
        phi_std = max(float(row["phi_std"]), 1e-6)
        uts_std = max(float(row["uts_std"]), 1e-6)

        d_samples = rng.normal(float(row["d_mean"]), d_std, size=N_SAMPLES_PER_ROW)
        # Grain size must stay positive.
        d_samples = np.clip(d_samples, 1e-6, None)
        phi_samples = rng.normal(float(row["phi_mean"]), phi_std, size=N_SAMPLES_PER_ROW)
        # Porosity in percent is physically non-negative.
        phi_samples = np.clip(phi_samples, 0.0, None)
        uts_samples = rng.normal(float(row["uts_mean"]), uts_std, size=N_SAMPLES_PER_ROW)

        for d, phi, uts in zip(d_samples, phi_samples, uts_samples):
            records.append(
                {
                    "group_id": int(row["group_id"]),
                    "P": float(row["P"]),
                    "v": float(row["v"]),
                    "d": float(d),
                    "phi": float(phi),
                    "UTS": float(uts),
                }
            )

    return pd.DataFrame.from_records(records)


def _group_split(
    df_aug: pd.DataFrame, seed: int = RANDOM_SEED, test_group_count: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    uniq_groups = np.array(sorted(df_aug["group_id"].unique().tolist()), dtype=int)
    if test_group_count <= 0 or test_group_count >= len(uniq_groups):
        raise ValueError(
            f"test_group_count must be in [1, {len(uniq_groups)-1}], got {test_group_count}."
        )
    rng = np.random.default_rng(seed)
    test_groups = set(rng.choice(uniq_groups, size=test_group_count, replace=False).tolist())
    idx = np.arange(len(df_aug))
    test_mask = df_aug["group_id"].isin(test_groups).to_numpy()
    train_idx = idx[~test_mask]
    test_idx = idx[test_mask]
    return train_idx, test_idx


def prepare(seed: int = RANDOM_SEED, test_group_count: int = 5) -> dict:
    ensure_output_dirs()
    row_df = _load_raw_rows()
    aug_df = _augment(row_df, seed=seed)
    train_idx, test_idx = _group_split(aug_df, seed=seed, test_group_count=test_group_count)

    train_df = aug_df.iloc[train_idx].reset_index(drop=True)
    test_df = aug_df.iloc[test_idx].reset_index(drop=True)
    test_groups = sorted(test_df["group_id"].unique().tolist())
    train_groups = sorted(train_df["group_id"].unique().tolist())
    train_raw_rows = row_df[row_df["group_id"].isin(train_groups)].copy().reset_index(drop=True)
    test_raw_rows = row_df[row_df["group_id"].isin(test_groups)].copy().reset_index(drop=True)

    return {
        "raw_rows": row_df,
        "train_raw_rows": train_raw_rows,
        "test_raw_rows": test_raw_rows,
        "augmented": aug_df,
        "train": train_df,
        "test": test_df,
        "train_groups": train_groups,
        "test_groups": test_groups,
    }
