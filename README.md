# phy-gpr

Hierarchical physics-informed Gaussian process regression for yield strength prediction in laser powder bed fusion.

## Overview

This repository contains a small research pipeline for modeling the relationship between laser powder bed fusion process parameters and mechanical response using Gaussian process regression (GPR).

The workflow is organized into three main stages:

1. Learn intermediate structure descriptors from process parameters.
2. Use those descriptors inside a physics-informed GPR model for strength prediction.
3. Solve an inverse design problem to search for process parameters that match target strength values.

An additional ablation script is included to compare baseline and physics-informed variants.

## What The Pipeline Does

The project uses laser power `P` and scan speed `v` as inputs.

Step 1 predicts latent or intermediate material descriptors:

- grain size `d`
- porosity `phi`

Step 2 predicts ultimate tensile strength `UTS` with two model families:

- a baseline GP using only process parameters
- a physics-informed GP that propagates Step 1 descriptor uncertainty into the final prediction

Step 3 performs inverse optimization over `(P, v)` to match held-out target `UTS` values.

Step 4 runs ablations to compare:

- untuned baseline GP
- tuned baseline GP
- full physics-informed GP
- tuned physics-informed GP
- Hall-Petch-only variant
- porosity-only variant

## Data

The current pipeline reads source data from:

- `data/AlSi10Mg PSP feature table.xlsx`

The data preparation code currently:

- reads 32 valid rows from the spreadsheet
- extracts process parameters and summary statistics
- uses Gaussian sampling to expand each original row into 100 synthetic samples
- splits train and test sets by original group id, not by individual augmented rows

Important notes:

- `data/Ti-6Al-4V PSP feature table.xlsx` is present in the repository but is not used by the current default pipeline
- the spreadsheet layout is assumed by fixed column indices in [src/data_prep.py](/Users/xiezy/Documents/github/phy-gpr/src/data_prep.py:1)

## Repository Layout

```text
phy-gpr/
├── data/
│   ├── AlSi10Mg PSP feature table.xlsx
│   └── Ti-6Al-4V PSP feature table.xlsx
├── outputs/
├── run_pipeline.py
├── run_ablations.py
└── src/
    ├── data_prep.py
    ├── step1_gpr.py
    ├── step2_gpr.py
    ├── step3_optim.py
    ├── step4_ablations.py
    └── utils.py
```

## Environment

This project is written in Python and uses common scientific computing packages.

At minimum, you should install dependencies equivalent to:

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl
```

If you want reproducible isolated setup, a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib openpyxl
```

## How To Run

### Main Pipeline

```bash
python run_pipeline.py
```

Optional argument:

```bash
python run_pipeline.py --test-groups 6
```

This script is intended to run:

1. data preparation
2. Step 1 dual GPR for `d` and `phi`
3. Step 2 physics-informed GPR for `UTS`
4. Step 3 inverse optimization

### Ablation Study

```bash
python run_ablations.py
```

Optional arguments:

```bash
python run_ablations.py --test-groups 6 --baseline-restarts 10 --physics-restarts 10
```

## Outputs

Generated artifacts are written to `outputs/`.

Typical files include:

- `step1_metrics.csv`
- `step1_predictions.png`
- `step1_errors.png`
- `step1_model.pth`
- `step1_scalers.pkl`
- `step2_metrics.csv`
- `step2_predictions.png`
- `step2_errors.png`
- `step2_pv_maps.png`
- `step2_pv_map_data.csv`
- `step2_baseline_pv_maps.png`
- `step2_baseline_pv_map_data.csv`
- `step2_model.pkl`
- `step2_scalers.pkl`
- `step3_metrics_summary.csv`
- `step3_convergence_data.csv`
- `step3_convergence.png`
- `step4_ablation_metrics.csv`

These outputs cover:

- train and test metrics
- prediction plots
- process-parameter response maps
- optimization convergence traces
- serialized model artifacts

## Method Notes

The implementation combines data-driven regression with simple physics-inspired feature construction:

- Step 1 estimates distributions over grain size and porosity
- Step 2 builds physics-related latent terms from those uncertain quantities
- uncertainty is propagated into the final GP rather than using only point estimates
- Step 3 treats the trained predictor as a surrogate model for inverse search

From the current code, the physics-informed formulation is primarily implemented in [src/step2_gpr.py](/Users/xiezy/Documents/github/phy-gpr/src/step2_gpr.py:1).

## Reproducibility

The code uses a fixed random seed defined in [src/utils.py](/Users/xiezy/Documents/github/phy-gpr/src/utils.py:1):

- `RANDOM_SEED = 42`

Train/test splitting and sample augmentation are both seed-controlled.

## Current Limitations

There are a few repo-specific caveats worth knowing before you run it:

- `run_pipeline.py` currently imports `src.report.write_technical_report`, but `src/report.py` is not present in this repository
- because of that missing module, `python run_pipeline.py` will fail unless that report module is restored or the call is removed
- the current README describes the intended workflow and the available source files as they exist now
- the project does not yet include a pinned `requirements.txt`, `pyproject.toml`, or environment file

## Suggested Next Improvements

- add a `requirements.txt` or `pyproject.toml`
- restore or remove the missing report-generation module
- document the exact spreadsheet schema in more detail
- include a short example of expected metric ranges or figures

## License

This repository already includes a [LICENSE](/Users/xiezy/Documents/github/phy-gpr/LICENSE:1) file. See it for usage terms.
