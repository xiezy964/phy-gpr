# phy-gpr

Hierarchical physics-informed Gaussian process regression for yield strength prediction in laser powder bed fusion.

## Overview

This repository contains a research pipeline for modeling the relationship between laser powder bed fusion process parameters and mechanical response using Gaussian process regression (GPR).

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

Step 2 predicts yield strength `YS` with two model families:

- a baseline GP using only process parameters
- a physics-informed GP that propagates Step 1 descriptor uncertainty into the final prediction

Step 3 performs inverse optimization over `(P, v)` to match held-out target `YS` values.

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

This script is intended to run:

1. data preparation
2. Step 1 dual GPR for `d` and `phi`
3. Step 2 physics-informed GPR for `YS`
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

These outputs cover:

- train and test metrics
- prediction plots
- process-parameter response maps
- optimization convergence traces
- serialized model artifacts

