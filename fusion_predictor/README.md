# fusion_predictor

`fusion_predictor` is a Python machine learning project for estimating fusion experiment neutron yield from plasma operating conditions such as density, temperature, and confinement time. It pairs a practical regression pipeline with a small Lawson criterion utility, so you can compare data-driven predictions with a simple physics-based ignition check.

## Project Overview

The repository ingests a fusion experiment CSV, normalizes common column names, validates and engineers features, trains multiple regression models, and saves the selected production model along with evaluation artifacts. Training now requires an explicit dataset choice: provide `--dataset-path` for a real CSV, or pass `--allow-synthetic` to generate demo data intentionally.

## Features

- Predicts `neutron_yield` from plasma and machine operating conditions.
- Requires an explicit training data source: `--dataset-path` for a real CSV or `--allow-synthetic` for generated demo data.
- Standardizes input columns through alias mapping and temperature normalization.
- Removes duplicate rows, fails fast on invalid physics inputs, and can aggregate time-resolved shots into shot-level records when grouping data is available.
- Engineers physics-inspired features such as `triple_product`, `lawson_ratio`, `density_temp`, `density_tau`, `purity_weighted_density`, and `tau_E_ipb98_s`.
- Excludes configured leakage-style columns from the training feature set.
- Compares multiple regressors, selects the winning model family by cross-validation metrics, reports a true holdout evaluation, and then refits the winner on the full prepared dataset before saving `best_model.joblib`.
- Produces metrics, feature-importance reports, residual plots, physics mismatch flags, and training metadata.
- Supports single-case CLI inference, deriving `ne_20` from `fuel_density_m3` when omitted and rejecting contradictory `ne_20` inputs when supplied.
- Includes unit tests for Lawson calculations and temperature conversions.

## Repository Structure

```text
fusion_predictor/
├── config.py
├── features.py
├── lawson.py
├── train_model.py
├── requirements.txt
├── tests/
│   ├── conftest.py
│   └── test_lawson.py
└── data/
    ├── raw/
    │   └── synthetic_nuclear_fusion_experiment.csv
    └── processed/
        ├── feature_importance.csv
        ├── fusion_dataset_processed.csv
        ├── metrics.csv
        ├── physics_mismatch_flags.csv
        ├── test_predictions.csv
        ├── training_metadata.json
        ├── models/
        │   └── best_model.joblib
        └── plots/
            ├── hist_gradient_boosting_residuals.png
            └── random_forest_feature_importance.png
```

## Installation

```bash
cd fusion_predictor
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## How to Run Training

Train on an explicit real dataset:

```bash
python3 train_model.py train --dataset-path data/raw/your_dataset.csv
```

Generate synthetic demo data only when you opt in explicitly:

```bash
python3 train_model.py train --allow-synthetic
```

Training will:

- create the required project directories if they do not already exist
- use the CSV passed with `--dataset-path`, or generate synthetic data only when `--allow-synthetic` is set
- audit, normalize, clean, and feature-engineer the dataset
- reject invalid rows instead of silently repairing them
- split the data with `GroupShuffleSplit` when repeated `shot_id` groups exist, otherwise use a standard random split
- train multiple regressors on a log-transformed target
- select the winner by cross-validation metrics (`cv_rmse_mean`, then `cv_mae_mean`)
- evaluate the selected model family on a true holdout split for reporting artifacts
- refit the winning model family on the full prepared dataset before saving `data/processed/models/best_model.joblib`
- save artifacts under `data/processed/`

The command prints a JSON summary containing the output paths, selected model, whether synthetic data was used, and the saved-model fit scope.

## How to Run a Single Prediction

Run training first so the default model and metadata files exist, then use the prediction CLI:

```bash
python3 train_model.py predict \
  --density-m3 1e20 \
  --temperature 15 \
  --temp-unit keV \
  --confinement-time-s 4
```

Optional inputs such as `--fuel-purity`, `--energy-input-mj`, `--pressure-pa`, `--ip-ma`, `--bt-t`, `--r-m`, `--a-m`, `--kappa`, `--ne-20`, `--m-amu`, and `--pin-mw` can also be supplied. If omitted, missing optional feature values are filled from defaults stored in `data/processed/training_metadata.json`.

`ne_20` is treated consistently with density:

- if you omit `--ne-20`, inference derives it as `fuel_density_m3 / 1e20`
- if you supply `--ne-20`, it must agree with `fuel_density_m3 / 1e20` within a small tolerance or prediction fails fast with a clear error

The prediction command returns JSON with these fields:

- `predicted_neutron_yield`
- `triple_product`
- `lawson_ratio`
- `status`
- `model_name`

## Lawson Criterion Utility

Use `lawson.py` for a direct Lawson criterion calculation without running the ML model:

```bash
python3 lawson.py \
  --density-m3 1e20 \
  --temperature 15 \
  --temp-unit keV \
  --confinement-time-s 4
```

The utility accepts temperatures in `keV`, `eV`, or `K` and returns:

- `triple_product`
- `lawson_ratio`
- `status`

## Generated Artifacts

Training writes outputs under `data/processed/`:

| Path | Description |
| --- | --- |
| `data/processed/fusion_dataset_processed.csv` | Deduplicated, validated, and feature-engineered dataset used for modeling |
| `data/processed/metrics.csv` | Cross-validation and holdout metrics for each trained regressor |
| `data/processed/test_predictions.csv` | Held-out predictions, actual values, and residuals for the selected model family |
| `data/processed/feature_importance.csv` | Feature importance values from the fitted random forest model |
| `data/processed/physics_mismatch_flags.csv` | Test cases with high predicted yield but low Lawson ratio |
| `data/processed/training_metadata.json` | Dataset source, feature columns, selection basis, holdout evaluation details, saved-model lifecycle, audit summary, and default feature values for inference |
| `data/processed/models/best_model.joblib` | Serialized production model refit on the full prepared dataset after model selection |
| `data/processed/plots/hist_gradient_boosting_residuals.png` | Residual plot for the current checked-in best model |
| `data/processed/plots/random_forest_feature_importance.png` | Bar chart of top random forest feature importances |

## Testing

Run the test suite with:

```bash
pytest
```

The current tests focus on Lawson criterion calculations, temperature conversion round trips, and basic input validation.
The pipeline tests also cover explicit dataset-source behavior, saved-model refitting, and `ne_20` consistency checks.

## Notes / Limitations

- Synthetic data is useful for demos and pipeline validation, but it is not a substitute for real experimental fusion data. The training CLI only uses it when you pass `--allow-synthetic`.
- The Lawson utility uses a simplified D-T ignition threshold from `config.py` and is best treated as a compact educational or screening tool rather than a full plasma physics simulator.
- Model quality depends on the dataset, feature coverage, and split behavior; holdout artifacts are for reporting, while the saved production model is refit on all prepared rows.
- The prediction CLI expects a trained model and metadata file unless you provide custom `--model-path` and `--metadata-path` values.
- Test coverage is currently stronger for the Lawson helper than for the full end-to-end training pipeline, so ML changes should be validated by rerunning training and reviewing the saved artifacts.
