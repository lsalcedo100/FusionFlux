# FusionFlux

`FusionFlux` is a Python machine learning project for estimating fusion experiment neutron yield from plasma operating conditions such as density, temperature, and confinement time. It pairs a practical regression pipeline with a small Lawson criterion utility, so you can compare data-driven predictions with a simple physics-based ignition check.

## Quickstart

```bash
# 1. Install (editable, with dev tooling) into a virtualenv
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e ".[dev]" -c constraints.txt

# 2. Train on the bundled synthetic demo data (writes the default saved model)
python3 train_model.py train --allow-synthetic

# 3. Predict a single operating point
python3 train_model.py predict --density-m3 1e20 --temperature 15 --temp-unit keV --confinement-time-s 4

# 4. Reproduce the CI quality gate locally
make check        # == ruff check . && mypy . && pytest -q
```

Use `--dataset-path data/raw/your_dataset.csv` instead of `--allow-synthetic` to train on a real CSV. The sections below document every option in detail.

## Project Overview

The repository ingests a fusion experiment CSV, normalizes common column names, validates and engineers features, trains multiple regression models, and saves the selected production model along with evaluation artifacts. Training now requires an explicit dataset choice: provide `--dataset-path` for a real CSV, or pass `--allow-synthetic` to generate demo data intentionally.

## Features

- Predicts `neutron_yield` from plasma and machine operating conditions.
- Requires an explicit training data source: `--dataset-path` for a real CSV or `--allow-synthetic` for generated demo data.
- Standardizes input columns through alias mapping and temperature normalization, and fails fast on bare `temperature` values unless a `temperature_unit` column is present or training is run with an explicit `--assume-temperature-unit`.
- Removes duplicate rows, fails fast on invalid physics inputs, and can aggregate time-resolved shots into shot-level records when grouping data is available.
- Engineers physics-inspired features such as `triple_product`, `lawson_ratio`, `density_temp`, `density_tau`, and `tau_E_ipb98_s`, plus `purity_weighted_density` when `fuel_purity` is available (it is skipped rather than imputed to a constant when purity is absent).
- Excludes configured leakage-style columns from the training feature set.
- Uses a row-targeted grouped holdout when repeated `shot_id` values exist, with an exact bitset-based selector that scales better than naive subset tracking as group counts grow.
- Selects holdout evaluation and explainability features from the training split, then rebuilds the saved production feature schema from the full prepared dataset before refitting the winning model family and saving `best_model.joblib`.
- Persists an explicit, versioned preprocessing contract (column set, feature schema, physics constants and tolerances, plus a hash of that structural description) with each training run and hard-fails inference when the saved contract no longer matches the current runtime contract. Retrain, or bump `PREPROCESSING_CONTRACT_VERSION` when preprocessing semantics change, to invalidate stale artifacts.
- Saves a small wrapper artifact around the trained regressor so even direct `joblib.load(...).predict(...)` usage still enforces preprocessing compatibility and clips negative predictions.
- Produces metrics, feature-importance reports, residual plots, physics mismatch flags with explicit threshold metadata, and training metadata under a per-run artifact directory.
- Supports single-case CLI inference, deriving `ne_20` from `fuel_density_m3` when omitted, rejecting contradictory `ne_20` inputs when supplied, clipping any negative model output back to `0.0`, and exposing explicit default artifact selection modes.
- Supports batch CSV/DataFrame inference, auto-generating row identity columns when omitted, requiring both reserved identity columns together when supplied, stamping each prediction row with the artifact metadata used to score it, and streaming non-grouped CSV scoring directly to disk.
- Includes Lawson and pipeline tests covering preprocessing, training, artifact compatibility, and inference behavior.

## Repository Structure

```text
FusionFlux/
├── artifact_model.py
├── config.py
├── features.py
├── fusionflux_cli.py
├── inference.py
├── lawson.py
├── storage.py
├── train_model.py
├── training.py
├── validation.py
├── Makefile
├── pyproject.toml
├── requirements.txt
├── constraints.txt
├── tests/
│   ├── conftest.py
│   ├── helpers.py
│   ├── test_lawson.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_inference.py
└── data/
    ├── raw/
    │   └── synthetic_nuclear_fusion_experiment.csv
    └── processed/
        ├── latest_training_run.json
        └── runs/
            └── <training_run_id>/
                ├── feature_importance.csv
                ├── fusion_dataset_processed.csv
                ├── metrics.csv
                ├── physics_mismatch_flags.csv
                ├── test_predictions.csv
                ├── training_metadata.json
                ├── models/
                │   └── best_model.joblib
                └── plots/
                    ├── <best_model>_residuals.png
                    └── feature_importance.png
```

## Installation

```bash
cd FusionFlux
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -c constraints.txt -r requirements.txt
```

Or install the project as an editable package (exposes the `fusionflux` command
and pulls in the lint/type/test tooling via the `dev` extra):

```bash
python3 -m pip install -e ".[dev]" -c constraints.txt
```

Then lint, type-check, and test with:

```bash
ruff check .
mypy .
python3 -m pytest -q
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

If your CSV only has a generic `temperature` column and no `temperature_unit`, opt into the assumption explicitly:

```bash
python3 train_model.py train \
  --dataset-path data/raw/your_dataset.csv \
  --assume-temperature-unit keV
```

Additional training controls:

- `--shot-prediction-cutoff-rows N` changes how many early rows per repeated `shot_id` are used when the pipeline aggregates time-resolved shots into one example. The default is `2`.
- `--skip-report-generation` skips residual plots and feature-importance artifacts for faster training runs.

Training will:

- create the required project directories if they do not already exist
- use the CSV passed with `--dataset-path`, or generate synthetic data only when `--allow-synthetic` is set
- audit, normalize, clean, and feature-engineer the dataset
- reject invalid rows instead of silently repairing them
- split the data with a row-targeted grouped holdout when repeated `shot_id` groups exist, otherwise use a standard random split
- choose grouped holdout rows with an exact bitset-based search so larger numbers of shots do not turn split selection into a Python object bottleneck
- train multiple regressors on a log-transformed target
- select the winner by log-space cross-validation metrics (`cv_rmse_log_mean`, then `cv_mae_log_mean`) so selection is not dominated by the few highest-magnitude shots; raw-space `cv_rmse_mean`/`cv_mae_mean` are still reported for interpretability
- choose the holdout evaluation feature schema from the training split
- evaluate the selected model family on a true holdout split for reporting artifacts
- rebuild the saved production feature schema from the full prepared dataset so late-appearing features are not dropped from the refit model
- refit the winning model family on the full prepared dataset before saving `data/processed/runs/<training_run_id>/models/best_model.joblib`
- save artifacts under `data/processed/runs/<training_run_id>/`
- update `data/processed/latest_training_run.json` so inference can discover the most recently trained run first

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

Optional inputs such as `--fuel-purity`, `--energy-input-mj`, `--pressure-pa`, `--ip-ma`, `--bt-t`, `--r-m`, `--a-m`, `--kappa`, `--ne-20`, `--m-amu`, and `--pin-mw` can also be supplied. If omitted, the saved model's preprocessing pipeline imputes missing optional values at prediction time; inference does not back-fill metadata defaults.

By default, prediction loads saved artifacts using `--default-artifact-selection best_compatibility`. That mode prefers the most runtime-compatible artifact even if a newer run exists. You can switch to `--default-artifact-selection newest_compatible` to prefer the newest loadable run instead. You can also pin a specific training run with `--training-run-id`, or pass both `--model-path` and `--metadata-path` explicitly. Explicit artifact selection is strict about runtime compatibility; default artifact selection may accept limited compatible version drift and emit warnings describing what was chosen.

`ne_20` is treated consistently with density:

- if you omit `--ne-20`, inference derives it as `fuel_density_m3 / 1e20` by default
- if you supply `--ne-20`, it only has to agree with `fuel_density_m3 / 1e20` to within an order-unity tolerance; the check is deliberately loose so that genuine electron/ion density divergence (impurities, Z_eff, isotope mix) is accepted while gross unit mistakes still fail fast with a clear error

The prediction command returns JSON with these fields:

- `predicted_neutron_yield`
- `triple_product`
- `lawson_ratio`
- `status`
- `model_name`
- `clipped_negative_prediction`
- `prediction_warnings`

## How to Run Batch Prediction

Score every row in a CSV with the default saved artifact and write a sidecar predictions file:

```bash
python3 train_model.py predict-batch \
  --input-csv data/raw/your_scoring_rows.csv
```

By default this writes `data/raw/your_scoring_rows_predictions.csv`. You can override the destination and optionally provide `--assume-temperature-unit` when the input only has a generic `temperature` column:

```bash
python3 train_model.py predict-batch \
  --input-csv data/raw/your_scoring_rows.csv \
  --output-path data/processed/scored_rows.csv \
  --assume-temperature-unit keV \
  --default-artifact-selection newest_compatible
```

Batch scoring applies the same alias mapping, temperature normalization, optional-field validation, feature engineering, preprocessing-contract checks, and model/runtime compatibility checks as single-case inference. If the input omits `original_row_index` and `raw_csv_row_number`, they are added automatically. If either reserved identity column is supplied on its own, prediction fails fast. Grouped time-series CSV inputs are loaded as a whole file so shot aggregation can see every row in a shot; non-grouped CSVs are streamed chunk-by-chunk and the CLI writes them directly to the output file instead of first accumulating every chunk in memory. You can also pin a specific artifact with `--training-run-id`, or by passing both `--model-path` and `--metadata-path`.

The output CSV retains row identity columns and appends:

- `predicted_neutron_yield`
- `artifact_training_run_id`
- `artifact_model_name`
- `artifact_schema_version`
- `artifact_model_path`
- `artifact_metadata_path`
- `artifact_created_at_utc`

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

## Simulation Assumptions

- All required physics inputs are finite numbers. Most must be strictly positive, `fuel_purity` must be between `0` and `1` inclusive, `neutron_yield` must be non-negative, and `a_m` must be smaller than `R_m`.
- Temperatures must be expressible as `keV`, `eV`, or `K`. A generic `temperature` column is only valid when paired with `temperature_unit` or an explicit `--assume-temperature-unit`.
- `ne_20` is derived from `fuel_density_m3 / 1e20` when omitted. When supplied it is kept as an independent value and only checked against `fuel_density_m3 / 1e20` within an order-unity tolerance that catches unit mistakes without rejecting physical electron/ion density divergence. The reference density and tolerances are centralized in `config.py`.
- Repeated `shot_id` rows are treated as time-resolved measurements from the same shot. When timestamps are present, the pipeline collapses each shot to one example using only the first `shot_prediction_cutoff_rows` rows, medians for most numeric covariates, and the cutoff-row value for `neutron_yield`, `time_s`, and `time_ms`.
- Missing optional machine parameters are allowed and are median-imputed by the model preprocessor during training and inference.
- The engineered `tau_E_ipb98_s` feature is only computed when `Ip_MA`, `Bt_T`, `R_m`, `a_m`, `kappa`, and `Pin_MW` are available. `M_amu` defaults to `2.5` inside that proxy when it is missing.
- `lawson_ratio` uses the D-T ignition threshold configured in `config.py` through `LAWSON_DT_IGNITION` and is best treated as a compact screening calculation, not a full plasma simulator.
- `physics_mismatch_flags.csv` is threshold-driven, not a direct physics verdict. By default it flags holdout rows whose predicted yield lands in the top 10 percent of that holdout set and whose `lawson_ratio` is below `LOW_LAWSON_RATIO_THRESHOLD`. Both the threshold mode and the concrete threshold used are written into the CSV and the training metadata.
- Default inference only loads artifacts whose metadata, preprocessing contract, and runtime versions are compatible with the current environment. `best_compatibility` may prefer an older exact-match run over a newer drifted run, while `newest_compatible` does the reverse when both are loadable.
- When `--allow-synthetic` is used, synthetic data generation still assumes the sampling ranges in `create_synthetic_dataset`, fixed six-row shot blocks, and a hand-crafted yield signal based on Lawson ratio, input energy, pressure, the IPB98 proxy, fuel purity, and Gaussian noise.

## Generated Artifacts

Training writes outputs under `data/processed/runs/<training_run_id>/`, and `data/processed/latest_training_run.json` records the most recently trained run as a discovery hint. Residual and feature-importance artifacts are only written when report generation is enabled:

| Path | Description |
| --- | --- |
| `data/processed/latest_training_run.json` | Manifest written by training for the most recent run, including its model and metadata paths |
| `data/processed/runs/<training_run_id>/fusion_dataset_processed.csv` | Deduplicated, validated, and feature-engineered dataset used for modeling |
| `data/processed/runs/<training_run_id>/metrics.csv` | Cross-validation and holdout metrics for each trained regressor, in both log space (`cv_rmse_log_mean`, `holdout_rmse_log`, used for selection) and raw yield space (`cv_rmse_mean`, `holdout_rmse`, reported for interpretability) |
| `data/processed/runs/<training_run_id>/test_predictions.csv` | Held-out predictions, actual values, residuals, `shot_id`, and preserved source-row identity columns |
| `data/processed/runs/<training_run_id>/feature_importance.csv` | Cross-validated feature-importance values for the selected model family |
| `data/processed/runs/<training_run_id>/physics_mismatch_flags.csv` | Holdout rows flagged by the configured mismatch rule, including the threshold mode and concrete thresholds used |
| `data/processed/runs/<training_run_id>/training_metadata.json` | Dataset source, feature schema details, preprocessing contract hash, selection basis, holdout evaluation details, mismatch flagging details, and saved-model lifecycle metadata |
| `data/processed/runs/<training_run_id>/models/best_model.joblib` | Serialized `FusionFluxModelArtifact` wrapper containing the production model, embedded preprocessing contract, and prediction guardrails |
| `data/processed/runs/<training_run_id>/plots/<best_model>_residuals.png` | Residual plot for the selected model family |
| `data/processed/runs/<training_run_id>/plots/feature_importance.png` | Bar chart of the top feature-importance values for the selected model family |

## Testing

Run the test suite with:

```bash
.venv/bin/python -m pytest -q
```

or run the full lint/type/test gate with `make check`. The suite is split by concern into `tests/test_preprocessing.py`, `tests/test_training.py`, and `tests/test_inference.py` (plus `tests/test_lawson.py`), with shared fixtures in `tests/conftest.py` and shared stubs/builders in `tests/helpers.py`. It covers Lawson calculations, temperature conversions, preprocessing and validation rules, grouped-shot aggregation, training split behavior, training artifact cleanup, preprocessing-contract compatibility checks, negative prediction clipping, and single/batch inference edge cases. CI runs the same gate on Python 3.9–3.12.

`test_committed_artifact_manifest_supports_relocation` needs a locally trained artifact under `data/processed/` (gitignored); it skips automatically on a fresh clone until you run training once.

## Notes / Limitations

- Synthetic data is useful for demos and pipeline validation, but it is not a substitute for real experimental fusion data. The training CLI only uses it when you pass `--allow-synthetic`.
- The Lawson utility uses a simplified D-T ignition threshold from `config.py` and is best treated as a compact educational or screening tool rather than a full plasma physics simulator.
- Model quality depends on the dataset, feature coverage, and split behavior; holdout artifacts are for reporting, while the saved production model is refit on all prepared rows.
- The prediction CLIs expect a trained model and metadata file unless you provide custom `--model-path` and `--metadata-path` values. They validate the saved preprocessing contract against the current runtime code before scoring. Explicit artifact selection requires exact recorded runtime versions, while default artifact selection may accept limited compatible drift with warnings.
- Batch CSV prediction only streams non-grouped inputs. Grouped time-series inputs are read as a whole file so shot-level aggregation can see every row for a shot.
- The strict preprocessing contract is intentional. In this repo, silent feature drift is more dangerous than the inconvenience of regenerating artifacts, because the goal is fail-fast behavior around physics results. The contract is an explicit, versioned structural description (columns, feature schema, physics constants and tolerances); it deliberately does not fingerprint function source or bytecode, since that broke on harmless reformatting and forced spurious retrains. Bump `PREPROCESSING_CONTRACT_VERSION` in `features.py` whenever you change preprocessing semantics.
- The test suite exercises many pipeline paths, but ML changes should still be validated by rerunning training and reviewing the saved artifacts.

## Module Ownership

- `training.py` owns training orchestration, artifact writing, holdout evaluation, and report generation.
- `inference.py` owns runtime loading, artifact selection, single prediction, and batch prediction.
- `train_model.py` is kept as a compatibility facade and CLI entrypoint so existing commands and imports continue to work.
