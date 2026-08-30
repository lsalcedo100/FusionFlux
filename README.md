# FusionFlux

`FusionFlux` is a Python machine learning project for fusion plasma performance modeling. It contains two independent pipelines plus a small physics utility:

- **Neutron yield** (`train_model.py`): estimates fusion experiment `neutron_yield` from plasma operating conditions such as density, temperature, and confinement time. Ships with a synthetic demo dataset and a strict, versioned artifact contract.
- **Energy confinement time** (`hdb5.py`): trains on the *real* ITPA Global H-mode Confinement Database (STD5), predicting thermal confinement time `TAUTH` and scoring every model against the analytic IPB98(y,2) scaling law as a physics baseline.
- **Lawson criterion** (`lawson.py`): a standalone triple-product and ignition-ratio calculation.

Together they let you compare data-driven predictions against simple physics-based checks, on both synthetic and real experimental data.

## Results

Full writeup with tables, limitations and reproduction steps: **[results/RESULTS.md](results/RESULTS.md)**. Regenerate with `python3 analysis_scaling_law.py` and `python3 analysis_extrapolation.py`.

Measured on the real ITPA H-mode confinement database (HDB5 STD5): 6228 quasi-stationary time slices from 4471 discharges across 18 tokamaks. No synthetic data is used in any reported result.

### The headline: a learned model beats the published scaling law, and that result does not survive contact with a new machine

![Interpolation against extrapolation](results/extrapolation.png)

Under cross-validation grouped by discharge, a random forest cuts RMSLE 41% below the analytic IPB98(y,2) law. But grouped CV holds out *shots*, so every machine in the held-out fold is also in the training fold. Hold out an entire tokamak instead, train on the other 12 and predict the 13th, and **the ranking of the three blind models reverses exactly**:

| model | CV, by discharge | leave-one-tokamak-out | ratio | CV rank | LOMO rank |
|---|---|---|---|---|---|
| random forest | 0.128 | 0.465 | **3.6x worse** | 1 | 5 |
| histogram gradient boosting | 0.130 | 0.359 | 2.8x worse | 2 | 4 |
| ridge, log-quadratic (control) | 0.158 | 0.300 | 1.9x worse | 3 | 3 |
| ridge, log-linear | 0.181 | 0.214 | 1.2x worse | 4 | 2 |
| IPB98(y,2), analytic (fitted on this database, not blind) | 0.199 | 0.188 | unchanged | 5 | 1 |

Both columns use the same nine features and the same models; only the split changes. The best model in this repository by cross-validation is the worst of the three on a machine it has not seen, and its 41% margin turns out to measure how much of JET is predictable from the rest of JET.

**The failure has a mechanism, and it is measurable.** The random forest's per-machine error correlates with how far that machine sits outside the training data at rho = **+0.85**; the log-linear power law's does not, at rho = **-0.06**. And when JET is held out, 48% of its rows lie above the highest confinement time in the remaining 12 machines: a tree ensemble averages training targets, so **no tree in the forest can output those values at all**, whatever the features say. That bound is asserted directly in `tests/test_extrapolation.py`.

**And it is the constraint that matters, not just the ability to extrapolate.** The obvious objection is that ridge only wins because it is the one model in the zoo that is not bounded by its training range. The `ridge_log_quadratic` control tests exactly that: it carries curvature and every pairwise log interaction, so it is much more flexible than plain ridge, but being a polynomial it still extrapolates without bound. It lands in between (0.300, 1.9x), and every column of the table above turns out to be monotone in flexibility. Unbounded extrapolation buys something real, but most of the advantage comes from the constrained power-law form. This is why the field still uses power laws it knows fit worse.

### The linear algebra underneath

![Singular value spectrum and disagreement decomposition](results/singular_value_spectrum.png)

**The model's own feature matrix is rank deficient by two, and this audit found it.** Standardized, the ten log features have rank 8. Two exact dependencies, each confirmed by projection onto the null space at a residual of order 1e-16: minor radius is derived as `a = eps * R`, and the IPB98 prior is a fixed log-linear combination of the other eight features. That second one means a published physics scaling, added as a feature, contributes exactly nothing to a log-linear model, however much it looks like added knowledge. Nothing crashed because `scipy.linalg.lstsq` inverts through the SVD pseudoinverse and silently returns the minimum-norm member of a two-parameter family.

**Refitting IPB98(y,2) from the database disagrees with the published exponents almost entirely where the data is blind.** Solving three ways from scratch (Cholesky on the normal equations, QR, SVD, agreeing to 8e-13) gives Ip 1.08 against 0.93 and R 1.58 against 1.97, while P and Bt come back essentially exactly. Decomposing that difference along the singular directions of the design matrix: **77% of it lies in the single weakest direction, which carries 0.3% of the matrix's variance**, and the three strongest directions carry 82% of the variance while accounting for 0.75% of the disagreement. That weak direction is plasma current traded against machine size, which is structurally hard to resolve because tokamaks are not designed to vary the two independently.

## Quickstart

```bash
# 1. Install (editable, with dev tooling) into a virtualenv
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e ".[dev]" -c constraints.txt

# 2. Train on freshly generated synthetic demo data (writes the default saved model)
python3 train_model.py train --allow-synthetic

# 3. Predict a single operating point
python3 train_model.py predict --density-m3 1e20 --temperature 15 --temp-unit keV --confinement-time-s 4

# 4. Reproduce the CI quality gate locally
make check        # == ruff check . && mypy . && pytest -q
```

Use `--dataset-path data/raw/your_dataset.csv` instead of `--allow-synthetic` to train on a real CSV. The sections below document every option in detail.

For the real-data confinement-time pipeline instead:

```bash
python3 hdb5.py train --download-if-missing   # fetches ITPA HDB5 STD5, trains, reports vs IPB98(y,2)
```

## Project Overview

Everything from here through [Generated Artifacts](#generated-artifacts) documents the **neutron-yield** pipeline. The real-data confinement-time pipeline is documented separately under [Confinement Time on Real HDB5 Data](#confinement-time-on-real-hdb5-data).

The neutron-yield pipeline ingests a fusion experiment CSV, normalizes common column names, validates and engineers features, trains multiple regression models, and saves the selected production model along with evaluation artifacts. Training now requires an explicit dataset choice: provide `--dataset-path` for a real CSV, or pass `--allow-synthetic` to generate demo data intentionally.

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
- Publishes each training run atomically: artifacts are staged under a hidden directory and renamed into place only after the run completes.
- Ships a `fusionflux` console script (installed with the package) that is equivalent to `python3 train_model.py` for every command below.
- Includes Lawson and pipeline tests covering preprocessing, training, artifact compatibility, and inference behavior.

## Repository Structure

```text
FusionFlux/
├── artifact_model.py          # saved-model wrapper with preprocessing + clipping guardrails
├── config.py                  # paths, column config, physics constants and tolerances
├── features.py                # alias mapping, validation, feature engineering, contract
├── fusionflux_cli.py          # argparse CLI behind the `fusionflux` console script
├── hdb5.py                    # real-data ITPA HDB5 confinement-time pipeline + CLI
├── inference.py               # single/batch prediction flow, public inference API
├── inference_artifacts.py     # artifact schema, metadata parsing, run-manifest writers
├── inference_selection.py     # artifact discovery, default selection, loading
├── lawson.py                  # standalone Lawson criterion utility
├── analysis_extrapolation.py  # Result 4: leave-one-tokamak-out study and figure
├── analysis_scaling_law.py    # Results 1 to 3: rank audit, IPB98 refit, conditioning
├── scaling_law.py             # from-scratch least squares; fits/audits scaling laws
├── storage.py                 # atomic file writes and JSON/CSV helpers
├── train_model.py             # compatibility facade and CLI entrypoint
├── training.py                # training orchestration and holdout evaluation
├── training_artifacts.py      # per-run path layout, staged write, atomic publish
├── training_registry.py       # preprocessor and candidate model factories
├── training_reports.py        # residual and feature-importance plots
├── training_split.py          # random and grouped holdout / CV split selection
├── validation.py              # physics input validation primitives
├── Makefile
├── pyproject.toml
├── requirements.txt
├── constraints.txt
├── LICENSE
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       └── ci.yml
├── tests/
│   ├── conftest.py
│   ├── helpers.py
│   ├── test_hdb5.py
│   ├── test_lawson.py
│   ├── test_preprocessing.py
│   ├── test_extrapolation.py
│   ├── test_scaling_law.py
│   ├── test_training.py
│   └── test_inference.py
└── data/
    ├── raw/
    │   ├── synthetic_nuclear_fusion_experiment.csv   # sample/reference copy only
    │   └── hdb5_std5.csv            # not committed; fetched via `python3 hdb5.py download`
    └── processed/
        ├── latest_training_run.json
        ├── hdb5_confinement/
        │   ├── confinement_model.joblib
        │   ├── confinement_metrics.csv
        │   └── confinement_metadata.json
        └── runs/
            └── <training_run_id>/
                ├── feature_importance.csv
                ├── fusion_dataset_processed.csv
                ├── synthetic_training_input.csv   # only for --allow-synthetic runs
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
- use the CSV passed with `--dataset-path`, or generate synthetic data only when `--allow-synthetic` is set (each `--allow-synthetic` run generates a fresh dataset into its own run directory as `synthetic_training_input.csv` rather than reading the sample copy in `data/raw/`)
- audit, normalize, clean, and feature-engineer the dataset
- reject invalid rows instead of silently repairing them
- split the data with a row-targeted grouped holdout when repeated `shot_id` groups exist, otherwise use a standard random split
- choose grouped holdout rows with an exact bitset-based search so larger numbers of shots do not turn split selection into a Python object bottleneck
- train the candidate regressors from `training_registry.py` (`baseline` median dummy, `random_forest`, `hist_gradient_boosting`) on a log-transformed target
- select the winner by log-space cross-validation metrics (`cv_rmse_log_mean`, then `cv_mae_log_mean`) so selection is not dominated by the few highest-magnitude shots; raw-space `cv_rmse_mean`/`cv_mae_mean` are still reported for interpretability
- choose the holdout evaluation feature schema from the training split
- evaluate the selected model family on a true holdout split for reporting artifacts
- rebuild the saved production feature schema from the full prepared dataset so late-appearing features are not dropped from the refit model
- refit the winning model family on the full prepared dataset before saving `data/processed/runs/<training_run_id>/models/best_model.joblib`
- write every artifact into a hidden `.staging` directory and rename the run into `data/processed/runs/<training_run_id>/` only once it is complete, so a crash mid-run never leaves a half-written run for the inference loader to discover
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

## Confinement Time on Real HDB5 Data

Everything above models `neutron_yield`, and its bundled dataset is synthetic. `hdb5.py` is a separate, self-contained pipeline that trains on **real experimental data**: the ITPA Global H-mode Confinement Database, standard analysis set STD5 (version 5.2.3), published on the [Open Science Framework](https://osf.io/drwcq). Each row is a quasi-stationary time slice from a real tokamak discharge.

The dataset is not redistributed in this repository. Fetch it once:

```bash
python3 hdb5.py download
```

This writes `data/raw/hdb5_std5.csv` (gitignored). Pass `--overwrite` to re-download. If you use this dataset, cite Verdoolaege et al., *Nucl. Fusion* **61** 076006 (2021).

### Train

```bash
python3 hdb5.py train --download-if-missing
```

Options: `--dataset-path` to point at your own copy of the STD5 CSV, `--cv-folds` to change the number of grouped cross-validation folds (default `5`), and `--download-if-missing` to fetch the default dataset first if it is not present.

Training cleans and maps the raw HDB5 columns to canonical names, cross-validates the model zoo grouped by discharge, refits the winner on all rows, and writes three artifacts under `data/processed/hdb5_confinement/`:

| Path | Description |
| --- | --- |
| `confinement_model.joblib` | Serialized `ConfinementArtifact` (fitted pipeline, feature schema, selected model name) that predicts in log space and exponentiates back to seconds |
| `confinement_metrics.csv` | Per-model `cv_rmsle`, `cv_r2_log`, and `cv_mae_s`, with `is_selected` and `is_physics_baseline` flags |
| `confinement_metadata.json` | Dataset source, row/group counts, tokamak list, feature schema, selected model, and its scores next to the IPB98(y,2) baseline including a `beats_physics_baseline` verdict |

The command prints that metadata as JSON.

### Evaluate

Cross-validate the whole model zoo and print a comparison report without saving artifacts:

```bash
python3 hdb5.py evaluate --cv-folds 5
```

### Extrapolate to an Unseen Machine

`evaluate` holds out discharges, so every tokamak in the held-out fold is also in the training fold. That measures interpolation. To measure the case a scaling law exists for, hold out a whole device:

```bash
python3 hdb5.py extrapolate                       # train on 12 machines, predict the 13th, rotate
python3 hdb5.py extrapolate --include-controls    # add ridge_log_quadratic (see Result 4d)
python3 hdb5.py extrapolate --output-dir results  # also write the per-machine and summary CSVs
```

Machines with fewer than `--min-rows` (default 30) held-out rows are skipped, since their RMSLE would not mean anything. The analytic IPB98(y,2) row is reported as a reference rather than a blind baseline: its exponents were fitted on this database, held-out machine included.

By default `log_ipb98y2_tau_s` is dropped from the feature set for this command, for the same reason. It is a fixed log-linear combination of the other features whose coefficients saw the held-out machine, so keeping it leaks. `--keep-ipb98-feature` restores it if you want to measure that leak.

`python3 analysis_extrapolation.py` runs both splits on the one shared feature set, adds the distance diagnostics and the figure, and writes everything under `results/`. See [Result 4](results/RESULTS.md#result-4-the-model-that-wins-on-cross-validation-loses-on-a-new-machine).

### Predict

```bash
python3 hdb5.py predict \
  --ip-ma 2.0 \
  --bt-t 3.0 \
  --ne-line-1e19-m3 5.0 \
  --p-loss-mw 10.0 \
  --r-m 3.0 \
  --kappa 1.7 \
  --inverse-aspect-ratio 0.32 \
  --m-eff-amu 2.0
```

All eight engineering inputs are required; `a_m` is derived as `inverse_aspect_ratio * r_m` exactly as it is during cleaning, so it is not requested. Use `--model-path` to score against a specific saved artifact. The command returns JSON with `predicted_tau_th_s`, `ipb98y2_tau_s`, `model_name`, and `model_path`.

### Modeling Notes and Assumptions

- The target is thermal energy confinement time `TAUTH` in seconds. The confinement time is never used as an input, so there is no target leakage.
- Raw HDB5 columns are mapped to canonical names (`TAUTH`, `IP`, `BT`, `NEL`, `PLTH`, `RGEO`, `KAPPAA`, `EPS`, `MEFF`, plus `TOK` and `SHOT` for identity). Current and field are taken as absolute values, since their sign encodes a direction that is physically irrelevant to confinement.
- Cleaning is strict: any row with a non-finite or non-positive value in the target or any base engineering column is dropped, and an empty result is an error rather than a silent pass.
- Features are the logs of the nine positive engineering inputs plus the log of the analytic IPB98(y,2) prediction, used as a physics prior. All models regress `log(tau)` and predictions are exponentiated back and clipped at `0.0`.
- Cross-validation is `GroupKFold` over `tokamak::shot`, so time slices from the same discharge never straddle the train/test boundary.
- The model zoo is `mean_baseline` (mean dummy), `ridge_loglinear`, `random_forest`, and `hist_gradient_boosting`. The winner is the trainable model with the lowest `cv_rmsle`.
- The analytic `ipb98y2_analytic` scaling law is scored alongside them as a real physics baseline, but is excluded from selection. It answers "did the model actually learn something" against published physics rather than against the mean.
- `ridge_loglinear` uses the SVD solver on purpose: the IPB98 prior and `log_a_m` are exact linear combinations of the other log features, so the design matrix is singular. The richer feature set is kept because the IPB98 prior measurably helps the tree models.
- This pipeline is deliberately independent of the neutron-yield pipeline. It does not use the preprocessing contract, the run-directory layout, or the artifact selection modes described above.

## Fitting Scaling Laws From Scratch

`scaling_law.py` is a library module (no CLI) that treats a confinement scaling law as what it actually is: ordinary least squares on a log design matrix. A power law

```text
tau_E = C * Ip^a1 * Bt^a2 * ne^a3 * P^a4 * R^a5 * eps^a6 * kappa^a7 * M^a8
```

becomes linear in logs, so fitting it is a least-squares problem and every question about the physics becomes a question about that matrix: its rank, its null space, its singular values, its condition number.

The three classical solvers are implemented here by hand, including the triangular substitutions, rather than delegated to scikit-learn, because the point is to show the numerics:

- `solve_lstsq_cholesky` (normal equations)
- `solve_lstsq_qr`
- `solve_lstsq_svd`
- plus `solve_lstsq_ridge`, `ridge_shrinkage_factors`, and `solve_constrained_lstsq` for the regularized and constrained variants

Typical use, against the cleaned HDB5 frame:

```python
import hdb5
from scaling_law import fit_scaling_law, analyze_conditioning, bootstrap_exponents

dataset = hdb5.prepare_dataset()

fit = fit_scaling_law(dataset, target_column="tau_th_s", solver="svd")
print(fit.coefficient, fit.exponents, fit.residual_std_log)

# How the refit exponents line up with the published IPB98(y,2) values.
print(fit.compare_to_published())

# What the design matrix can and cannot determine.
print(fit.conditioning.rank, fit.conditioning.condition_number, fit.conditioning.is_rank_deficient)

# Grouped percentile bootstrap: resample whole discharges, not individual slices.
print(bootstrap_exponents(dataset, "tau_th_s", group_column="group_id", n_resamples=1000))
```

Notes:

- `analyze_conditioning` standardizes the non-constant columns by default, and that matters. On raw physical columns the largest singular value is set by whichever feature carries the biggest units, so the default rank tolerance discards perfectly informative directions and reports a rank deficiency that is an artifact of unit choice. Standardizing first makes the reported rank a statement about collinearity.
- `bootstrap_exponents` accepts `group_column` so a discharge contributing several quasi-stationary time slices is resampled as a unit; row-level resampling would understate the intervals. It returns a frame of `variable`, `fitted`, `ci_low`, `ci_high`, `published_ipb98y2`, and `published_inside_ci`.
- The published IPB98(y,2) coefficient is carried as `0.0562` (ITER Physics Basis) with the commonly rounded `0.056` exposed separately as `IPB98Y2_COEFFICIENT_ROUNDED`. The difference shifts `log C` by 0.0036, far inside the fitted confidence interval, so nothing here depends on the choice.

References: ITER Physics Basis, *Nucl. Fusion* **39** 2175 (1999) for IPB98(y,2); Verdoolaege et al., *Nucl. Fusion* **61** 076006 (2021) for HDB5.

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
| `data/processed/runs/<training_run_id>/synthetic_training_input.csv` | Only written for `--allow-synthetic` runs: the exact generated raw dataset that run trained on, so the run stays reproducible |
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

or run the full lint/type/test gate with `make check` (`ruff check .`, then `mypy .`, then `pytest -q`). The suite is split by concern into `tests/test_preprocessing.py`, `tests/test_training.py`, and `tests/test_inference.py` (plus `tests/test_lawson.py`, `tests/test_hdb5.py`, and `tests/test_scaling_law.py`), with shared fixtures in `tests/conftest.py` and shared stubs/builders in `tests/helpers.py`. It covers Lawson calculations, temperature conversions, preprocessing and validation rules, grouped-shot aggregation, training split behavior, training artifact cleanup, preprocessing-contract compatibility checks, negative prediction clipping, and single/batch inference edge cases. CI runs the same gate on Python 3.9–3.12 with coverage reporting (`pytest -q --cov --cov-report=term-missing`; the measured module list lives under `[tool.coverage.run]` in `pyproject.toml`). The 3.9 job installs against `constraints.txt` to reproduce the tested training and artifact-loading environment, while the newer-Python jobs resolve current releases so the `>=3.9` support claim is actually exercised. Runs use pip caching and cancel superseded runs for the same ref, and Dependabot opens weekly grouped update PRs for the GitHub Actions and pip dependencies.

`test_committed_artifact_manifest_supports_relocation` needs a locally trained artifact under `data/processed/` (gitignored); it skips automatically on a fresh clone until you run training once. `tests/test_hdb5.py` exercises the confinement pipeline against small in-memory frames, so it does not require the downloaded HDB5 dataset. `tests/test_scaling_law.py` checks the three hand-written least-squares solvers against a problem with a known closed-form answer and against each other.

## Notes / Limitations

- Synthetic data is useful for demos and pipeline validation, but it is not a substitute for real experimental fusion data. The training CLI only uses it when you pass `--allow-synthetic`.
- The Lawson utility uses a simplified D-T ignition threshold from `config.py` and is best treated as a compact educational or screening tool rather than a full plasma physics simulator.
- Model quality depends on the dataset, feature coverage, and split behavior; holdout artifacts are for reporting, while the saved production model is refit on all prepared rows.
- The prediction CLIs expect a trained model and metadata file unless you provide custom `--model-path` and `--metadata-path` values. They validate the saved preprocessing contract against the current runtime code before scoring. Explicit artifact selection requires exact recorded runtime versions, while default artifact selection may accept limited compatible drift with warnings.
- Batch CSV prediction only streams non-grouped inputs. Grouped time-series inputs are read as a whole file so shot-level aggregation can see every row for a shot.
- The strict preprocessing contract is intentional. In this repo, silent feature drift is more dangerous than the inconvenience of regenerating artifacts, because the goal is fail-fast behavior around physics results. The contract is an explicit, versioned structural description (columns, feature schema, physics constants and tolerances); it deliberately does not fingerprint function source or bytecode, since that broke on harmless reformatting and forced spurious retrains. Bump `PREPROCESSING_CONTRACT_VERSION` in `features.py` whenever you change preprocessing semantics.
- The HDB5 dataset is third-party scientific data. It is fetched on demand from OSF and is not redistributed in this repository, so `data/raw/hdb5_std5.csv` is gitignored. Commands that need it and cannot find it raise a `FileNotFoundError` naming the OSF source and the `--dataset-path` override; run `python3 hdb5.py download` (or `train --download-if-missing`) to fetch it.
- The confinement pipeline reports against the analytic IPB98(y,2) scaling law rather than against a mean baseline alone. Treat `beats_physics_baseline` in `confinement_metadata.json` as the headline result: a model that does not beat published physics on grouped cross-validation has not learned anything useful.
- The test suite exercises many pipeline paths, but ML changes should still be validated by rerunning training and reviewing the saved artifacts.

## Module Ownership

Training and inference are each split into a thin orchestration module plus focused helpers, so the pieces can change independently without an import cycle.

Training side:

- `training.py` owns training orchestration, holdout evaluation, metric/metadata assembly, and artifact writing.
- `training_split.py` owns holdout and cross-validation split selection, including the exact bounded subset-sum search for row-targeted grouped holdouts and its linear greedy fallback for very large group sets.
- `training_registry.py` owns the preprocessing transformer and the candidate model factories that training cross-validates and selects among.
- `training_artifacts.py` owns the per-run path layout plus the staged-write and atomic-publish/cleanup logic for a run directory.
- `training_reports.py` owns the best-effort diagnostic plots; matplotlib and seaborn are imported lazily, and failures here degrade to "reports skipped" instead of discarding a successful run.

Inference side:

- `inference.py` owns the single-case and batch prediction flow and re-exports the public inference API, so `import inference` stays the one stable entry point.
- `inference_artifacts.py` owns the versioned artifact schema, the strict metadata parsers/validators, and the run-manifest writers that training persists.
- `inference_selection.py` owns artifact discovery, compatibility ranking under the configured selection mode, and deserialization of the first loadable candidate.

Shared and entrypoints:

- `config.py` owns paths, column configuration, physics constants, and tolerances.
- `features.py` owns alias mapping, temperature normalization, feature engineering, and the versioned preprocessing contract.
- `validation.py` owns the physics input validation primitives used by both pipelines and by `lawson.py`.
- `artifact_model.py` owns the `FusionFluxModelArtifact` wrapper that enforces preprocessing compatibility and clips negative predictions.
- `storage.py` owns atomic file writes and the JSON/CSV output helpers.
- `fusionflux_cli.py` owns the argparse CLI behind the installed `fusionflux` console script.
- `scaling_law.py` owns the from-scratch linear algebra: the three classical least-squares solvers, design-matrix conditioning analysis, scaling-law fitting, and bootstrap confidence intervals. It deliberately does not call scikit-learn.
- `hdb5.py` owns the entire real-data confinement-time pipeline (download, cleaning, features, model zoo, training, prediction, and its own CLI). It shares only `config.py` and `storage.py` with the neutron-yield pipeline.
- `train_model.py` is kept as a compatibility facade and CLI entrypoint so existing commands and imports continue to work.
