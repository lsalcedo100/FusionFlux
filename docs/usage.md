# Using the pipelines

Installation, then the command line for each pipeline in the repository. The
results these commands produce, and the argument for them, are in
[results/RESULTS.md](../results/RESULTS.md); this page is reference.

## Installation

```bash
cd FusionFlux
python3.12 -m venv .venv      # any 3.10+ interpreter; see requires-python
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

## Confinement Time on Real HDB5 Data

`hdb5.py` is the pipeline behind every result above, and it is self-contained: cleaning, features, model zoo, training, prediction and its own CLI all live in that one module. It trains on **real experimental data**: the ITPA Global H-mode Confinement Database, standard analysis set STD5 (version 5.2.3), published on the [Open Science Framework](https://osf.io/drwcq). Each row is a quasi-stationary time slice from a real tokamak discharge.

The dataset is not redistributed in this repository. Fetch it once:

```bash
python3 hdb5.py download
```

This writes `data/raw/hdb5_std5.csv` (gitignored). Pass `--overwrite` to re-download. The download is verified against a pinned SHA-256 and a pinned byte count, and the canonical path is verified again on every load, so a revised or truncated file is refused by name rather than quietly analysed; `python3 hdb5.py verify` checks the copy on disk, and `--no-verify` waives the check at the cost of results that are no longer comparable to the ones reported here. A file passed explicitly with `--dataset-path` is reported rather than enforced.

If you use this dataset, cite Verdoolaege et al., *Nucl. Fusion* **61** 076006 (2021).

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

`python3 analysis_extrapolation.py` runs both splits on the one shared feature set, adds the distance diagnostics, the machine-level bootstrap intervals, the polynomial flexibility ladder and the figure, and writes everything under `results/`. See [Result 4](../results/RESULTS.md#result-4-the-model-that-wins-on-cross-validation-loses-on-a-new-machine).

### Sweep flexibility instead of sampling it

Result 4d compares three polynomial degrees at one ridge penalty, which is enough to notice that the tail grows with flexibility and not enough to claim it must. `python3 analysis_flexibility_sweep.py` runs the grid instead: degrees 1 to 4 against nine decades of penalty, 36 models, each scored under both splits.

```bash
python3 analysis_flexibility_sweep.py   # writes flexibility_sweep.{json,png,csv} under results/
```

It answers the two objections Result 4d leaves open. The worst held-out machine grows 2.27x to 3.21x per degree at **every** penalty that leaves plain ridge intact, so the trend is not an artifact of the one alpha that happened to be reported; and no amount of regularization rescues a flexible form, even when each degree's penalty is chosen with hindsight on the machines it is then scored on. The grid's `alpha = 1.0` column reproduces Result 4d's table exactly, which is what makes it the same family of models rather than a different one. See [Result 4e](../results/RESULTS.md#result-4e-flexibility-is-a-family-and-the-whole-family-degrades).

The fits use the repository's own SVD ridge solver rather than scikit-learn's estimator: the penalty enters only through a per-direction filter, so one factorization per fold serves the entire penalty axis and the grid costs what its degree axis alone would. `tests/test_flexibility_sweep.py` pins the hand-rolled path against the scikit-learn pipelines it replaces.

### Extrapolate in the direction ITER sits

Leave-one-tokamak-out holds out a machine but leaves twelve others spanning its range, so it still interpolates in size. To extrapolate in size, order the machines by major radius, cut, train below the cut and predict everything above it:

```bash
python3 hdb5.py size-extrapolate                                  # sweep every cut, report the ITER-matched one
python3 hdb5.py size-extrapolate --conventional-aspect-ratio-only # drop the spherical tokamaks (Result 5b)
python3 hdb5.py size-extrapolate --output-dir results             # also write the sweep and matched-cut CSVs
```

The reported cut is chosen as the one whose size ratio is closest to ITER's 6.2 m divided by the largest major radius in the database, so it is a property of the data rather than a hand-picked split: add a larger machine and the matched rung moves on its own.

`python3 analysis_size_extrapolation.py` adds the three-question escalation, the per-machine breakdown, the aspect-ratio control, the truncation diagnostics and the figure. See [Result 5](../results/RESULTS.md#result-5-the-same-jump-iter-asks-for-measured-inside-the-database).

### Build a model that extrapolates, and put an interval on it

The power law plus a damped correction on its log residuals, swept from plain ridge (`lambda = 0`) to an undamped correction (`lambda = 1`), scored under all three splits:

```bash
python3 analysis_hybrid.py        # the shrinkage sweep, the frontier, the mechanism, the figure
```

The damping factor is selected on grouped CV, which is the only split a team actually has; the rung that would be best on a held-out machine is reported separately and labelled as the oracle it is. See [Result 6](../results/RESULTS.md#result-6-a-model-that-is-flexible-in-range-and-still-extrapolates).

Split-conformal intervals on the log residuals, and their coverage under each split:

```bash
python3 hdb5.py conformal                      # pooled coverage under all three splits
python3 hdb5.py conformal --alpha 0.05         # 95% intervals instead of 90%
python3 hdb5.py conformal --output-dir results # also write the per-machine coverage table
python3 analysis_conformal.py                  # adds the per-machine collapse, the widths and the figure
```

Coverage is reported next to interval width throughout, because coverage alone is trivial to win: an interval wide enough to be useless covers everything. See [Result 7](../results/RESULTS.md#result-7-the-intervals-are-not-merely-wrong-they-are-confident).

### Constrain the fit with dimensional analysis

The Connor-Taylor hierarchy, derived in code from the definitions of rho*, beta and nu*, fitted through the KKT solver already in `scaling_law.py`, and scored under all three splits alongside the prior-shrinkage family it is measured against:

```bash
python3 analysis_dimensional.py   # Results 8 and 9: the constraint hierarchy, then the prior
```

Prints the distance from each constraint surface for the published law and the free refit, what each rung costs in sample, the three splits, the whole size sweep, and the targeting-against-isotropic control. The collisionless rung is the best blind model at the ITER-matched cut; cross-validation cannot select it, which is reported rather than glossed. See [Result 8](../results/RESULTS.md#result-8-one-line-of-physics-beats-every-model-built-so-far).

### Repair the intervals, and find where the repair stops

```bash
python3 analysis_conformal_shift.py   # Result 10: three calibration schemes, two shifted arms
```

Calibrating on held-out *machines* rather than held-out discharges restores near-nominal coverage on an unseen machine and does not restore it across the size cut, which is what the exchangeability diagnosis predicts. Slow: it costs one extra fit per training machine per fold. See [Result 10](../results/RESULTS.md#result-10-repairing-the-interval-collapse-and-finding-the-limit-of-the-repair).

### Replicate on rows the standard set does not contain

```bash
python3 -c "import replication; replication.download_db523()"   # the full DB5.2.3 revision, pinned
python3 analysis_replication.py                                 # Result 11: both disjoint arms
```

Fetches the full database revision from the same OSF project, verifies its own SHA-256, and runs Result 4's protocol on the 5358 H-mode rows STD5 does not contain and on 3860 ohmic and L-mode rows scored against ITER89-P. See [Result 11](../results/RESULTS.md#result-11-the-reversal-reproduces-on-rows-this-database-never-contained).

### Write down a prediction before the answer exists

```bash
python3 analysis_forecast.py      # Result 12: SPARC, JT-60SA and ITER, with intervals and a digest
```

Writes `results/forecast.json` with each model's prediction for three real machines, the date, the dataset digest, and a content digest over the forecast rows so a later edit leaves a mark. The parameter sets are checked against the IPB98(y,2) figure each design paper quotes. See [Result 12](../results/RESULTS.md#result-12-a-locked-prediction-for-three-machines-that-have-no-data).

### Predict with an interval, and be refused when appropriate

The command a fresh install puts on the path:

```bash
fusionflux predict --ip-ma 15 --bt-t 5.3 --ne-line-1e19-m3 10 --p-loss-mw 87 \
                   --r-m 6.2 --inverse-aspect-ratio 0.3226 --kappa 1.7 --m-eff-amu 2.5
fusionflux predict ... --json      # the same thing as JSON
fusionflux card                    # rebuild results/predictor.json
fusionflux neutron train ...       # the synthetic demo pipeline, one level down
```

Reports the point estimate, a nominal 90% interval from Result 10's calibration, the Mahalanobis distance of the operating point from the training data, and an explicit refusal when that point sits beyond what this study measured. Two conditions trigger the refusal and both are read off the results rather than chosen: the query is farther out than any machine Result 4 held out and scored, or the analytic law predicts a confinement time above the training ceiling, in which case Result 4c says no range-bounded model can be right there whatever its tuning.

The equivalent from Python is `predictor.predict(...)`, keyword-only, returning a `ConfinementPrediction`. Both read `results/predictor.json`, which is committed, so neither needs the dataset. See [`predictor.py`](../predictor.py).

Note that `python3 hdb5.py predict` is the older, lower-level path: it returns a saved artifact's point estimate with no interval, no distance and no refusal, which is exactly the gap `predictor.py` exists to close. Prefer `fusionflux predict` unless you specifically want a particular saved artifact's number.

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
- This pipeline is deliberately independent of the neutron-yield pipeline. It does not use the preprocessing contract, the run-directory layout, or the artifact selection modes described under [the neutron-yield infrastructure](neutron-yield-pipeline.md).

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
