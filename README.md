# FusionFlux

[![CI](https://github.com/lsalcedo100/FusionFlux/actions/workflows/ci.yml/badge.svg)](https://github.com/lsalcedo100/FusionFlux/actions/workflows/ci.yml)
[![Pages](https://github.com/lsalcedo100/FusionFlux/actions/workflows/pages.yml/badge.svg)](https://lsalcedo100.github.io/FusionFlux/)
[![Python 3.9 - 3.12](https://img.shields.io/badge/python-3.9%20--%203.12-blue.svg)](https://github.com/lsalcedo100/FusionFlux/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
[![data: ITPA HDB5 STD5](https://img.shields.io/badge/data-ITPA%20HDB5%20STD5-8a3ffc.svg)](https://osf.io/drwcq)

**[Read the interactive summary](https://lsalcedo100.github.io/FusionFlux/)** · [six-page paper](paper/paper.pdf) · [full writeup](results/RESULTS.md)

`FusionFlux` asks how well machine learning predicts tokamak energy confinement time on **real experimental data**, and what happens to that answer when the model is asked about a machine nobody has built yet.

The data is the ITPA Global H-mode Confinement Database, standard analysis set STD5: 6228 quasi-stationary time slices from 4471 discharges across 18 tokamaks. Every model is scored against the analytic IPB98(y,2) scaling law, the published physics baseline, rather than against a mean predictor, and under splits that hold out a whole discharge, then a whole machine, then a whole size range.

The short version: the model that wins on cross-validation is the worst of three on a machine it has not seen, and at the size jump ITER asks for it lands closer to a constant predictor than to the power law it beat. The failure has a measurable mechanism rather than a tuning fix. And the design matrix these scaling laws are fitted on turns out to be rank deficient by two.

What is in the repository:

- **[Confinement time on real HDB5 data](#confinement-time-on-real-hdb5-data)** (`hdb5.py`) is the pipeline behind every reported result: cleaning, features, the model zoo, the IPB98(y,2) baseline, and the leave-one-machine-out and size-ordered splits.
- **[Fitting scaling laws from scratch](#fitting-scaling-laws-from-scratch)** (`scaling_law.py`) treats a confinement scaling law as the least-squares problem it is, with the three classical solvers written by hand plus the conditioning, null-space and bootstrap analysis behind Results 1 to 3.
- **[The Lawson criterion utility](#lawson-criterion-utility)** (`lawson.py`) is a standalone triple-product and ignition-ratio calculation.
- **[The neutron-yield pipeline](#infrastructure-the-neutron-yield-pipeline)** (`train_model.py` and the `neutron_yield/` package) is ML *infrastructure*: a versioned preprocessing contract, atomic run publishing, strict artifact compatibility. **It ships with a synthetic demo dataset and supports no scientific claim.** It is documented last, deliberately.

## Results

**Three ways in, shortest first:**

| | |
|---|---|
| **[Interactive summary](https://lsalcedo100.github.io/FusionFlux/)** | One page: the reversal, the ITER-direction result, and a panel where you pick the held-out machine and watch the ranking rearrange. Published from `main` on every change to `results/`; rebuild locally with `python3 site/build_page.py`. |
| **[Six-page paper](paper/paper.pdf)** (`paper/paper.tex`) | Abstract, method, all seven results, limitations. Build with `tectonic paper/paper.tex`. |
| **[results/RESULTS.md](results/RESULTS.md)** | The full writeup: every claim, table, mechanism and limitation, with nothing left out. |

Full writeup with tables, limitations and reproduction steps: **[results/RESULTS.md](results/RESULTS.md)**. Regenerate with `python3 analysis_scaling_law.py`, `python3 analysis_extrapolation.py`, `python3 analysis_flexibility_sweep.py`, `python3 analysis_size_extrapolation.py`, `python3 analysis_hybrid.py` and `python3 analysis_conformal.py`.

Measured on the real ITPA H-mode confinement database (HDB5 STD5): 6228 quasi-stationary time slices from 4471 discharges across 18 tokamaks. No synthetic data is used in any reported result. The dataset is fetched from OSF rather than committed, and is **pinned by SHA-256**: the pipeline verifies it on load and refuses to run on anything else, so every number below is tied to a specific set of bytes rather than to whatever the host is currently serving. Check with `python3 hdb5.py verify`.

### The headline: a learned model beats the published scaling law, and that result does not survive contact with a new machine

![Interpolation against extrapolation](results/extrapolation.png)

Under cross-validation grouped by discharge, a random forest cuts RMSLE 41% below the analytic IPB98(y,2) law. But grouped CV holds out *shots*, so every machine in the held-out fold is also in the training fold. Hold out an entire tokamak instead, train on the other 12 and predict the 13th, and **the ranking of the three blind models reverses exactly**:

| model | CV, by discharge | leave-one-tokamak-out | 95% interval | ratio |
|---|---|---|---|---|
| random forest | 0.128 | 0.465 | [0.376, 0.560] | **3.6x worse** |
| histogram gradient boosting | 0.130 | 0.359 | [0.279, 0.442] | 2.8x worse |
| ridge, log-linear | 0.181 | 0.214 | [0.183, 0.241] | 1.2x worse |
| IPB98(y,2), analytic (fitted on this database, not blind) | 0.199 | 0.188 | [0.158, 0.219] | unchanged |

Both columns use the same nine features and the same models; only the split changes. The best model in this repository by cross-validation is the worst of the three on a machine it has not seen, and its 41% margin turns out to measure how much of JET is predictable from the rest of JET. Intervals are a 95% percentile bootstrap resampling **machines**, since that is the sampling unit the claim is about. They overlap, so the gaps are also tested paired by machine, which is the statistic that removes the enormous differences in how hard each machine is: **the random forest is worse than the power law on 13 of 13 machines**, gap +0.251 [+0.157, +0.342].

**The failure has a mechanism, and it is measurable.** The random forest's per-machine error correlates with how far that machine sits outside the training data at rho = **+0.85**; the log-linear power law's does not, at rho = **-0.06**. And when JET is held out, 48% of its rows lie above the highest confinement time in the remaining 12 machines: a tree ensemble averages training targets, so **no tree in the forest can output those values at all**, whatever the features say. That bound is asserted directly in `tests/test_extrapolation.py`.

**And what the constraint buys is variance, not accuracy.** The obvious objection is that ridge only wins because it is the one model here not bounded by its training range. Polynomial controls in the log features test that directly: degree 2 and degree 3 are far more flexible than plain ridge but still extrapolate without bound. Degree 2 turns out to be *no worse on a typical machine* than degree 1 (median 0.238 against 0.216, better on 8 of 13, paired interval contains zero). What flexibility costs is the tail: degree 1's worst machine of thirteen is 0.289, degree 2's is 1.083 and degree 3's is 4.601, all on C-Mod. And the tree ensembles, mediocre everywhere, are never catastrophic (worst 0.686 and 0.857) because the same Result 4c ceiling that stops them predicting JET also caps how wrong they can be. For a next-step device you get one machine and one shot, so the tail is the statistic that matters. This is why the field still uses power laws it knows fit worse.

### And at the size jump ITER actually asks for, they are closer to a constant than to the power law

![Size-ordered extrapolation](results/size_extrapolation.png)

Leave-one-tokamak-out still leaves twelve machines spanning the held-out one's range, so it extrapolates in identity while interpolating in size. ITER's major radius is 6.2 m against 3.40 m for the largest row here: a factor of 1.82 beyond the database. That factor turns out to be available *inside* the database. Train on the 14 smallest machines (up to DIII-D, R = 1.865 m, 3498 rows) and predict the 4 largest (up to JT-60U, R = 3.40 m, 2730 rows) and the size jump demanded is 1.823, matching ITER's 1.824 to 0.03% in log terms.

Same models, same nine features, three questions of increasing difficulty:

| model | held-out shot | held-out machine | machine larger than any in training | skill |
|---|---|---|---|---|
| IPB98(y,2), analytic (not blind) | 0.199 | 0.188 | **0.194** | 1.00 |
| ridge, log-linear | 0.181 | 0.214 | **0.278** | 0.93 |
| random forest | 0.128 | 0.465 | **0.938** | 0.41 |
| histogram gradient boosting | 0.130 | 0.359 | **1.072** | 0.31 |
| mean baseline | 0.869 | 0.994 | 1.459 | 0.00 |

`skill` places each model between predicting a constant (0.0) and the analytic power law (1.0). **The power law keeps 93% of that distance; the trees keep 31% and 41%.** The histogram gradient booster scores 1.072 where predicting a single constant scores 1.459: the best cross-validated model families in this repository, asked the question a scaling law exists to answer, land closer to a constant than to the law they beat by 41% under cross-validation.

It is size rather than plasma shape: dropping the spherical tokamaks, which are small and would otherwise sit in every training set, moves the random forest from 0.938 to 0.936. And the mechanism is the Result 4c bound at scale. At this cut **34% of held-out rows lie above the highest confinement time in training**, with the best shot 3.9x above anything any tree can output, so for a third of the test set the trees are not making a bad prediction but the largest one available to them.

### The cure the diagnosis implies: a power law with a bounded correction

![The hybrid frontier](results/hybrid.png)

Everything above is a negative result, and it is specific enough to build against. Result 4d says the problem is functional *form*, and Result 4c says a tree ensemble's boundedness is what stops it reaching a bigger machine. So fit the power law, learn a correction on its **log residuals**, and damp that correction by a factor `lambda`, with `lambda = 0` reproducing plain ridge exactly.

Across the ITER-matched size cut, a boosted-tree correction moves the power law from 0.278 to **0.206**, better on all three held-out machines individually and by the widest margin on JT-60U, the largest and most distant. That makes it **the best blind model at that cut**, 26% below plain ridge, 4.6x below the random forest, and within 6% of IPB98(y,2), which was fitted with those machines included. A degree-2 polynomial correction, damped the same way, goes the other direction: 0.278 to 0.356.

The mechanism is measured rather than asserted. Fitted on the 14 small machines, the power law is *biased* on the big ones (mean log residual -0.218, so it over-predicts by about 20%) while its scatter is no worse than in-sample. The tree correction never leaves the range it was trained on, on any held-out row, and inside that bound it points the right way and is largest exactly where the bias is largest. **The same boundedness that makes a tree useless as a predictor of `tau` on a larger machine makes it safe as a corrector**, because the quantity it is bounded on is now a residual centred on zero rather than a target that grows with size.

The honest limits, and they are real: the gain is along the size axis only, and off it the correction hurts (C-Mod goes from 0.173 to 0.401). Scored at every rung of the size sweep rather than only the matched one, the hybrid wins at **5 of 8 well-powered cuts**, not all, and no rule over those eight points separates the wins from the losses. Cross-validation selects the least damped rung in both families, and the rung that is best on a held-out machine is plain ridge, which nothing in the CV signal points at. What survives is narrow: at the cut matching the jump to ITER, robustly across all nine correction settings tried there, a bounded correction beats the power law and the reason is measurable. See [Result 6](results/RESULTS.md#result-6-a-model-that-is-flexible-in-range-and-still-extrapolates).

### The intervals are not merely wrong, they are confident

![Conformal coverage](results/conformal.png)

For a next-step device the point error is not the deliverable; the interval is. Split-conformal prediction on the log residuals, calibrated on 25% of held-back *discharges*, gives every model a nominal 90% interval.

| model | grouped CV | held-out machine | ITER-matched cut |
|---|---|---|---|
| IPB98(y,2), analytic (not blind) | 90% | 89% | 88% |
| ridge, log-linear | 90% | 83% | 70% |
| hybrid (above) | 90% | 64% | **76%** |
| hist gradient boosting | 91% | 45% | **0%** |
| random forest | 91% | 35% | **3%** |

The control arm works: every model lands within a point of nominal where the exchangeability the method assumes actually holds, which is what licenses reading the rest. Out of distribution it does not. **The random forest's 90% interval covers 3% of the rows across the ITER-matched cut, and the histogram gradient booster's covers none of the 2730.** And the widths do not move: no model's interval changes width by more than 1.5% between the two arms, because the half-width is set by calibration rows drawn the same way in both. The intervals do not become vague out of distribution. They stay the same size and miss.

Coverage collapses along the same axis the errors grow, but only for the trees: coverage against Mahalanobis distance runs rho = -0.77 for the random forest and -0.54 for the gradient booster, against +0.27 for the power law, mirroring Result 4b's +0.85 and -0.06 on point error with the sign flipped. None of this is a defect in conformal prediction; it is an assumption being false, measured. See [Result 7](results/RESULTS.md#result-7-the-intervals-are-not-merely-wrong-they-are-confident).

### The linear algebra underneath

![Singular value spectrum and disagreement decomposition](results/singular_value_spectrum.png)

**The model's own feature matrix is rank deficient by two, and this audit found it.** Standardized, the ten log features have rank 8. Two exact dependencies, each confirmed by projection onto the null space at a residual of order 1e-16: minor radius is derived as `a = eps * R`, and the IPB98 prior is a fixed log-linear combination of the other eight features. That second one means a published physics scaling, added as a feature, contributes exactly nothing to a log-linear model, however much it looks like added knowledge. Nothing crashed because `scipy.linalg.lstsq` inverts through the SVD pseudoinverse and silently returns the minimum-norm member of a two-parameter family.

**Refitting IPB98(y,2) from the database disagrees with the published exponents almost entirely where the data is blind.** Solving three ways from scratch (Cholesky on the normal equations, QR, SVD, agreeing to 8e-13) gives Ip 1.08 against 0.93 and R 1.58 against 1.97, while P and Bt come back essentially exactly. Decomposing that difference along the singular directions of the design matrix: **77% of it lies in the single weakest direction, which carries 0.3% of the matrix's variance**, and the three strongest directions carry 82% of the variance while accounting for 0.75% of the disagreement. That weak direction is plasma current traded against machine size, which is structurally hard to resolve because tokamaks are not designed to vary the two independently.

## Quickstart

```bash
# 1. Install (editable, with dev tooling) into a virtualenv
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e ".[dev]" -c constraints.txt

# 2. Fetch the real HDB5 STD5 dataset (content-hash verified), train, report against IPB98(y,2)
python3 hdb5.py train --download-if-missing

# 3. Ask the question a scaling law exists for: hold out a whole machine, then hold out size
python3 hdb5.py extrapolate
python3 hdb5.py size-extrapolate

# 4. Regenerate every number and figure under results/
python3 analysis_scaling_law.py
python3 analysis_extrapolation.py
python3 analysis_flexibility_sweep.py
python3 analysis_size_extrapolation.py
python3 analysis_hybrid.py
python3 analysis_conformal.py

# 5. Reproduce the CI quality gate locally
make check        # == ruff check . && mypy . && pytest -q
```

The synthetic-data neutron-yield pipeline has its own commands, documented under [Infrastructure: the Neutron-Yield Pipeline](#infrastructure-the-neutron-yield-pipeline).

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

`python3 analysis_extrapolation.py` runs both splits on the one shared feature set, adds the distance diagnostics, the machine-level bootstrap intervals, the polynomial flexibility ladder and the figure, and writes everything under `results/`. See [Result 4](results/RESULTS.md#result-4-the-model-that-wins-on-cross-validation-loses-on-a-new-machine).

### Sweep flexibility instead of sampling it

Result 4d compares three polynomial degrees at one ridge penalty, which is enough to notice that the tail grows with flexibility and not enough to claim it must. `python3 analysis_flexibility_sweep.py` runs the grid instead: degrees 1 to 4 against nine decades of penalty, 36 models, each scored under both splits.

```bash
python3 analysis_flexibility_sweep.py   # writes flexibility_sweep.{json,png,csv} under results/
```

It answers the two objections Result 4d leaves open. The worst held-out machine grows 2.27x to 3.21x per degree at **every** penalty that leaves plain ridge intact, so the trend is not an artifact of the one alpha that happened to be reported; and no amount of regularization rescues a flexible form, even when each degree's penalty is chosen with hindsight on the machines it is then scored on. The grid's `alpha = 1.0` column reproduces Result 4d's table exactly, which is what makes it the same family of models rather than a different one. See [Result 4e](results/RESULTS.md#result-4e-flexibility-is-a-family-and-the-whole-family-degrades).

The fits use the repository's own SVD ridge solver rather than scikit-learn's estimator: the penalty enters only through a per-direction filter, so one factorization per fold serves the entire penalty axis and the grid costs what its degree axis alone would. `tests/test_flexibility_sweep.py` pins the hand-rolled path against the scikit-learn pipelines it replaces.

### Extrapolate in the direction ITER sits

Leave-one-tokamak-out holds out a machine but leaves twelve others spanning its range, so it still interpolates in size. To extrapolate in size, order the machines by major radius, cut, train below the cut and predict everything above it:

```bash
python3 hdb5.py size-extrapolate                                  # sweep every cut, report the ITER-matched one
python3 hdb5.py size-extrapolate --conventional-aspect-ratio-only # drop the spherical tokamaks (Result 5b)
python3 hdb5.py size-extrapolate --output-dir results             # also write the sweep and matched-cut CSVs
```

The reported cut is chosen as the one whose size ratio is closest to ITER's 6.2 m divided by the largest major radius in the database, so it is a property of the data rather than a hand-picked split: add a larger machine and the matched rung moves on its own.

`python3 analysis_size_extrapolation.py` adds the three-question escalation, the per-machine breakdown, the aspect-ratio control, the truncation diagnostics and the figure. See [Result 5](results/RESULTS.md#result-5-the-same-jump-iter-asks-for-measured-inside-the-database).

### Build a model that extrapolates, and put an interval on it

The power law plus a damped correction on its log residuals, swept from plain ridge (`lambda = 0`) to an undamped correction (`lambda = 1`), scored under all three splits:

```bash
python3 analysis_hybrid.py        # the shrinkage sweep, the frontier, the mechanism, the figure
```

The damping factor is selected on grouped CV, which is the only split a team actually has; the rung that would be best on a held-out machine is reported separately and labelled as the oracle it is. See [Result 6](results/RESULTS.md#result-6-a-model-that-is-flexible-in-range-and-still-extrapolates).

Split-conformal intervals on the log residuals, and their coverage under each split:

```bash
python3 hdb5.py conformal                      # pooled coverage under all three splits
python3 hdb5.py conformal --alpha 0.05         # 95% intervals instead of 90%
python3 hdb5.py conformal --output-dir results # also write the per-machine coverage table
python3 analysis_conformal.py                  # adds the per-machine collapse, the widths and the figure
```

Coverage is reported next to interval width throughout, because coverage alone is trivial to win: an interval wide enough to be useless covers everything. See [Result 7](results/RESULTS.md#result-7-the-intervals-are-not-merely-wrong-they-are-confident).

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
- This pipeline is deliberately independent of the neutron-yield pipeline. It does not use the preprocessing contract, the run-directory layout, or the artifact selection modes described under [the neutron-yield infrastructure](#infrastructure-the-neutron-yield-pipeline).

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

## Repository Structure

Grouped by what it is for rather than alphabetically: the real-data study first, the shared plumbing next, the neutron-yield infrastructure last.

```text
FusionFlux/
│
│   # real-data confinement study: everything the reported results come from
├── hdb5.py                          # ITPA HDB5 pipeline: download, pin check, cleaning, model zoo, CLI
├── scaling_law.py                   # from-scratch least squares; fits and audits scaling laws
├── analysis_scaling_law.py          # Results 1 to 3: rank audit, IPB98 refit, conditioning,
│                                    #   bootstrap resolution (2b) and the solver sweep (2c)
├── analysis_extrapolation.py        # Result 4: leave-one-tokamak-out study and figure
├── analysis_flexibility_sweep.py    # Result 4e: polynomial degree against ridge penalty
├── analysis_size_extrapolation.py   # Result 5: size-ordered cut at the ITER-matched jump
├── analysis_hybrid.py               # Result 6: power law plus a damped residual correction
├── analysis_conformal.py            # Result 7: split-conformal coverage under each split
├── lawson.py                        # standalone Lawson criterion utility
├── results/
│   ├── RESULTS.md                   # the writeup: every claim, table and limitation
│   ├── extrapolation.png            # Result 4 figure, plus its .json/.csv companions
│   ├── flexibility_sweep.png        # Result 4e figure, plus its .json/.csv companions
│   ├── size_extrapolation.png       # Result 5 figure, plus its .json/.csv companions
│   ├── hybrid.png                   # Result 6 figure, plus its .json/.csv companions
│   ├── conformal.png                # Result 7 figure, plus its .json/.csv companions
│   ├── singular_value_spectrum.png  # Results 1 to 3 figure
│   ├── solver_conditioning.png      # Result 2c: forward error against condition number
│   ├── analysis.json                # rank audit, refit exponents, conditioning, solver sweep
│   ├── bootstrap_resolution.csv     # Result 2b: exponent intervals at all three units
│   └── ipb98_refit_exponents.csv    # refit against published, with bootstrap intervals
│
├── paper/
│   ├── paper.tex                    # six-page writeup; build with `tectonic paper/paper.tex`
│   └── README.md                    # how to build it, and the Zenodo DOI flow
├── site/
│   ├── page.template.html           # the one-page interactive summary
│   └── build_page.py                # fills it from results/; writes site/index.html
├── docs/
│   └── neutron-yield-pipeline.md    # operating detail for the synthetic-data infrastructure
│
│   # shared plumbing, used by both pipelines
├── config.py                        # paths, column config, physics constants and tolerances
├── storage.py                       # atomic file writes and JSON/CSV helpers
├── validation.py                    # physics input validation primitives
│
│   # neutron-yield infrastructure (synthetic demo data, no scientific claim)
├── train_model.py                   # CLI entrypoint and compatibility facade over the package
├── neutron_yield/                   # the pipeline itself, packaged away from the science
│   ├── __init__.py                  # states the scope: infrastructure, not a physical claim
│   ├── fusionflux_cli.py            # argparse CLI behind the `fusionflux` console script
│   ├── features.py                  # alias mapping, validation, feature engineering, contract
│   ├── artifact_model.py            # saved-model wrapper with preprocessing + clipping guardrails
│   ├── training.py                  # training orchestration and holdout evaluation
│   ├── training_artifacts.py        # per-run path layout, staged write, atomic publish
│   ├── training_registry.py         # preprocessor and candidate model factories
│   ├── training_reports.py          # residual and feature-importance plots
│   ├── training_split.py            # random and grouped holdout / CV split selection
│   ├── inference.py                 # single/batch prediction flow, public inference API
│   ├── inference_artifacts.py       # artifact schema, metadata parsing, run-manifest writers
│   └── inference_selection.py       # artifact discovery, default selection, loading
│
├── Makefile
├── pyproject.toml
├── requirements.txt
├── constraints.txt
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       ├── ci.yml
│       └── pages.yml
├── tests/
│   ├── conftest.py
│   ├── helpers.py
│   ├── test_hdb5.py                 # confinement pipeline, on small in-memory frames
│   ├── test_dataset_integrity.py    # the HDB5 content pin, including how it fails
│   ├── test_scaling_law.py          # the three hand-written solvers against a known answer
│   ├── test_solver_conditioning.py  # Result 2c: the kappa^2 vs kappa slope separation
│   ├── test_bootstrap_resolution.py # Result 2b: which exponents widen, and why
│   ├── test_extrapolation.py        # Result 4, including the tree ceiling bound
│   ├── test_flexibility_sweep.py    # Result 4e, incl. the sklearn-equivalence cross-check
│   ├── test_size_extrapolation.py   # Result 5, including that the cut is data-picked
│   ├── test_hybrid.py               # Result 6, incl. the bounded/unbounded correction contrast
│   ├── test_conformal.py            # Result 7, incl. the finite-sample conformal rank
│   ├── test_lawson.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_inference.py
└── data/
    ├── raw/
    │   ├── hdb5_std5.csv            # not committed; fetched via `python3 hdb5.py download`
    │   └── synthetic_nuclear_fusion_experiment.csv   # sample/reference copy only
    └── processed/
        ├── hdb5_confinement/
        │   ├── confinement_model.joblib
        │   ├── confinement_metrics.csv
        │   └── confinement_metadata.json
        ├── latest_training_run.json
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

## Testing

Run the test suite with:

```bash
.venv/bin/python -m pytest -q
```

or run the full lint/type/test gate with `make check` (`ruff check .`, then `mypy .`, then `pytest -q`).

The tests that back the reported results come first: `tests/test_hdb5.py` exercises the confinement pipeline against small in-memory frames, so it does not require the downloaded HDB5 dataset; `tests/test_dataset_integrity.py` covers the content pin, including each way it can fail, the guarantee that a corrupt download never lands at the target path, and that every generated JSON under `results/` carries the digest of the bytes it was computed from; `tests/test_scaling_law.py` checks the three hand-written least-squares solvers against a problem with a known closed-form answer and against each other, while `tests/test_solver_conditioning.py` measures that Cholesky's forward error grows at the square of the condition number where QR and SVD grow linearly, which is simultaneously a check that the implementations are what they claim to be; `tests/test_bootstrap_resolution.py` covers Result 2b, asserting the sharp version of the claim (the exponents that genuinely vary between machines widen under coarse resampling and the ones that do not, do not) rather than the blanket one; `tests/test_extrapolation.py` covers Result 4, including the assertion that no tree in the forest can output a value above its training range; `tests/test_flexibility_sweep.py` covers Result 4e, with its load-bearing test pinning the hand-rolled ridge path against the scikit-learn pipelines it replaces; and `tests/test_size_extrapolation.py` covers Result 5, including that the ITER-matched cut is read off the data rather than hardcoded.

The neutron-yield pipeline is covered by `tests/test_preprocessing.py`, `tests/test_training.py`, and `tests/test_inference.py`, with `tests/test_lawson.py` for the physics utility, shared fixtures in `tests/conftest.py`, and shared stubs/builders in `tests/helpers.py`. Between them they cover Lawson calculations, temperature conversions, preprocessing and validation rules, grouped-shot aggregation, training split behavior, training artifact cleanup, preprocessing-contract compatibility checks, negative prediction clipping, and single/batch inference edge cases.

CI runs the same gate on Python 3.9–3.12 with coverage reporting (`pytest -q --cov --cov-report=term-missing`; the measured module list lives under `[tool.coverage.run]` in `pyproject.toml`). The 3.9 job installs against `constraints.txt` to reproduce the tested training and artifact-loading environment, while the newer-Python jobs resolve current releases so the `>=3.9` support claim is actually exercised. Runs use pip caching and cancel superseded runs for the same ref, and Dependabot opens weekly grouped update PRs for the GitHub Actions and pip dependencies.

Two suites skip rather than fail on a fresh clone: `test_committed_artifact_manifest_supports_relocation` needs a locally trained artifact under `data/processed/` (gitignored), and the parts of `tests/test_dataset_integrity.py` that fingerprint the real STD5 file need it downloaded. The checks that matter most in that file do not, since they synthesise a mismatched file and assert on the refusal.

## Infrastructure: the Neutron-Yield Pipeline

**Nothing in this section supports a scientific claim.** The neutron-yield pipeline predicts `neutron_yield` from plasma operating conditions, and the dataset it ships with is synthetic: generated by `create_synthetic_dataset` from a hand-crafted signal, so any accuracy number it produces measures how learnable that generator is and nothing else. None of the results above come from it, and it shares only `config.py` and `storage.py` with the pipeline that does.

It is here as engineering rather than as a result. It is a complete training and inference stack built to fail loudly instead of drifting silently:

- a **versioned preprocessing contract** (column set, feature schema, physics constants and tolerances, plus a hash of that structural description) persisted with every training run and checked before any prediction, so an artifact trained under different preprocessing semantics is rejected rather than scored
- **atomic run publishing**: artifacts are staged under a hidden directory and renamed into place only once the run completes, so a crash mid-run never leaves a half-written run for the inference loader to find
- a saved-model wrapper, `FusionFluxModelArtifact`, that enforces the same compatibility check and clips negative predictions **even on a bare `joblib.load(...).predict(...)`** that bypasses the inference API entirely

Training requires an explicit dataset choice: `--dataset-path` for a real CSV, or `--allow-synthetic` to generate demo data on purpose. There is no implicit default.

```bash
python3 train_model.py train --allow-synthetic
python3 train_model.py predict --density-m3 1e20 --temperature 15 --temp-unit keV --confinement-time-s 4
```

Full operating detail, including the training and prediction CLIs, the synthetic generator's assumptions and every generated artifact, is in **[docs/neutron-yield-pipeline.md](docs/neutron-yield-pipeline.md)**. It is a separate document so that this README stays about the science.

## Notes / Limitations

The limitations that bear on the reported results are stated in full in [results/RESULTS.md](results/RESULTS.md#limitations). What follows is repository-level.

**Real data and physics**

- The HDB5 dataset is third-party scientific data. It is fetched on demand from OSF and is not redistributed in this repository, so `data/raw/hdb5_std5.csv` is gitignored. Commands that need it and cannot find it raise a `FileNotFoundError` naming the OSF source and the `--dataset-path` override; run `python3 hdb5.py download` (or `train --download-if-missing`) to fetch it.
- The confinement pipeline reports against the analytic IPB98(y,2) scaling law rather than against a mean baseline alone. Treat `beats_physics_baseline` in `confinement_metadata.json` as the headline result: a model that does not beat published physics on grouped cross-validation has not learned anything useful.
- The Lawson utility uses a simplified D-T ignition threshold from `config.py` and is best treated as a compact educational or screening tool rather than a full plasma physics simulator.

**Neutron-yield infrastructure**

- Synthetic data is useful for demos and pipeline validation, but it is not a substitute for real experimental fusion data. The training CLI only uses it when you pass `--allow-synthetic`.
- Model quality depends on the dataset, feature coverage, and split behavior; holdout artifacts are for reporting, while the saved production model is refit on all prepared rows.
- The prediction CLIs expect a trained model and metadata file unless you provide custom `--model-path` and `--metadata-path` values. They validate the saved preprocessing contract against the current runtime code before scoring. Explicit artifact selection requires exact recorded runtime versions, while default artifact selection may accept limited compatible drift with warnings.
- Batch CSV prediction only streams non-grouped inputs. Grouped time-series inputs are read as a whole file so shot-level aggregation can see every row for a shot.
- The strict preprocessing contract is intentional. In this repo, silent feature drift is more dangerous than the inconvenience of regenerating artifacts, because the goal is fail-fast behavior around physics results. The contract is an explicit, versioned structural description (columns, feature schema, physics constants and tolerances); it deliberately does not fingerprint function source or bytecode, since that broke on harmless reformatting and forced spurious retrains. Bump `PREPROCESSING_CONTRACT_VERSION` in `neutron_yield/features.py` whenever you change preprocessing semantics.

**Both**

- The test suite exercises many pipeline paths, but ML changes should still be validated by rerunning training and reviewing the saved artifacts.

## Module Ownership

Each pipeline is split into a thin orchestration module plus focused helpers, so the pieces can change independently without an import cycle.

Real-data confinement study:

- `hdb5.py` owns the entire real-data confinement-time pipeline (download, cleaning, features, model zoo, training, prediction, and its own CLI). It shares only `config.py` and `storage.py` with the neutron-yield pipeline.
- `scaling_law.py` owns the from-scratch linear algebra: the three classical least-squares solvers, design-matrix conditioning analysis, scaling-law fitting, and bootstrap confidence intervals. It deliberately does not call scikit-learn.
- `lawson.py` owns the standalone triple-product and ignition-ratio calculation, and is the one physics utility both pipelines can borrow from.

Shared and entrypoints:

- `config.py` owns paths, column configuration, physics constants, and tolerances.
- `storage.py` owns atomic file writes and the JSON/CSV output helpers.
- `validation.py` owns the physics input validation primitives used by both pipelines and by `lawson.py`.

Neutron-yield infrastructure, training side:

- `neutron_yield/training.py` owns training orchestration, holdout evaluation, metric/metadata assembly, and artifact writing.
- `neutron_yield/training_split.py` owns holdout and cross-validation split selection, including the exact bounded subset-sum search for row-targeted grouped holdouts and its linear greedy fallback for very large group sets.
- `neutron_yield/training_registry.py` owns the preprocessing transformer and the candidate model factories that training cross-validates and selects among.
- `neutron_yield/training_artifacts.py` owns the per-run path layout plus the staged-write and atomic-publish/cleanup logic for a run directory.
- `neutron_yield/training_reports.py` owns the best-effort diagnostic plots; matplotlib and seaborn are imported lazily, and failures here degrade to "reports skipped" instead of discarding a successful run.

Neutron-yield infrastructure, inference side:

- `neutron_yield/inference.py` owns the single-case and batch prediction flow and re-exports the public inference API, so `from neutron_yield import inference` stays the one stable entry point.
- `neutron_yield/inference_artifacts.py` owns the versioned artifact schema, the strict metadata parsers/validators, and the run-manifest writers that training persists.
- `neutron_yield/inference_selection.py` owns artifact discovery, compatibility ranking under the configured selection mode, and deserialization of the first loadable candidate.

Neutron-yield infrastructure, both sides:

- `neutron_yield/features.py` owns alias mapping, temperature normalization, feature engineering, and the versioned preprocessing contract.
- `neutron_yield/artifact_model.py` owns the `FusionFluxModelArtifact` wrapper that enforces preprocessing compatibility and clips negative predictions.
- `neutron_yield/fusionflux_cli.py` owns the argparse CLI behind the installed `fusionflux` console script.
- `train_model.py` stays at the repository root as the CLI entrypoint and a compatibility facade over the package, so every documented `python3 train_model.py ...` command and every `train_model.<name>` import keeps working across the move.
