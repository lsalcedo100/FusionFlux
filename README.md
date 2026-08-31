# FusionFlux

[![CI](https://github.com/lsalcedo100/FusionFlux/actions/workflows/ci.yml/badge.svg)](https://github.com/lsalcedo100/FusionFlux/actions/workflows/ci.yml)
[![Pages](https://github.com/lsalcedo100/FusionFlux/actions/workflows/pages.yml/badge.svg)](https://lsalcedo100.github.io/FusionFlux/)
[![Python 3.10 - 3.12](https://img.shields.io/badge/python-3.10%20--%203.12-blue.svg)](https://github.com/lsalcedo100/FusionFlux/actions/workflows/ci.yml)
[![Reproduce](https://github.com/lsalcedo100/FusionFlux/actions/workflows/reproduce.yml/badge.svg)](https://github.com/lsalcedo100/FusionFlux/actions/workflows/reproduce.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
[![data: ITPA HDB5 STD5](https://img.shields.io/badge/data-ITPA%20HDB5%20STD5-8a3ffc.svg)](https://osf.io/drwcq)

**[Read the interactive summary](https://lsalcedo100.github.io/FusionFlux/)** · [six-page paper](paper/paper.pdf) · [full writeup](results/RESULTS.md)

A tokamak holds a plasma hot enough to fuse inside a magnetic field, and the plasma leaks its heat back out. How long it holds that heat, the **energy confinement time**, is what decides how large the machine has to be before fusion produces more energy than it consumes. Nobody can compute that time from first principles, so the field predicts it with a power law fitted across the machines already built, and ITER is being built on that prediction.

So it is worth asking whether machine learning does the job better. On this database it does, by a lot. **Then you ask it about a machine it has not seen, and the answer inverts.**

> A random forest beats the published physics law by **41%** under the standard way these models are validated. Under a split that holds out an entire tokamak, it is **worse than the physics law on 13 of 13 machines**. The standard validation does not merely overstate the gain, it reverses the ranking, and the reason is measurable rather than a matter of tuning.

Everything here is measured on real experimental data: the ITPA Global H-mode Confinement Database, standard analysis set STD5, 6228 quasi-stationary time slices from 4471 discharges across 18 tokamaks. Every model is scored against the analytic IPB98(y,2) scaling law, the published physics baseline, rather than against a mean predictor. No synthetic data enters any reported result. The dataset is fetched from OSF rather than committed and is **pinned by SHA-256**, verified on every load, so each number below is tied to a specific set of bytes rather than to whatever the host is currently serving.

**Three ways in, shortest first:**

| | |
|---|---|
| **[Interactive summary](https://lsalcedo100.github.io/FusionFlux/)** | One page: the reversal, the ITER-direction result, and a panel where you pick the held-out machine and watch the ranking rearrange. |
| **[Six-page paper](paper/paper.pdf)** (`paper/paper.tex`) | Abstract, method, all seven results, limitations. |
| **[results/RESULTS.md](results/RESULTS.md)** | The full writeup: every claim, table, mechanism and limitation, with nothing left out. |

## Results

### The headline: a learned model beats the published scaling law, and that result does not survive contact with a new machine

![Interpolation against extrapolation](results/extrapolation.png)

Under cross-validation grouped by discharge, a random forest cuts RMSLE 41% below the analytic IPB98(y,2) law (0.118 against 0.199, on the full ten-feature set; the table below drops the IPB98 prior as a feature and so reads 0.128, a 36% margin, for the same model). But grouped CV holds out *shots*, so every machine in the held-out fold is also in the training fold. Hold out an entire tokamak instead, train on the other 12 and predict the 13th, and **the ranking of the three blind models reverses exactly**:

| model | CV, by discharge | leave-one-tokamak-out | 95% interval | ratio |
|---|---|---|---|---|
| random forest | 0.128 | 0.465 | [0.376, 0.560] | **3.6x worse** |
| histogram gradient boosting | 0.130 | 0.359 | [0.279, 0.442] | 2.8x worse |
| ridge, log-linear | 0.181 | 0.214 | [0.183, 0.241] | 1.2x worse |
| IPB98(y,2), analytic (fitted on this database, not blind) | 0.199 | 0.188 | [0.158, 0.219] | unchanged |

Both columns use the same nine features and the same models; only the split changes. The best model in this repository by cross-validation is the worst of the three on a machine it has not seen, and its 41% margin turns out to measure how much of JET is predictable from the rest of JET. Intervals are a 95% percentile bootstrap resampling **machines**, since that is the sampling unit the claim is about. They overlap, so the gaps are also tested paired by machine, which removes the enormous differences in how hard each machine is: **the random forest is worse than the power law on 13 of 13 machines**, gap +0.251 [+0.157, +0.342].

**The failure has a mechanism.** The random forest's per-machine error correlates with how far that machine sits outside the training data at rho = **+0.85**; the log-linear power law's does not, at rho = **-0.06**. And when JET is held out, 48% of its rows lie above the highest confinement time in the remaining 12 machines: a tree ensemble averages training targets, so **no tree in the forest can output those values at all**, whatever the features say. That bound is asserted directly in `tests/test_extrapolation.py`.

**And what the constraint buys is variance, not accuracy.** Polynomial controls in the log features test the obvious objection, that ridge only wins by being unbounded. Degree 2 is far more flexible than plain ridge and still extrapolates without bound, and it is *no worse on a typical machine* than degree 1 (median 0.238 against 0.216, better on 8 of 13). What flexibility costs is the tail: degree 1's worst machine of thirteen is 0.289, degree 2's is 1.083, degree 3's is 4.601. For a next-step device you get one machine and one shot, so the tail is the statistic that matters. This is why the field still uses power laws it knows fit worse. See [Result 4](results/RESULTS.md#result-4-the-model-that-wins-on-cross-validation-loses-on-a-new-machine).

### And at the size jump ITER actually asks for, they are closer to a constant than to the power law

![Size-ordered extrapolation](results/size_extrapolation.png)

Leave-one-tokamak-out still leaves twelve machines spanning the held-out one's range, so it extrapolates in identity while interpolating in size. ITER's major radius is 6.2 m against 3.40 m for the largest row here, a factor of 1.82 beyond the database, and that factor turns out to be available *inside* the database: train on the 14 smallest machines and predict the 4 largest, and the size jump demanded matches ITER's to 0.03% in log terms.

| model | held-out shot | held-out machine | machine larger than any in training | skill |
|---|---|---|---|---|
| IPB98(y,2), analytic (not blind) | 0.199 | 0.188 | **0.194** | 1.00 |
| ridge, log-linear | 0.181 | 0.214 | **0.278** | 0.93 |
| random forest | 0.128 | 0.465 | **0.938** | 0.41 |
| histogram gradient boosting | 0.130 | 0.359 | **1.072** | 0.31 |
| mean baseline | 0.869 | 0.994 | 1.459 | 0.00 |

`skill` places each model between predicting a constant (0.0) and the analytic power law (1.0). **The power law keeps 93% of that distance; the trees keep 31% and 41%.** The best cross-validated model families in this repository, asked the question a scaling law exists to answer, land closer to a constant than to the law they beat by 41% under cross-validation. It is size rather than plasma shape: dropping the spherical tokamaks moves the random forest from 0.938 to 0.936. See [Result 5](results/RESULTS.md#result-5-the-same-jump-iter-asks-for-measured-inside-the-database).

### The cure the diagnosis implies: a power law with a bounded correction

![The hybrid frontier](results/hybrid.png)

Everything above is a negative result, and it is specific enough to build against. Fit the power law, learn a correction on its **log residuals**, and damp that correction by a factor `lambda`. Across the ITER-matched size cut, a boosted-tree correction moves the power law from 0.278 to **0.206**, which makes it the best blind model at that cut and within 6% of IPB98(y,2), a law fitted with those machines included. **The same boundedness that makes a tree useless as a predictor on a larger machine makes it safe as a corrector**, because the quantity it is bounded on is now a residual centred on zero rather than a target that grows with size.

The limits are real and stated in full: the gain is along the size axis only, off it the correction hurts, the hybrid wins at 5 of 8 well-powered cuts rather than all, and cross-validation does not select the rung that turns out to be best. See [Result 6](results/RESULTS.md#result-6-a-model-that-is-flexible-in-range-and-still-extrapolates).

### The intervals are not merely wrong, they are confident

![Conformal coverage](results/conformal.png)

For a next-step device the point error is not the deliverable; the interval is. Split-conformal prediction on the log residuals gives every model a nominal 90% interval.

| model | grouped CV | held-out machine | ITER-matched cut |
|---|---|---|---|
| IPB98(y,2), analytic (not blind) | 90% | 89% | 88% |
| ridge, log-linear | 90% | 83% | 70% |
| hybrid (above) | 90% | 64% | **76%** |
| hist gradient boosting | 91% | 45% | **0%** |
| random forest | 91% | 35% | **3%** |

The control arm works: every model lands within a point of nominal where the exchangeability the method assumes actually holds, which is what licenses reading the rest. Out of distribution it does not. **The random forest's 90% interval covers 3% of the rows across the ITER-matched cut, and the histogram gradient booster's covers none of the 2730.** And the widths do not move: no model's interval changes width by more than 1.5% between the two arms. The intervals do not become vague out of distribution. They stay the same size and miss. See [Result 7](results/RESULTS.md#result-7-the-intervals-are-not-merely-wrong-they-are-confident).

### The linear algebra underneath

![Singular value spectrum and disagreement decomposition](results/singular_value_spectrum.png)

**The model's own feature matrix is rank deficient by two, and this audit found it.** Standardized, the ten log features have rank 8. Two exact dependencies, each confirmed by projection onto the null space at a residual of order 1e-16: minor radius is derived as `a = eps * R`, and the IPB98 prior is a fixed log-linear combination of the other eight features. That second one means a published physics scaling, added as a feature, contributes exactly nothing to a log-linear model, however much it looks like added knowledge. Nothing crashed, because `scipy.linalg.lstsq` inverts through the SVD pseudoinverse and silently returns the minimum-norm member of a two-parameter family.

**Refitting IPB98(y,2) from the database disagrees with the published exponents almost entirely where the data is blind.** Solving three ways from scratch (Cholesky on the normal equations, QR, SVD, agreeing to 8e-13) gives Ip 1.08 against 0.93 and R 1.58 against 1.97, while P and Bt come back essentially exactly. Decomposing that difference along the singular directions of the design matrix: **77% of it lies in the single weakest direction, which carries 0.3% of the matrix's variance**. That weak direction is plasma current traded against machine size, structurally hard to resolve because tokamaks are not designed to vary the two independently.

## Quickstart

```bash
# 1. Install (editable, with dev tooling) into a virtualenv.
#    Use 3.10 or newer: `python3` is still 3.9 on stock macOS, and the venv it
#    builds fails the type check and four tests rather than refusing to install.
python3.12 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e ".[dev]" -c constraints.txt

# 2. Fetch the real HDB5 STD5 dataset (content-hash verified), train, report against IPB98(y,2)
python3 hdb5.py train --download-if-missing

# 3. Ask the question a scaling law exists for: hold out a whole machine, then hold out size
python3 hdb5.py extrapolate
python3 hdb5.py size-extrapolate

# 4. Regenerate every number and figure under results/, then rebuild the page
make results      # the six analyses in dependency order, then site/build_page.py

# 5. Reproduce the CI quality gate locally
make check        # == ruff check . && mypy . && pytest -q

# 6. Check that results/ still follows from the raw data and the prose still matches it
make reproduce    # regenerates, compares numerically, then runs the prose claims
```

## What is in the repository

The real-data confinement study is the whole of the argument above:

- `hdb5.py` is the pipeline behind every reported result: cleaning, features, the model zoo, the IPB98(y,2) baseline, and the leave-one-machine-out and size-ordered splits.
- `scaling_law.py` treats a confinement scaling law as the least-squares problem it is, with the three classical solvers written by hand plus the conditioning, null-space and bootstrap analysis behind Results 1 to 3. It deliberately does not call scikit-learn.
- `analysis_*.py` are the six scripts that regenerate every number and figure under `results/`.
- `lawson.py` is a standalone triple-product and ignition-ratio calculation.

## Infrastructure: the Neutron-Yield Pipeline

**Nothing in this section supports a scientific claim.** `train_model.py` and the `neutron_yield/` package predict `neutron_yield` from plasma operating conditions, and the dataset they ship with is synthetic, generated from a hand-crafted signal, so any accuracy number they produce measures how learnable that generator is and nothing else. None of the results above come from it, and it shares only `config.py` and `storage.py` with the pipeline that does.

It is here as engineering rather than as a result: a complete training and inference stack built to fail loudly instead of drifting silently, with a **versioned preprocessing contract** checked before any prediction, **atomic run publishing** so a crash never leaves a half-written run for the loader to find, and a saved-model wrapper that enforces both **even on a bare `joblib.load(...).predict(...)`** that bypasses the inference API. Full operating detail is in [docs/neutron-yield-pipeline.md](docs/neutron-yield-pipeline.md).

## Documentation

| | |
|---|---|
| [results/RESULTS.md](results/RESULTS.md) | The full writeup: every result, mechanism, table and limitation. |
| [docs/usage.md](docs/usage.md) | Installation and the command line for every pipeline, plus the modeling notes and assumptions. |
| [docs/testing.md](docs/testing.md) | How the suite is organised, what CI enforces, and how `results/` is checked against the raw data. |
| [docs/repository.md](docs/repository.md) | File-by-file layout, module ownership, and repository-level caveats. |
| [docs/neutron-yield-pipeline.md](docs/neutron-yield-pipeline.md) | Operating detail for the synthetic-data infrastructure above. |
| [paper/README.md](paper/README.md) | How to build the paper, and the Zenodo DOI flow. |

## Sources

- ITER Physics Basis, *Nucl. Fusion* **39** 2175 (1999). IPB98(y,2) scaling.
- G. Verdoolaege et al., *Nucl. Fusion* **61** 076006 (2021). The HDB5 database.
- ITPA Global H-mode Confinement Database, STD5 v5.2.3, <https://osf.io/drwcq>. If you use this dataset, cite Verdoolaege et al.
