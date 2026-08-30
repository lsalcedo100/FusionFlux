# Confinement scaling as a linear algebra problem

Three results on the ITPA global H-mode confinement database (HDB5, standard
analysis set STD5), regenerated end to end by `python3 analysis_scaling_law.py`.

**Data.** 6228 quasi-stationary time slices from 4471 discharges across 18
tokamaks (JET, ASDEX Upgrade, DIII-D, JT-60U, C-Mod, NSTX, MAST, START and
others). Third-party scientific data, fetched from OSF rather than
redistributed here; `python3 hdb5.py download` retrieves it. No synthetic data
appears anywhere in this document.

## Why this is a linear algebra problem at all

A confinement scaling law is a power law,

    tau_E = C * Ip^a1 * Bt^a2 * ne^a3 * P^a4 * R^a5 * eps^a6 * kappa^a7 * M^a8

so in log space it is exactly a linear model,

    log tau_E = log C + a1 log Ip + a2 log Bt + ... + a8 log M

and fitting a scaling law is ordinary least squares on a log design matrix.
Every question about the physics becomes a question about that matrix. All three
results below are properties of the matrix rather than of any particular model.

The published baseline is IPB98(y,2):

    tau_E = 0.0562 * Ip^0.93 * Bt^0.15 * ne19^0.41 * P^-0.69
                   * R^1.97 * eps^0.58 * kappa_A^0.78 * M^0.19

with Ip in MA, Bt in T, ne in 10^19 m^-3, P in MW, R in m. The leading
coefficient is quoted as 0.0562 in the ITER Physics Basis and rounded to 0.056
in several later summaries; we carry 0.0562 and note the difference rather than
picking one silently. It shifts log C by 0.0036, far inside the interval fitted
below, so nothing here depends on the choice. `kappa_A` is areal elongation,
which is the `KAPPAA` column of HDB5 rather than the separatrix elongation.

---

## Result 1: the model's own feature matrix is rank deficient by exactly two

The confinement model in `hdb5.py` is trained on ten log features. Standardized,
that matrix has **rank 8 of 10**, with the last two singular values at 4.9e-14
and 2.4e-14 against a numerical-zero tolerance of 2.3e-10.

Two exact dependencies, confirmed by projecting the expected vector onto the
null space (`ConditioningReport.null_space_residual`, relative norm):

| dependency | kind | residual |
|---|---|---|
| `log a = log R + log eps` | definitional: minor radius is *derived* as `a = eps * R` in cleaning | 7.3e-16 |
| `log tau_IPB98 = 0.93 log Ip + 0.15 log Bt + ... + 0.19 log M` | the IPB98 prior is a fixed log-linear combination of the other eight features | 9.9e-16 |
| `log Ip` alone (control) | not a dependency | 8.9e-01 |

The second one is the interesting one. Adding a published physics scaling as a
model feature feels like adding knowledge. In log space it provably is not: the
prior is a fixed linear combination of features the model already has, so it
contributes nothing to a log-linear fit. It does help the tree models, which
cannot form that combination themselves, and that is the honest reason to keep
it. It contributes exactly zero to the ridge model it sits beside.

**Why nothing ever crashed.** `scikit-learn`'s linear models call
`scipy.linalg.lstsq`, which inverts through the SVD pseudoinverse and returns
the *minimum-norm* solution. Of the two-parameter family of coefficient vectors
that fit identically well, it silently returns the shortest. No error, no
warning, and no metric moves. `tests/test_scaling_law.py` demonstrates this: it
takes the returned solution, adds a null-space vector, and confirms the
predictions are unchanged to 1e-9 while the coefficients are not.

### Two ways this analysis goes wrong, both of which it went wrong before

**Reading the printed basis instead of projecting.** The null space is a
*subspace*, and the SVD returns an arbitrary orthonormal basis for it. When the
deficiency is greater than one, the vectors numpy prints generally will not
resemble the dependency you are looking for even when you are exactly right. The
correct test is whether your vector lies in the span, `||v - N^T N v||`.

**Comparing vectors across coordinate systems.** Standardizing rescales the null
space. A dependency `sum_j c_j x_j = 0` among raw columns becomes `c_j * s_j`
once each column is divided by its standard deviation. Checking the raw `c`
against a standardized null space gives residuals of 0.41 and 0.96 here, for
dependencies that are exactly true. That is indistinguishable from being wrong.
Both traps are pinned down by tests.

**Standardize before reading the rank.** `numpy.linalg.matrix_rank` on the
unstandardized matrix reports rank **9**, not 8, because its tolerance scales
with the largest singular value and one column's units dominate. Reported rank
is otherwise a statement about someone's choice of SI prefix rather than about
the data.

---

## Result 2: refitting IPB98(y,2) from the database

Solved three ways, implemented from scratch in `scaling_law.py` (including the
triangular substitutions) rather than by calling `scikit-learn`. Design matrix
6228 x 9, condition number 10.7. Timings are per solve, averaged over 20 runs.

| solver | time | max deviation from the SVD solution |
|---|---|---|
| normal equations, Cholesky | 0.073 ms | 7.6e-13 |
| QR | 0.343 ms | 2.2e-15 |
| SVD pseudoinverse | 0.395 ms | 0 (reference) |

Cholesky is five times faster and three orders of magnitude less accurate, which
is the expected trade: it forms `X^T X` and therefore works at the *square* of
the condition number. At cond 10.7 that costs nothing real. It is also the only
one of the three that fails loudly on a rank-deficient matrix, which is a
feature.

Exponents, with 95% percentile bootstrap intervals over 1000 resamples,
**resampling whole discharges** rather than rows (the several time slices from
one shot are not independent observations; row-level resampling returns
intervals that are too narrow):

| variable | refit | 95% CI | IPB98(y,2) | published value inside CI |
|---|---|---|---|---|
| Ip | 1.080 | 1.034 to 1.130 | 0.93 | no |
| Bt | 0.120 | 0.080 to 0.163 | 0.15 | **yes** |
| ne | 0.217 | 0.190 to 0.242 | 0.41 | no |
| P | -0.688 | -0.707 to -0.672 | -0.69 | **yes** |
| R | 1.579 | 1.502 to 1.647 | 1.97 | no |
| eps | 0.167 | 0.060 to 0.264 | 0.58 | no |
| kappa | 0.898 | 0.833 to 0.966 | 0.78 | no |
| M | 0.256 | 0.221 to 0.290 | 0.19 | no |

Coefficient: 0.0548 fitted against 0.0562 published. Loss-power and toroidal
field come back essentially exactly; density, major radius and inverse aspect
ratio do not.

**This disagreement is expected, and pretending otherwise would be the error.**
IPB98 was fit to a selected ITER-relevant subset under the ITPA's own standard
selection criteria. The fit above uses every STD5 row that is finite and
positive, which includes the spherical tokamaks (NSTX, MAST, START) whose
inverse aspect ratio sits far outside the conventional range, and it applies no
selection filters at all. A refit on a different population is a different
regression. The comparison is still worth making, because of Result 3.

Against the published law on the same rows, in-sample: RMSLE **0.181** for the
refit against **0.199** for IPB98(y,2). Out of sample, under grouped
cross-validation by discharge (`python3 hdb5.py evaluate`):

| model | CV RMSLE | CV R^2 (log) |
|---|---|---|
| random forest | 0.118 | 0.981 |
| histogram gradient boosting | 0.119 | 0.981 |
| ridge, log-linear | 0.181 | 0.957 |
| **IPB98(y,2), analytic, no training** | **0.199** | **0.947** |
| mean baseline | 0.869 | 0.000 |

41% lower RMSLE than the published scaling law, against a real physics baseline
rather than against the mean.

---

## Result 3: the disagreement lives where the data is blind

![singular value spectrum](singular_value_spectrum.png)

The design matrix is **not** ill conditioned: cond 10.7, which is unremarkable.
So the individual exponents are determined, and the interesting question is not
whether but *where* the refit and IPB98 part ways.

Decomposing the difference between the two exponent vectors along the singular
directions of the design matrix (after mapping into standardized coordinates,
where the singular vectors live):

| direction | singular value | share of what the data determines | share of the disagreement |
|---|---|---|---|
| 1 | 135.6 | 36.9% | 0.6% |
| 2 | 114.6 | 26.4% | 0.1% |
| 3 | 95.5 | 18.3% | 0.1% |
| 4 | 63.6 | 8.1% | 1.8% |
| 5 | 50.1 | 5.0% | 2.3% |
| 6 | 38.2 | 2.9% | 10.6% |
| 7 | 31.7 | 2.0% | 7.1% |
| **8** | **12.7** | **0.3%** | **77.4%** |

**77% of the disagreement lies in the single weakest direction, which carries
0.3% of the design matrix's variance.** The three strongest directions carry 82%
of the variance and account for 0.75% of the disagreement. The two laws agree on
essentially everything the database determines well and differ almost entirely
in the combination it barely constrains.

That weakest direction is

    -0.62 log Ip + 0.55 log R + 0.43 log eps + 0.29 log Bt + ...

which is plasma current traded against machine size. It is weak for a structural
reason rather than a statistical one: tokamaks are not designed to vary current
and major radius independently, so no amount of additional shots from existing
machines separates them. This is why published scaling laws can disagree
noticeably on individual exponents while predicting almost identically over the
range where they were fit, and it is a known problem in the field rather than an
artifact here.

Ridge regression makes the same point constructively. Written through the SVD,
ridge multiplies each singular direction by `s^2 / (s^2 + alpha)`:

| direction | singular value | alpha = 1 | alpha = 100 |
|---|---|---|---|
| 1 | 135.6 | 0.99995 | 0.995 |
| 7 | 31.7 | 0.999 | 0.909 |
| 8 | 12.7 | 0.994 | **0.616** |

At alpha = 100 the seven well-determined directions are untouched and the weak
one is shrunk by 38%. Regularization is not a knob that makes numbers behave. It
is a decision about which physics you are declining to resolve, and the SVD says
exactly which.

---

## Limitations

- **The refit population is not IPB98's population.** No ITPA standard-set
  selection criteria are applied beyond finiteness and positivity. Spherical
  tokamaks are included. The refit exponents should not be read as a correction
  to IPB98.
- **Two machines dominate.** JET and ASDEX Upgrade (including their ILW and W
  variants) supply 4772 of 6228 rows, 77%. The bootstrap resamples discharges,
  not machines; resampling machines would give wider intervals and is the more
  honest uncertainty for a claim about tokamaks in general.
- **In-sample RMSLE favors the refit by construction.** The 0.181 against 0.199
  comparison fits and evaluates on the same rows. The grouped-CV table is the
  one to trust.
- **The rank deficiency was shipped.** It is present in the model that produced
  the results in `data/processed/`, and it was found by this audit rather than
  before the fact. Its practical effect there is limited, because the selected
  models are tree ensembles rather than linear ones, but the ridge model in the
  same comparison was fitting an arbitrary point in a two-dimensional family.
- **A separate synthetic pipeline uses `log1p`, not `log`.** In `features.py`,
  `log_triple_product = log1p(n T tau)`, which is *not* `log1p(n) + log1p(T) +
  log1p(tau)`; the additivity residual reaches 4.6%. The engineered features
  there are near-collinear rather than exactly dependent, and the two exact
  dependencies in that matrix are unit duplications (`ne_20` against
  `fuel_density_m3`, `lawson_ratio` against `triple_product`). Different problem,
  same lesson.

## Reproducing

```
python3 hdb5.py download          # fetch HDB5 STD5 from OSF
python3 analysis_scaling_law.py   # regenerate everything above
python3 -m pytest tests/test_scaling_law.py tests/test_hdb5.py
```

## Sources

- ITER Physics Basis, *Nucl. Fusion* **39** 2175 (1999). IPB98(y,2) scaling.
- G. Verdoolaege et al., *Nucl. Fusion* **61** 076006 (2021). The HDB5 database.
- ITPA Global H-mode Confinement Database, STD5 v5.2.3, https://osf.io/drwcq
