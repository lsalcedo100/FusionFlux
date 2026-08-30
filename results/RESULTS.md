# Confinement scaling as a linear algebra problem

Four results on the ITPA global H-mode confinement database (HDB5, standard
analysis set STD5). Results 1 to 3 are regenerated end to end by `python3
analysis_scaling_law.py`, Result 4 by `python3 analysis_extrapolation.py`.

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
Every question about the physics becomes a question about that matrix. Results 1
to 3 are properties of that matrix rather than of any particular model. Result 4
is what those properties cost you when you try to predict a machine that is not
in it.

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

**Read that table with Result 4 in hand.** Grouped CV holds out *discharges*,
and every machine in the held-out fold also appears in the training fold. It
therefore measures interpolation within machines the model has already seen. On
the split that a scaling law actually exists for, holding out a whole device,
this ranking reverses top to bottom and the 41% becomes a 2.2x loss.

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

---

## Result 4: the model that wins on cross-validation loses on a new machine

Results 1 to 3 are about a matrix built from 18 existing tokamaks. A confinement
scaling law is not for those machines. It is for the next one, and every number
above is quoted under grouped cross-validation by *discharge*, which holds out
some shots from JET and then trains on the rest of JET. Every machine in the
held-out fold also sits in the training fold. That measures interpolation.

So hold out an entire device. Train on 12 tokamaks, predict the 13th, rotate.
Regenerate with `python3 analysis_extrapolation.py`, or `python3 hdb5.py
extrapolate` for the table alone.

![interpolation against extrapolation](extrapolation.png)

**Both columns below use the same nine features and the same models.** The only
thing that changes is what the split holds out. That matters more than it might
look: the model's default feature set includes the analytic IPB98 prior, whose
exponents were fitted on this database *including whichever machine is held
out*, so leaving it in would leak the answer into every fold. Dropping it only
in the extrapolation arm would then confound the feature set with the split, so
it is dropped from both (`hdb5.BLIND_FEATURE_COLUMNS`).

| model | CV, by discharge | leave-one-tokamak-out | ratio | CV rank | LOMO rank |
|---|---|---|---|---|---|
| random forest | 0.128 | 0.465 | **3.6x worse** | 1 | 5 |
| histogram gradient boosting | 0.130 | 0.359 | 2.8x worse | 2 | 4 |
| ridge, log-quadratic (control) | 0.158 | 0.300 | 1.9x worse | 3 | 3 |
| ridge, log-linear | 0.181 | 0.214 | 1.2x worse | 4 | 2 |
| IPB98(y,2), analytic* | 0.199 | 0.188 | 1.0x, unchanged | 5 | 1 |
| mean baseline | 0.869 | 0.994 | 1.1x worse | 6 | 6 |

\* not a blind baseline: IPB98's exponents were fitted on this database, held-out
machine included. It is a reference point for what a power law achieves here,
not a competitor that never saw the data. The ranking claim below is about the
three models that actually fit something and are genuinely blind; the
log-quadratic row is a control introduced in Result 4d and is likewise excluded
from it.

**Among those three, the order under one split is the exact reverse of the order
under the other** (rho = -1.00; with three contenders the reversal itself is the
statistic worth quoting, not the correlation). The random forest is the best
model in the repository by cross-validation and the worst of the three on a
machine it has not seen. Its 41% margin over the published scaling law in Result
2 is not a margin over the published scaling law. It is a measurement of how
much of JET is predictable from the rest of JET.

### Result 4b: the trees fail as a function of distance, and the power law does not

Per machine, ordered by how far it sits outside the training data (Mahalanobis
distance of its mean log-feature vector from the training mean, in training
covariance units, via the pseudo-inverse because that covariance is singular by
Result 1):

| held out | rows | distance | IPB98 | ridge | hist GB | random forest |
|---|---|---|---|---|---|---|
| D3D | 388 | 1.1 | 0.252 | 0.251 | 0.246 | 0.406 |
| AUG | 1377 | 1.2 | 0.197 | 0.216 | 0.281 | 0.279 |
| AUGW | 767 | 1.6 | 0.218 | 0.207 | 0.212 | 0.219 |
| JETILW | 866 | 1.7 | 0.257 | 0.216 | 0.244 | 0.318 |
| JET | 1762 | 2.2 | 0.148 | 0.198 | 0.487 | 0.478 |
| JT60U | 100 | 2.5 | 0.275 | 0.285 | 0.307 | 0.291 |
| PDX | 97 | 4.0 | 0.227 | 0.233 | 0.323 | 0.407 |
| ASDEX | 431 | 5.2 | 0.131 | 0.194 | 0.207 | 0.507 |
| MAST | 39 | 5.4 | 0.136 | 0.154 | 0.317 | 0.581 |
| JFT2M | 69 | 5.4 | 0.095 | 0.094 | 0.234 | 0.450 |
| CMOD | 45 | 6.9 | 0.119 | 0.173 | 0.569 | 0.521 |
| NSTX | 185 | 7.8 | 0.222 | 0.289 | 0.686 | 0.857 |
| PBXM | 59 | 10.2 | 0.172 | 0.274 | 0.559 | 0.727 |

Rank correlation between a model's per-machine error and that distance:

| model | rho |
|---|---|
| random forest | **+0.85** |
| histogram gradient boosting | +0.54 |
| ridge, log-quadratic (control) | +0.25 |
| ridge, log-linear | **-0.06** |
| IPB98(y,2), analytic | -0.49 |

The random forest's errors are explained by extrapolation distance. The power
law's are not: at rho = -0.06 its error is uncorrelated with how far the machine
sits from anything it was trained on. On the four most distant machines (MAST,
CMOD, NSTX, PBXM, all of them small or spherical) the forest is 2.6x to 3.8x the
power law's error; on the four nearest it is within 1.6x.

Both columns of the previous table and this one are monotone in model
flexibility, which is the subject of Result 4d.

### Result 4c: one of the two failure modes is a hard bound, not a shortfall

JET is the visible outlier in the right panel: close to the training
distribution, badly predicted by the trees. That is a second, separable failure
mode.

A tree ensemble predicts by averaging training targets, so **every prediction it
can possibly make lies inside `[min(y_train), max(y_train)]`**, whatever the
features say. When JET is held out, 48% of its rows have confinement times above
the maximum in the remaining 12 machines, and its best shot is **3.7x above
anything any tree in the forest is able to output**. JET is the largest device in
the database, so the rest of the database does not contain its performance
envelope. The forest scores 0.478 there against the power law's 0.198.

This is not a shortfall that more data or better features would close. It is the
functional form. `tests/test_extrapolation.py` asserts the bound directly, by
fitting a forest with a machine held out and checking no prediction exceeds the
training maximum while the held-out truth does.

### Result 4d: it is not enough to be able to extrapolate; the form has to be constrained

Ridge beating the trees on an unseen machine has two candidate explanations, and
the model zoo above cannot tell them apart:

1. the log-linear power-law form is close to physically right, or
2. ridge is simply the only model in the zoo that extrapolates *at all*, since a
   tree ensemble is bounded by its training range by Result 4c.

`ridge_log_quadratic` (`hdb5.build_control_models`) is the discriminating case.
It is a degree-2 polynomial in the log features, so it carries curvature and
every pairwise interaction and is far more flexible than plain ridge, but it is
still a polynomial and so it still extrapolates without bound. If mere
extrapolation ability were what mattered, it should behave like ridge. If
flexibility is what costs you, it should behave like the trees.

It behaves like neither, and lands in between:

| model | flexibility | can extrapolate | LOMO mean | degradation | rho(distance) |
|---|---|---|---|---|---|
| ridge, log-linear | log-linear | yes | 0.214 | 1.18x | -0.06 |
| ridge, log-quadratic | + curvature, interactions | yes | 0.300 | 1.89x | +0.25 |
| hist gradient boosting | nonparametric | no | 0.359 | 2.77x | +0.54 |
| random forest | nonparametric | no | 0.465 | 3.64x | +0.85 |

Every column is monotone in flexibility, and the answer to the question is
therefore "both, and flexibility is the larger term". Being able to extrapolate
buys the log-quadratic model something real: it is better than either tree
ensemble on average, and it never hits the hard ceiling of Result 4c. But it
still gives up most of plain ridge's advantage, so the constraint is doing the
work rather than the mere absence of a bound.

The worst case is the sharper version of the point. On C-Mod the log-quadratic
control scores **1.083**, worse than any other model tested including the mean
baseline's C-Mod score, while plain ridge scores 0.173 on the same machine. A
polynomial extrapolates without bound in *every* direction, including the
weakly-determined one from Result 3, so on a compact high-field machine far from
the training set its curvature terms are free to diverge. The tree ensembles
cannot fail that way, because the bound that makes them useless on JET also
caps how wrong they can be on C-Mod. Unbounded extrapolation is not a virtue on
its own; it is only useful along a form the physics constrains.

All three failure modes point the same way, and it is the same way Result 3
points. A power law is not used in this field because it fits better; Result 2
shows it fits worse. It is used because it is the functional form that survives
leaving the data behind, and ITER is 6.2 m of major radius outside every row in
this table.

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
- **13 of 18 machines are scored, not all 18.** START, TCV, COMPASS, TDEV and
  TFTR have fewer than 30 rows each, too few for a held-out RMSLE to mean
  anything, so they are excluded from Result 4 and remain in every training
  fold. They are also the machines most unlike the rest, so the extrapolation
  gap reported here is if anything the optimistic one.
- **The control is one model, not a family.** Result 4d rests on a single
  log-quadratic ridge. It is the right discriminating case, but "flexibility
  costs extrapolation" would be better supported by a sweep over polynomial
  degree and ridge penalty than by one point, and the C-Mod blowup in
  particular is one machine's behaviour rather than a demonstrated law.
- **Leave-one-tokamak-out understates what ITER faces.** Holding out JET still
  leaves 12 tokamaks spanning much of its parameter range. No held-out machine
  here is outside the database the way a next-step device would be.
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
python3 hdb5.py download           # fetch HDB5 STD5 from OSF
python3 analysis_scaling_law.py    # regenerate Results 1 to 3
python3 analysis_extrapolation.py  # regenerate Result 4
python3 -m pytest tests/test_scaling_law.py tests/test_hdb5.py tests/test_extrapolation.py
```

## Sources

- ITER Physics Basis, *Nucl. Fusion* **39** 2175 (1999). IPB98(y,2) scaling.
- G. Verdoolaege et al., *Nucl. Fusion* **61** 076006 (2021). The HDB5 database.
- ITPA Global H-mode Confinement Database, STD5 v5.2.3, https://osf.io/drwcq
