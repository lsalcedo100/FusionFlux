# scaling-audit

**Your model is better under cross-validation. Ask it about a group it has never seen.**

A scaling law is a power law fitted across the systems you have already built, used to predict one you have not. Machine learning usually beats the published law under cross-validation. On three datasets from three sciences, that result reverses when an entire group is held out, and it reverses for a reason you can measure in advance.

This package is the audit, with nothing domain-specific in it. Three independent pieces; use whichever applies.

```bash
pip install scaling-audit          # once released; see Provenance below
```

## The problem it exists for

Grouped cross-validation by *record* holds out rows while leaving every group in the training fold. That measures interpolation inside systems the model has already seen. A scaling law exists to answer a different question: predict a system that was not in the training set at all.

Those two numbers can rank models in opposite orders. On the ITPA tokamak confinement database a random forest beats the published physics law by 41% under record-level cross-validation, and is worse than it on **13 of 13** machines when an entire machine is held out. The standard validation does not merely overstate the gain, it reverses the ranking.

Reporting the held-out score alone makes that look like noise. The point of this package is that it is not noise and it is not a surprise: two cheap diagnostics predict it.

## `audit_groups` — the score, and why the score is what it is

Leave-one-group-out, reporting per group the score *alongside* the two quantities that explain it.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from scaling_audit import audit_groups, distance_score_correlation, summarize

report = audit_groups(
    frame[features],
    np.log(frame["target"]),            # log target in, so the score is RMSLE
    frame["machine"],
    {"ridge": Ridge(), "forest": RandomForestRegressor()},
    min_held_out_rows=30,
)

summarize(report)                        # mean, median, worst and the correlation
distance_score_correlation(report)       # the diagnostic that matters
```

Each row carries:

| field | what it tells you |
|---|---|
| `score` | RMSE on the held-out group, on whatever scale `y` is in |
| `mahalanobis` | how far the held-out group sits from the training data, in training-covariance units |
| `fraction_above_train_max` | fraction of held-out rows whose target exceeds anything a range-bounded model can emit |
| `log_target_headroom` | how far above the training maximum the group reaches |
| `prediction_bounded_by_train_range` | whether this estimator's predictions were in fact confined to the training range |

**Read the correlation, not just the mean.** A model whose per-group error tracks `mahalanobis` is failing *because* the group is far away, which is a statement about what it will do on your next system. A model whose error is flat against distance is not.

On the tokamak data the random forest scores **+0.85** and the log-linear power law **-0.06**. On mammalian metabolic rate, +0.64 against +0.39. Same diagnostic, different science, same sign.

`fraction_above_train_max` is the harder bound. A tree ensemble averages training targets, so it cannot output a value above the largest one it has seen, whatever the features say. When 48% of a held-out group lies above that ceiling, the model is not merely uncertain there: it is structurally incapable, and no amount of tuning changes it.

### Before you fit anything: `group_diagnostic`

Both diagnostics are properties of the *split*, not of any model, so they can be computed before a model exists and before the held-out targets do. That is the useful case: a device or a batch you have not built yet has features but no measurements.

```python
from scaling_audit import group_diagnostic

d = group_diagnostic(X, y, train_index, test_index, group="ITER")
d.mahalanobis                            # how far outside the data this sits
d.fraction_above_train_max               # what no range-bounded model can reach
d.log_target_headroom
```

`group_diagnostic` returns a `GroupDiagnostic`, and `audit_groups` is essentially this run over every group with each estimator scored beside it.

## `OrderedGroupSplit` — the split a scaling law actually has to survive

Leave-one-group-out still surrounds the held-out group with others like it. If you hold out one tokamak, twelve remain that span its size range, so you are extrapolating in *identity* while interpolating in *size*. A next-step system is outside the data on the axis you care about.

```python
from sklearn.model_selection import cross_val_score
from scaling_audit import OrderedGroupSplit

# `order` maps each group label to a scalar: the axis you are extrapolating along.
sizes = frame.groupby("machine")["radius"].median().to_dict()
split = OrderedGroupSplit(sizes, min_train_groups=3, min_test_rows=30)

cross_val_score(model, X, y, cv=split, groups=frame["machine"])
```

Train on one end of an ordering, predict the far end. It is a scikit-learn splitter, so it drops into anything that takes a `cv`.

This is where the differences get large. At a size cut matched to the jump a next-step device asks for, the tree ensembles in the tokamak study land closer to predicting a constant than to the published law they beat by 41%.

## `ConstrainedLinearRegression` — a physics assumption as a constructor argument

Equality-constrained least squares: minimise `||Xb - y||²` subject to `Cb = d`, solved through the KKT system.

```python
from scaling_audit import ConstrainedLinearRegression

# Kleiber's law: metabolic rate scales as mass^(3/4), exponent held rather than fitted.
# With fit_intercept=True the design is [intercept, log_mass], so the constraint
# row names the second column and leaves the intercept free.
model = ConstrainedLinearRegression(constraint=[[0.0, 1.0]], rhs=[0.75])
model.fit(np.log(mass).reshape(-1, 1), np.log(rate))

model.coef_                              # array([0.75])
model.constraint_violation()             # 1.2e-15
```

Dimensional analysis, when it applies, imposes *linear equality constraints on log-linear exponents*. That makes a physics assumption a matrix rather than a rewrite, and it is often the cheapest thing that helps: a constraint names a surface rather than a point, which is weaker information than a prior and in the tokamak study was worth considerably more.

## Three worked domains

The same three objects, unchanged, on three datasets that share no measurement conventions:

| domain | groups | published law | what the audit found |
|---|---|---|---|
| **Tokamak confinement** (ITPA HDB5, 6228 records, 18 devices) | machine | IPB98(y,2) | Forest beats the law by 41% under record-level CV; loses on **13 of 13** held-out machines. Error against distance: +0.85 for the forest, -0.06 for the law. |
| **Mammalian metabolic rate** (541 species, 11 orders) | taxonomic order | Kleiber, mass^(3/4) | Trees lose to both power laws at all 8 mass cuts. The reversal does *not* appear: with one predictor the trees never win the easy split either. |
| **Tree allometry** (BAAD, 3599 plants, 53 species) | species | West-Brown-Enquist, diameter^(8/3) | Power law wins the held-out species at **4 of 4** feature counts. The reversal appears at exactly 3 predictors, where the trees first win interpolation. |

The third resolves what the first two leave ambiguous, and it is the practically useful part. **The extrapolation failure is unconditional; the ranking reversal is not.** Varying only the number of predictors on a fixed set of plants, the interpolation gain runs -1.7%, -0.6%, +1.8%, +6.8% while the extrapolation deficit stays flat and slightly grows. The reversal appears exactly where the interpolation gain crosses zero.

So the reversal is the *warning light*, not the disease. It is what makes a flexible model look good enough to adopt in the first place. Below three predictors the danger is identical and the light is off.

## What it does not do

- It reports RMSE on whatever `y` you pass. Pass a log target and you get RMSLE; the transform is yours because only you know whether your target is positive.
- It does not choose your groups. If your groups are not the unit your claim is about, none of this helps.
- `distance_score_correlation` over a handful of groups is a rank correlation over a handful of points. The tokamak study computes it over 13 and says so.
- It is not a fix. It tells you when the comparison you ran does not answer the question you asked.

## Provenance

Extracted from [FusionFlux](https://github.com/lsalcedo100/FusionFlux), a study of confinement scaling on the ITPA database, where it is the module deliberately containing no plasma physics. The repository runs its own analyses through this exact file rather than a copy, so "domain-agnostic" is tested rather than asserted. Every number quoted above regenerates from raw, SHA-256-pinned data.

This package is not on PyPI yet. Until it is, use it from a checkout of the repository, where
`scaling_audit.py` sits at the root and imports nothing else in the project.

MIT licensed.
