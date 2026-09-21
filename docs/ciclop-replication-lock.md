# CICLOP replication: prospective analysis lock

Status: in force from the commit that adds this file. Written 19 September 2026,
before the CICLOP file was obtained and before any model was fitted to CICLOP
data. The hash of that commit is the lock, and it predates the commit that
records the file's SHA-256. A later change to this file is an amendment, entered
in the deviations log at the end with its date and its reason.

The facts this draft rests on are in
[ciclop-dataset-audit.md](ciclop-dataset-audit.md).

## The question

When the same broad model comparison is moved from within-known-device
validation to prediction of an entirely held-out physical device, how does model
performance change in CICLOP?

The question is not whether another dataset can be found where random forests
lose. The result is reported whichever of the five verdicts in section 9 it
lands on, including the one that says CICLOP cannot answer.

## 1. Dataset

The IAEA/IEA CICLOP database, version 7.3 (Litaudon et al., Nucl. Fusion 66,
095001, 2026), or the version the IAEA serves on the day access is granted if
7.3 has been superseded. The file is analysed as delivered and pinned by SHA-256
and byte count, under the same rules as `hdb5.HDB5_STD5_SHA256`.

CICLOP is the only external fusion dataset this replication uses. If it turns
out not to be evaluable, the manuscript says so. Any other external dataset
needs its own lock written before its scores are seen, and the CICLOP outcome is
reported whatever a later dataset shows.

## 2. Population

Tokamak pulses only. LHD and W7-X are excluded, because the manuscript is about
tokamak scaling and a stellarator has no plasma current in the sense the power
law uses.

All confinement regimes are kept in the primary analysis. The ranking claim is
between models refitted inside each fold, none of which uses a regime-specific
law, and the long-pulse devices that HDB5 lacks run largely in L-mode, so an
H-mode restriction would remove the devices that make CICLOP worth having.
Regime is not added as a feature, because the comparison is on the manuscript's
features.

Rows are dropped only for a missing, non-finite or non-positive value in the
target or in a used feature. This is `hdb5.analysed_row_mask` applied to the
resolved feature set. No outlier rule, winsorising or reweighting is applied.

## 3. Target

The energy confinement time as CICLOP delivers it, modelled and scored in
natural logs. If the file carries both a thermal and a total value, the thermal
one is used, to match `TAUTH`. A difference in definition between facilities is
a stated limitation and is not corrected.

## 4. Features

The manuscript's nine: plasma current, toroidal field, density, power, major
radius, elongation, inverse aspect ratio, effective mass and minor radius, all in
natural logs. Where CICLOP offers more than one candidate for a quantity, the
choice follows a fixed order of preference.

- Density: line-averaged electron density, then core electron density, then core
  ion density.
- Power: a loss or absorbed power if the file has one, then injected additional
  power. Radiated power is not subtracted.
- Elongation: the areal elongation if the file distinguishes it, then whatever
  single elongation it carries.
- Inverse aspect ratio: minor radius divided by major radius.
- Effective mass, from the fuel species label: hydrogen 1, deuterium 2,
  tritium 3, helium 4. A D-T pulse takes the number-weighted mean if the file
  gives fractions, and 2.5 if it gives only the label.

A column is usable if it is finite and positive on at least 90% of tokamak rows.
An unusable column is dropped from the feature set for every model, and the
manuscript's table for CICLOP names what was dropped. Usability is decided from
missingness alone. The rule is not all nine or nothing: the frozen set is the
largest subset of the nine that CICLOP supports.

HDB5 is rerun on that identical frozen set, with the same models, splits and
aggregation, and the CICLOP result is compared with that rerun. The manuscript's
nine-feature headline is not the comparator unless all nine survive. Where
CICLOP's definition of a quantity differs from STD5's and DB5.2.3 carries the
matching column, the rerun uses the matching column, joined through
`replication.db523_columns`. Power is the case that matters. DB5.2.3 records the
power each heating system injected or coupled, beside the loss power `PLTH`, and
the counterpart of CICLOP's injected additional power is the sum
`PINJ + PINJ2 + PICRHC + PECRHC`, with a missing or negative entry counted as
zero. On the STD5 rows that sum is positive on 6196 of 6228 and its median ratio
to `PLTH` is 1.04. `PINJ` alone is one neutral beam system and is not the total.
A plasma with no auxiliary heating has zero injected power and no logarithm, so
the cleaning rule of section 2 removes it. On STD5 that is 32 rows, 23 of them
the ohmic H-modes of COMPASS and TCV, which takes both devices below ten rows.
The same holds for any CICLOP pulse without auxiliary heating.

HDB5 scores are already known, so this rerun can happen before CICLOP is scored
without unblinding anything. The rerun holds out physical devices, and it is
reported at the 10-row threshold CICLOP is scored at and at the manuscript's 30.

The column mapping is written by hand during the schema pass, from column names,
units and label values, following the orders of preference above. It is recorded
in `ciclop_analysis_plan.json` with the rank taken for each quantity. Everything
downstream of the mapping is computed by code.

The confinement factor relative to a scaling law, the stored energy, the fusion
triple product and the pulse duration are never features. The first three
contain the target, and the fourth is what the pulses were selected on.

## 5. Devices

The held-out unit is the physical device. JET with the carbon wall and JET with
the ITER-like wall are one device, which is the manuscript's own rule. Tore Supra
and WEST are one device, because WEST reuses Tore Supra's vacuum vessel and
toroidal field coils, and holding out WEST with Tore Supra still in training
would not hold out an unseen machine.

## 6. Models

The manuscript's models with their settings and seed unchanged, as built by
`hdb5.build_model_zoo`: ridge on the log features (`alpha=1.0`, SVD solver,
standardised inside the pipeline), random forest (300 trees, `max_features=1.0`),
histogram gradient boosting at its scikit-learn 1.7.2 defaults, and the mean
baseline, all seeded at 42. No hyperparameter is tuned on CICLOP for any reason,
including the small sample. The polynomial ridge controls of degree 2 and 3 are
scored as the manuscript scores them, outside the ranking.

The headline pair is the random forest against the log-linear ridge. The
gradient booster against the same ridge goes through the same rule and is
reported second.

## 7. Splits and scores

Within known devices: `GroupKFold` with five folds and no shuffling, grouped on
facility and pulse number, over every tokamak row that survives cleaning. With
one row per pulse this is five-fold cross-validation over rows. If a pulse
number repeats within a facility, the grouping keeps its rows together.

Held-out device: each eligible device in turn, trained on every other tokamak
row, including the rows of devices too small to score.

The score is the root-mean-square error in log confinement time. Both splits are
summarised in the same way, so that only the split differs between them. The
log-RMSE is taken per eligible device, from out-of-fold predictions for the first
split and from held-out predictions for the second, and then averaged with each
device counting once. Call these `cv` and `lodo` for a model. The table in the
manuscript's own format, with cross-validation pooled over all rows, is reported
beside it and plays no part in the verdict.

This matched comparison is stricter than the matched row of the manuscript's
robustness table. That row reruns cross-validation on the scored labels' rows
alone, so its training folds lose the small machines that the holdout folds
keep. Here both splits draw their training rows from the same universe and score
the same rows, and the one thing that differs is whether the scored device's
other pulses are in training. The HDB5 rerun of section 4 uses this definition
too, so the two datasets are compared like for like.

## 8. Eligibility

A device is scored if it has at least 10 rows after cleaning. Ten is the lowest
value in the threshold sweep the manuscript already publishes, so it is not a
number chosen for CICLOP. The manuscript's own 30 rows is run as a sensitivity.
On the published counts it leaves two or three devices, which section 9 treats as
too few to evaluate.

## 9. The verdict

Let k be the number of eligible devices and W the number of them on which the
forest's held-out log-RMSE is higher than the ridge's. Let
`D = (lodo_forest / cv_forest) / (lodo_ridge / cv_ridge)`, which says how much
more the forest degrades than the ridge does. On HDB5 the same quantity, from
the matched row of the manuscript's robustness table, is 3.3.

The first condition that holds gives the verdict.

1. Cannot be evaluated. Access was refused or never granted, or the target is
   unusable, or fewer than six of the nine features are usable, or fewer than
   100 tokamak rows survive cleaning, or k is below 5.
2. Reproduces the inversion. `cv_forest < cv_ridge`, `lodo_forest > lodo_ridge`,
   W is more than half of k, and D is at least 1.5.
3. Contradicts the original result. `lodo_forest < lodo_ridge`, W is less than
   half of k, and D is at most 1.
4. Degradation without a literal inversion. D is at least 1.5.
5. No important differential degradation. D is below 1.5.

If verdict 1 holds for a reason that still lets the models be fitted, such as k
of 4, the scores are computed and reported as descriptive, and the verdict stays
"cannot be evaluated".

Verdict 2 carries the same size requirement as verdict 4. Without it, a forest
that won cross-validation by 0.0001 and lost the holdout by 0.0001 on four of
seven devices would count as a reproduction, and the manuscript itself treats a
gap of 0.002 as inside its resolution. A literal crossing with D below 1.5 falls
through to verdict 5, and the text then says that the ranks crossed and that the
crossing was small.

The 1.5 is a judgement made in advance: it is about a third of the nine-feature
HDB5 effect on a log scale. It does not move when the feature-matched HDB5 value
of D is computed, and that value is reported beside the CICLOP one. The verdict
is a label for the pattern. The magnitudes and intervals of section 10 are
reported beside it whatever it says.

## 10. Statistics

These match the manuscript. The paired percentile bootstrap resamples devices,
with 2000 resamples seeded at 20240617, on the per-device difference between the
forest and the ridge, and again for the booster. An exact sign test is run on W.
Both are read as the manuscript reads them, as summaries of spread over devices
that share most of their training rows, and not as tests over independent
replicates.

With k at most 7, the smallest two-sided p an exact sign test can return is
0.016, and at k of 5 it is 0.0625. The intervals will be wide. A verdict of 4 or
5 is therefore weak evidence against the original result, and a verdict of 2 on
seven devices is weak evidence for it. The manuscript says both.

## 11. What is reported

The replication exists to answer one objection, that the HDB5 result belongs to
one data compilation, and it does not repeat the manuscript on a second dataset.
Three outputs carry it, and each is reported for CICLOP beside the
feature-matched HDB5 rerun.

1. Each model's error under matched cross-validation and under device holdout,
   with the ratio between them.
2. The paired comparison of the forest and the ridge across held-out devices:
   the mean gap, its bootstrap interval, and W of k.
3. The rank correlation between each model's per-device error and the
   Mahalanobis distance of that device from its training rows, as
   `hdb5.extrapolation_diagnostic` computes it. At most seven points carry this
   correlation, so it is described and not tested.

The rest are secondary. Each is reported, and none of them can change the
verdict.

- the 30-row threshold
- H-mode pulses only, if that subset passes the floors in verdict 1, and marked
  as not evaluable if it does not
- Tore Supra and WEST as separate devices
- both splits pooled over rows in place of averaged over devices
- the linear-plus-RBF Gaussian process of `gp.py`, settings unchanged
- IPB98(y,2) evaluated on the H-mode rows as a historical reference with no rank,
  labelled as using injected power if the file has no loss power
- the count of (facility, pulse) pairs that also appear in DB5.2.3, and the
  primary analysis with those rows removed
- if the file carries a pulse date, the primary analysis with the
  cross-validation folds grouped by device and calendar year and not by pulse.
  CICLOP is a database of record pulses, so one device's pulses from one campaign
  are often repeats of one scenario, and a fold boundary between two repeats
  flatters any model that interpolates. If the file carries no date, this is
  reported as not computable
- outputs 1 and 2 over the devices HDB5 does not contain, which on the published
  counts are EAST, KSTAR, and Tore Supra with WEST. This is descriptive, because
  at most three devices can be scored

Conformal coverage is not run. Ten rows per device cannot estimate a coverage
rate. Anything not listed in this document is exploratory and is labelled so
wherever it appears.

## 12. Order of operations

1. This document is committed with its decisions closed.
2. The pipeline is written and tested against a synthetic frame that has the
   CICLOP schema. It fits and scores through the same `hdb5` and
   `analysis_robustness` functions the manuscript's numbers come from, in the
   way `replication.py` does for DB5.2.3.
3. The author obtains the file. Its SHA-256, byte count, version, access date
   and terms of use are committed before any model is run on it. Access was
   requested from the site's owners on 20 September 2026. If it has not been
   granted by 18 October 2026, the manuscript is submitted without CICLOP, and
   its Limitations says that a candidate external database could not be obtained
   in time. That date governs the submission and not the replication: this lock
   stays in force, and if access comes later the analysis is run under it
   unchanged and reported, whatever it shows, at revision or on its own.
4. A schema pass reads column names, units, per-column summaries, missingness
   and counts. It computes nothing that joins the target to a feature, fits
   nothing, and evaluates no published scaling law.
5. The rules of sections 2 to 8 are applied to what the schema pass found. The
   outcome is written to `ciclop_analysis_plan.json`: the file's SHA-256, the
   column mapping, the frozen feature set with the reason for each omission, the
   device definitions, the row counts, the eligible devices, and every threshold
   in this document. That file is committed. Its commit hash is what the
   manuscript cites when it calls the analysis prospectively specified, so it has
   to predate step 7. The same commit sets two pins in `ciclop.py`, the SHA-256
   of the data file and the SHA-256 of the plan. They ship unset, and
   `analysis_ciclop.py` refuses to fit anything until both are set and both
   match, so the order of these steps is enforced by the code and recorded by
   the history.
6. HDB5 is rerun on the frozen feature set.
7. The analysis is run once on the real file, and that run is the run of record.
   A rerun forced by a bug is entered in the deviations log with the bug, and
   both sets of output are kept. Any change to `ciclop_analysis_plan.json` after
   its commit is entered in the same log.

## 13. Not allowed after the first score

Changing the threshold, the feature set, the device definitions, the population,
the metric or the verdict rule. Tuning any model. Dropping a device or a row for
a reason not already in this document. Replacing CICLOP with another dataset, or
adding one, because of what the CICLOP scores show.

## Decisions closed

The author settled four points in a note of 19 September 2026, before any data
was obtained, and the sections above carry them: tokamaks only (section 2), the
largest common feature subset with HDB5 rerun on it (section 4), matched
aggregation as the basis of the claim (section 7), and three core outputs in
place of a second copy of the manuscript (section 11).

The draft listed the judgement calls it had made on the author's behalf. On the
same day, still before any data was obtained, the author instructed that the
draft proceed, and each was closed at the draft's default.

1. All confinement regimes in the primary analysis, with H-mode only as a
   sensitivity (sections 2 and 11).
2. Tore Supra and WEST as one device, with the split as a sensitivity (sections 5
   and 11).
3. Ten rows as the primary threshold, with 30 as a sensitivity (section 8).
4. The floors in verdict 1: five devices, 100 rows, six of nine features.
5. The 1.5 boundary between verdicts 4 and 5.
6. The HDB5 rerun matches CICLOP's power definition where CICLOP carries
   injected power, through the sum of injected and coupled powers given in
   section 4.

Two items from the draft were not decisions about the analysis. The brief that
asked for this lock arrived cut off, and its remainder was never received, so
the lock stands on this document and the author's note. The per-device counts in
the audit were read through a summarising tool and not from the papers' PDFs.
No rule here depends on them: every rule is applied to the counts in the file
itself, so an error in the audit's transcription cannot change the analysis.

Any of these can be amended before step 3 by a commit that says so in the log
below, and none of them after it.

## Deviations log

| Date | What changed | Why | Before or after the first score |
| --- | --- | --- | --- |
| 19 September 2026 | Section 4 and closed decision 6: the HDB5 counterpart of injected power is `PINJ + PINJ2 + PICRHC + PECRHC`, where the lock as first committed named `PINJ` alone. | Measured on the STD5 rows while building the pipeline, `PINJ` has a median ratio to `PLTH` of 0.66 and is zero on 361 rows heated by radio frequency alone, so it is one beam system and not the total. The check used HDB5 only. | Before. No CICLOP file had been obtained. |
| 19 September 2026 | Section 4 states a consequence of the injected-power definition: a plasma with no auxiliary heating is removed by the cleaning rule. No rule changed. | Found when the HDB5 rerun was first exercised: its 10-row and 30-row arms held out the same 11 devices, because 32 STD5 rows have zero injected power. | Before. No CICLOP file had been obtained. |
| 20 September 2026 | Section 9: verdict 2 also requires D of at least 1.5. | A critical re-read found that "reproduces the inversion" had no size requirement while verdicts 4 and 5 did, so a crossing inside the noise would have counted as a reproduction. The change makes the favourable verdict harder to reach. | Before. No CICLOP file had been obtained. |
| 20 September 2026 | Section 11: a conditional secondary analysis with cross-validation folds grouped by device and calendar year. It cannot change the verdict. | Near-repeat pulses within a campaign are the bias the audit expects most, and nothing in the lock measured it. | Before. No CICLOP file had been obtained. |
| 20 September 2026 | Section 12, step 3: a date, 18 October 2026, after which the manuscript is submitted without CICLOP. The lock stays in force past it. | Access is granted by the site's owners and may take weeks or be declined. | Before. No CICLOP file had been obtained. |
