# ISHCDB replication: prospective analysis lock

Status: in force from the commit that adds this file. Written 21 September 2026,
before the data file was downloaded and before any model was fitted to it. What
had been read is the public directory listing and the database's own
documentation. A later change to this file is an amendment, entered in the
deviations log at the end with its date and its reason.

The facts this rests on are in [ishcdb-dataset-audit.md](ishcdb-dataset-audit.md).

## The question

When the same broad model comparison is moved from validation within known
devices to prediction of an entirely held-out physical device, how does model
performance change in the stellarator-heliotron confinement database?

The result is reported whichever of the five verdicts it lands on. The audit
gives a reason, known in advance, why a power law may transfer badly here, so
"contradicts the original result" is a live outcome and not a formality.

## 1. Relation to the CICLOP lock

This is a second, separate replication. [The CICLOP lock](ciclop-replication-lock.md)
stays in force as archived in v0.4.13 (doi:10.5281/zenodo.22859310), and its
outcome is reported whatever this one shows, including "cannot be evaluated" if
access never comes. No CICLOP score existed when this was written.

Sections 6, 7, 9 and 10 of that lock are adopted here unchanged: the models and
their settings, the two splits and the matched way of scoring them, the five-way
verdict rule with its boundary of 1.5 on D, and the statistics. Where this
document is silent, that one governs. What follows is what differs.

## 2. Dataset

ISHCDB version 26, the file `ISHCDB_26.txt` in the open directory
`https://ishpdb.ipp-hgw.mpg.de/ISS/`. It is analysed as delivered and pinned by
SHA-256 and byte count, under the rules `hdb5.HDB5_STD5_SHA256` follows. It can
be fetched unattended, so unlike CICLOP it joins `make results`.

## 3. Population

Measured rows only. The documentation says the rows labelled W7-X and ITER are
predictive data, so every row whose `STELL` is W7-X or ITER is excluded, from
training as well as from scoring.

The primary analysis is the standard set, `STDSET = 1`, which is the counterpart
of the standard analysis set STD5 the manuscript uses. Every measured row,
whatever its `STDSET`, is a secondary analysis, as the rows outside STD5 were in
the manuscript's own replication.

Rows are dropped only for a missing, non-finite or non-positive value in the
target or in a used feature. No outlier rule, winsorising or reweighting is
applied.

## 4. Target

The confinement time the standard set is defined on: `TAUEDIA` for every device
except Heliotron E and TJ-II, and `TAUETH` for those two. That is the
documentation's own convention, stated in its sections IV.A and IV.E, and it is
followed and not improved on. It is modelled and scored in natural logs.
`TAUEDIA` for every device is a secondary analysis.

## 5. Features

The six regressors of ISS04, in natural logs, each from the column the
documentation names:

| Quantity | Column | Conversion |
| --- | --- | --- |
| effective minor radius | `AEFF` | m, as delivered |
| major radius | `RGEO` | m, as delivered |
| absorbed power | `PTOT` | W to MW |
| line-averaged density | `NEBAR` | m^-3 to 1e19 m^-3 |
| toroidal field | `BT` | T, absolute value |
| rotational transform | `IOTA23` | absolute value |

The unit conversions are checked during the schema pass against the ranges the
columns take, and a disagreement with this table is an amendment made before any
score. The manuscript's nine tokamak features do not apply: there is no plasma
current to scale with. The working gas mass is not a feature, because ISS04 has
no mass dependence and the feature set is the published law's.

A column is usable if it is finite and positive on at least 90% of the rows of
the primary population. An unusable column is dropped for every model and named
in the results. Usability is decided from missingness alone.

Nothing derived from the stored energy is ever a feature, since the target is
the stored energy divided by power.

There is no feature-matched rerun of HDB5 here. The tokamak feature set has no
counterpart of the rotational transform, so the two would not be the same
experiment. The comparison between the two databases is of the pattern, as it is
for the manuscript's two allometry datasets.

## 6. Devices and discharges

The held-out unit is the device as `STELL` names it, with no merging. W7-A and
W7-AS are different machines, and so are Heliotron E and Heliotron J.

Cross-validation is grouped on device and `SHOT`, so every row of one discharge
falls in one fold. A row with no shot number is its own group.

## 7. Eligibility

A device is scored if it has at least 30 rows after cleaning, which is the
manuscript's own threshold. This database is large enough to use it. Ten rows is
run as a sensitivity.

## 8. The floors of verdict 1

The dataset cannot be evaluated if the target is usable on under 90% of rows, or
fewer than five of the six features are usable, or fewer than 100 rows survive
cleaning, or fewer than five devices are eligible.

## 9. The published law

ISS04, evaluated analytically with its renormalisation factor set to one:
`tau = 0.134 a^2.28 R^0.64 P^-0.61 n^0.54 B^0.84 iota^0.41`. It is reported as a
historical reference outside the ranking, as IPB98(y,2) is in the manuscript. It
was fitted to this database's standard set, so it is not blind to any device
held out here. With the factor at one it is expected to be biased on most
devices, by design: the factor is ISS04's own statement that one constant does
not fit them all.

## 10. What is reported

The same three outputs as for CICLOP: each model's error under matched
cross-validation and under device holdout, with the ratio; the paired comparison
of the forest and the ridge across held-out devices; and the rank correlation
between each model's per-device error and that device's Mahalanobis distance
from its training rows.

The rest are secondary. Each is reported, and none can change the verdict.

- every measured row, in place of the standard set
- the 10-row threshold
- `TAUEDIA` as the target for every device
- both splits pooled over rows in place of averaged over devices
- the linear-plus-RBF Gaussian process of `gp.py`, settings unchanged
- each model's error with the device's mean log residual removed, under both
  splits. The audit expects a refitted power law to miss a held-out stellarator
  by a roughly constant factor. This is the number that says whether a holdout
  error is a missing constant, which is what ISS04's renormalisation factor
  stands for, or wrong trends

Conformal coverage and the size-ordered cut are not run. Anything not listed
here is exploratory and is labelled so wherever it appears.

## 11. Order of operations

1. This document is committed and pushed.
2. The pipeline is written and tested against a synthetic frame that has the
   documented columns.
3. The file is downloaded. Its SHA-256, byte count and download date are
   committed as a pin in `ishcdb.py` before any model is run on it.
4. A schema pass reads column names, per-column summaries, missingness and
   counts. It computes nothing that joins the target to a feature, fits nothing,
   and evaluates no scaling law.
5. The rules above are applied to what the schema pass found, and the outcome is
   written to `ishcdb_analysis_plan.json`: the pin, the columns, the usable
   features, the row counts, the eligible devices and every threshold here. That
   file is committed together with a second pin, its own SHA-256.
   `analysis_ishcdb.py` refuses to fit anything until both pins are set and both
   match.
6. The analysis is run once on the real file, and that run is the run of record.
   A rerun forced by a bug is entered in the log below with the bug, and both
   sets of output are kept.

## 12. Not allowed after the first score

Changing the threshold, the feature set, the population, the target convention,
the metric or the verdict rule. Tuning any model. Dropping a device or a row for
a reason not already in this document. Replacing this database with another, or
adding one, because of what its scores show.

## Deviations log

| Date | What changed | Why | Before or after the first score |
| --- | --- | --- | --- |
| 21 September 2026 | Section 4: TJ-II takes `TAUETH` as its target, as Heliotron E does. The lock as first committed named Heliotron E alone. | The lock's rule is to follow the documentation's convention, and it misread the documentation. Section IV.A, written for ISS95, says the thermal time is used "only for Heliotron-E". Section IV.E, on the TJ-II data added later, says "For TJ-II, the thermal confinement time has been used". The schema pass showed the consequence: the file has no `TAUEDIA` for any TJ-II row, so under the first wording 316 standard-set rows and one of seven devices would have dropped out. The change rests on the documentation and on missingness. | Before any model was fitted, and after the file was opened for the schema pass. |
