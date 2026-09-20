# CICLOP dataset audit

Audit date: 19 September 2026. Status: literature-level only. The data file has
not been obtained, for the reason given under Access, so nothing below was read
from the file itself.

No model has been fitted to CICLOP data. No CICLOP score, residual or
feature-target relationship has been computed or looked at, here or anywhere
else in this repository. The rules the replication will follow are in
[ciclop-replication-lock.md](ciclop-replication-lock.md), and this audit is
what those rules were written from.

## What CICLOP is

CICLOP is the Coordination on International Challenges on Long duration
OPeration, a network of experts set up by the IAEA. Its database was developed
under the IEA Technology Collaboration Programmes on tokamaks and on
stellarators and heliotrons, with a group coordinated by CEA. It is a 0-D
multi-machine database of long-duration plasmas from tokamaks and stellarators.

Two open-access papers describe it, both CC BY 4.0:

- X. Litaudon et al., Nucl. Fusion 64, 015001 (2024),
  [doi:10.1088/1741-4326/ad0606](https://doi.org/10.1088/1741-4326/ad0606).
  109 pulses, about 29 variables per pulse, data to January 2022.
- X. Litaudon et al., Nucl. Fusion 66, 095001 (2026),
  [doi:10.1088/1741-4326/ae89cc](https://doi.org/10.1088/1741-4326/ae89cc).
  Version 7.3: 239 pulses, up to 54 variables per pulse, experiments to May 2025.

How these were read matters for how far to trust the details. Both papers were
read through a fetch tool that summarises a page, and the OSTI full-text copies
refused the connection, so no PDF was opened directly. The per-device counts
below sum to the stated total of 239, which is a consistency check and not a
verification. Check the counts and the variable list against Table 1 of the 2026
paper and Annex 2 of the 2024 paper before any of them is quoted in the
manuscript. No rule in the lock depends on them, because every rule is applied
to the counts in the file itself.

## Access

The 2024 paper prints
`https://nucleus.iaea.org/sites/fusionportal/ciclop/SitePages/CICLOP-DB.aspx`.
That address now redirects to the Fusion Portal home page. The page has moved to
`https://nucleus.iaea.org/sites/fusion-portal/ciclop/SitePages/CICLOP-DB.aspx`,
last modified 11 December 2025, and the 2026 paper cites the site's home page.

An anonymous visitor gets the page, which holds a description of the database
and one image, with no file link. The data sits in a SharePoint library at
`/sites/fusion-portal/ciclop/CICLOPDB`. That library answers HTTP 403, "Access
denied", to an unauthenticated request, and it is missing from the six lists the
site exposes to anonymous visitors. The restriction is deliberate, and nothing
was done to get around it.

Getting the file needs an IAEA NUCLEUS account, and possibly membership of the
CICLOP site on top of that. Which of the two is required was not established.
The author has to do this, since it means registering a person with the IAEA.

Neither paper attaches the database as supplementary material. The 2024 paper
points to the IAEA page and notes that the LHD subset is open at
[doi:10.57451/lhd.analyzed-data](https://doi.org/10.57451/lhd.analyzed-data).
LHD is a stellarator, so that subset does not help here.

No licence or terms of use for the data were found. Read them once access is
granted, before anything derived from the file is committed. Until then, assume
the HDB5 discipline applies: the file is fetched on demand, gitignored, pinned by
SHA-256, and never redistributed. One cost is already visible. If the file sits
behind a login, `make reproduce` cannot fetch it unattended, and the paper's
data-availability statement has to say so.

## Size and structure

One row is one pulse. Per facility, from Table 1 of the 2026 paper:

| Facility | Type | Pulses | In HDB5 STD5 |
| --- | --- | ---: | --- |
| JET (carbon wall and ITER-like wall together) | tokamak | 69 | yes |
| DIII-D | tokamak | 49 | yes |
| WEST | tokamak | 27 | no |
| W7-X | stellarator | 21 | no |
| EAST | tokamak | 19 | no |
| ASDEX Upgrade | tokamak | 11 | yes |
| JT-60U | tokamak | 11 | yes |
| KSTAR | tokamak | 11 | no |
| Tore Supra | tokamak | 8 | no |
| LHD | stellarator | 5 | no |
| TCV | tokamak | 4 | yes |
| TFTR | tokamak | 4 | yes |
| Total | | 239 | |

The ten tokamaks hold 213 pulses, which is 3.4% of the 6228 rows the manuscript
analyses. The number of tokamaks that can be held out depends on the row
threshold, and the manuscript's own sweep of that threshold used 10, 20, 30 and
50:

| Minimum rows to score a device | Eligible tokamaks | Which |
| ---: | ---: | --- |
| 10 | 7 | ASDEX Upgrade, DIII-D, EAST, JET, JT-60U, KSTAR, WEST |
| 20 | 3 | DIII-D, JET, WEST |
| 30 (the manuscript's value) | 2 | DIII-D, JET |
| 50 | 1 | JET |

These counts are before cleaning. ASDEX Upgrade, JT-60U and KSTAR have 11 pulses
each, so each of them drops out of the 10-row set if it loses two rows to a
missing value.

## Schema against the manuscript's features

The CICLOP column is what the papers document, and none of it has been confirmed
against the file.

| Quantity | HDB5 column | CICLOP as documented | Standing |
| --- | --- | --- | --- |
| Target | `TAUTH`, thermal confinement time | "volume-averaged plasma energy confinement time" tau_E [s], supplied by each facility | Thermal or total is not stated, and it may differ by facility |
| Plasma current | `IP` | plasma current [MA] | present |
| Toroidal field | `BT` | toroidal field on axis [T] | present |
| Density | `NEL`, line averaged | The 2024 list has only the core ion density n_i0 [1e20 m^-3]. The 2026 paper adds a core electron density and uses a Greenwald fraction, which needs a line-averaged density | unverified |
| Power | `PLTH`, loss power | injected additional power [MW]; core radiated power "when available" from v7.3 | No loss power. Injected power excludes the ohmic part, the radiated part and dW/dt |
| Major radius | `RGEO` | major radius [m] | present |
| Minor radius | derived as eps times R | minor radius a [m] | present |
| Inverse aspect ratio | `EPS` | derive as a/R | derivable |
| Elongation | `KAPPAA`, areal | "elongation", definition not stated | unverified |
| Effective mass | `MEFF`, numeric | main fuel species as a label: hydrogen, deuterium, tritium, helium | Needs a fixed mapping. D-T mixtures are ambiguous |
| Discharge | `SHOT` | pulse number | present |
| Device | `TOK` | facility name | JET's two walls are separated by a first-wall material column, not by facility |
| Regime | all ELMy H-mode | H-mode or L-mode label | mixed population |

Version 7.3 also adds, "when available", the stored energy, a confinement factor
relative to the H-mode scaling law, core electron temperature, and pedestal
values. The confinement factor is the target divided by a published law, so it
must never be used as a feature.

## What differs from the manuscript's setting

All of the following was known before any score exists. It is written down now
so that none of it can later be offered as an explanation only when the result
needs one.

1. Two devices clear the manuscript's 30-row threshold, and both are in HDB5.
   The protocol moved across unchanged cannot test a ranking claim. The lock
   says what happens instead.
2. With one row per pulse, cross-validation "grouped by discharge" is five-fold
   cross-validation over rows. It still measures prediction inside devices the
   model has seen, but the grouping does no work.
3. Each facility's contact person chose the pulses as "the most representative
   fusion scenarios" by duration or fusion performance. That is selection on
   quantities close to the target, and it is a set of records, not a sample of
   operating space. Repeats of one scenario inside a device give near-duplicate
   rows, which favours an interpolating model under cross-validation. The
   expected direction of this bias is toward a larger cross-validated advantage
   for the tree ensembles.
4. The target is not defined uniformly. A difference in definition between
   facilities adds to every model's error on a held-out device, and no feature
   can explain it.
5. Injected power stands in for loss power. The gap is small for a stationary,
   weakly radiating pulse and large for a strongly radiating one.
6. L-mode and H-mode pulses are mixed, and regime is not one of the nine
   features. IPB98(y,2) is an ELMy H-mode law and is the wrong reference for the
   L-mode rows.
7. CICLOP was assembled separately from HDB5, but six of its ten tokamaks are
   HDB5 devices, and some pulses may be HDB5 discharges. EAST, KSTAR, Tore Supra
   and WEST are absent from HDB5 and hold 65 pulses between them. The replication
   trains and tests inside CICLOP only, so shared pulses leak nothing between
   folds, but they limit how independent the two datasets can be called.
8. The models keep the manuscript's settings, which were fixed on 6228 rows.
   `HistGradientBoostingRegressor` defaults to `min_samples_leaf=20`, and with
   fewer than 200 training rows it can make few splits. Its behaviour at this
   size is part of the result and will not be tuned away.
9. The stellarators have no plasma current in the tokamak sense, and the
   manuscript is about tokamak scaling.
10. WEST reuses the vacuum vessel and toroidal field coils of Tore Supra with a
    different magnetic configuration: a diverted, D-shaped plasma in place of a
    circular limiter plasma. Whether the two are one physical device decides
    whether Tore Supra's 8 pulses stay in training when WEST is held out, and
    whether the pair, at 35 pulses, clears the 30-row threshold.

## What has to be confirmed from the file

Each item is a count, a column name or a missingness rate. None of them needs a
model, and the lock says what follows from each answer.

- the version string, the file format, and the SHA-256 of the bytes
- the actual column names and units, against the table above
- whether a line-averaged density exists, and under what name
- which elongation the file carries
- whether the target is thermal or total, per facility, if the file or its
  documentation says
- whether a loss or absorbed power is present
- how fuel species is coded, including D-T pulses
- whether a pulse number can repeat within a facility
- missingness per column and per facility, and the per-facility counts after
  complete-case cleaning
- the regime label's values and their counts per facility
- how many (facility, pulse) pairs also appear in DB5.2.3
- the licence or terms of use
