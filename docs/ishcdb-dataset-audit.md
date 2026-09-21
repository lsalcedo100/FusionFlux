# ISHCDB dataset audit

Audit date: 21 September 2026. Status: from the documentation only. The data file
has not been downloaded. What was read is the public directory listing and
`documentationDB07_26.pdf`, which is the database's own description of its
columns.

No model has been fitted to this database, here or anywhere else in this
repository, and no score, residual or feature-target relationship has been
looked at. The rules the replication will follow are in
[ishcdb-replication-lock.md](ishcdb-replication-lock.md), and this audit is what
they were written from.

## Why a second external dataset

The first one, CICLOP, sits behind an IAEA login. Access was requested on 20
September 2026 and had not been granted a day later. It may come in a week, or
not at all, and [its lock](ciclop-replication-lock.md) already says what happens
in each case. That lock also says that any other external dataset needs its own
lock before its scores are seen, and that the CICLOP outcome is reported whatever
a later dataset shows. This is that other dataset. It does not replace CICLOP.
No CICLOP score exists, so nothing about this choice could have been made in
response to one.

## What ISHCDB is

The International Stellarator-Heliotron Confinement Database is the stellarator
counterpart of the ITPA confinement database the manuscript rests on. It is kept
under the IEA Implementing Agreement on the stellarator concept, hosted jointly
by IPP Greifswald and NIFS, and it is the database the ISS95 and ISS04 scalings
were fitted to, as IPB98(y,2) was fitted to the ITPA one.

Version 26 is documented by A. Kus and A. Dinklage, dated 19 June 2012. It holds
4940 observations. A version 27, documented on 1 April 2022, adds Wendelstein
7-X, but no public copy of its data was found, so version 26 is the one used.

The references are H. Yamada et al., Nucl. Fusion 45, 1684 (2005), for ISS04 and
the database it was fitted to, and U. Stroth et al., Nucl. Fusion 36, 1063
(1996), for ISS95. Both have to be checked against the published record before
they are cited in the manuscript.

## Access

Open, with no login. The files are in a plain directory on the IPP server,
`https://ishpdb.ipp-hgw.mpg.de/ISS/`, linked from the public page of the
International Stellarator-Heliotron Profile Database. The listing shows the same
version in several formats: `ISHCDB_26.txt` at 3.2 MB, `ishcdb_26.xlsx`,
`ISHCDB_26.sas7bdat` and `ISHCDB_26.JMP`. Every file there is dated 21 March
2016.

So this file can be fetched unattended, pinned by SHA-256 and verified on load,
under the rules `hdb5.HDB5_STD5_SHA256` follows. Unlike CICLOP it can join `make
results` and the reproduce workflow.

No licence or terms of use appear in the documentation or on the page. The
documentation names the collaboration and the teams that supplied the data, and
an acknowledgement of them is owed. The file is not redistributed here.

## Structure, as documented

The column conventions are those of the ITPA database, which is where this
database's design came from.

| Role | Column | As documented |
| --- | --- | --- |
| Device | `STELL` | ATF, CHS, HELE, HELJ, LHD, TJ-II, W7-A, W7-AS, HSX. Rows labelled W7-X and ITER are "predictive data", which is to say they are not measurements |
| Standard set | `STDSET` | 1 for observations in the standard set, 0 otherwise |
| Discharge | `SHOT` | shot number, or the first of a sequence. `SHOT TIME` is the time within it, so a discharge can supply several rows |
| Target | `TAUEDIA` | `WDIA / (PTOT - DWDIA)`, seconds |
| Target, Heliotron E | `TAUETH` | `WTH / (PTOT - DWTH)`, seconds |
| Minor radius | `AEFF` | effective minor radius, m |
| Major radius | `RGEO` | major radius of the last closed flux surface, m |
| Power | `PTOT` | total absorbed power, W: `PABSECH + PABSNBI + PABSICH + POH` |
| Density | `NEBAR` | line-averaged electron density, m^-3 |
| Field | `BT` | vacuum toroidal field at `RGEO`, T |
| Rotational transform | `IOTA23` | at two thirds of the minor radius |
| Working gas | `PGASA` | mass number: 1, 2, 3 or 4 |

The standard set is defined in section IV of the documentation. It deletes
helium discharges and a list of named discharges, and for W7-AS it deletes
discharges above a threshold in power per unit density. `STDSET` records the
outcome, so no rule here has to restate those criteria. It uses the diamagnetic
confinement time, except for Heliotron E, where it uses the thermal one. The
documentation's count for the standard set, 1750, is given beside a count of
1476 for the rest. Those sum to 3226 and the file holds 4940, so they date from
an earlier version. The counts that matter are the ones in the file.

ISS04 is
`tau = 0.134 a^2.28 R^0.64 P^-0.61 n^0.54 B^0.84 iota^0.41`, with `P` in MW and
`n` in 1e19 m^-3.

## What differs from the manuscript's setting

All of this is known before any score exists, and is written down so that none
of it can be offered later as an explanation only when the result needs one.

1. These are stellarators and heliotrons, and the manuscript is about tokamaks.
   The regressors are ISS04's six and not the manuscript's nine. There is no
   plasma current to scale with, and the rotational transform takes its place.
   A result here says whether the validation-protocol effect appears in the
   sister database of the same field. It does not say anything about tokamaks
   that HDB5 has not already said.
2. No device is shared with HDB5. That is the property CICLOP lacks, where six of
   ten tokamaks are HDB5 devices.
3. A single power law is already known to transfer poorly between these devices.
   ISS04 could not unify them with one constant. It carries a renormalisation
   factor for each device and configuration, running from about 0.25 to 1. So a
   power law refitted on the other devices should be expected to miss a held-out
   device by a roughly constant factor, whatever its exponents do. That cuts
   against the log-linear ridge, which is the model the manuscript finds
   transfers best. The direction of the outcome cannot be called in advance, and
   the lock adds a secondary analysis that separates a missing constant from
   wrong trends.
4. The devices differ in kind more than tokamaks do: torsatrons, heliotrons, a
   heliac and shearless advanced stellarators. Holding one out is a larger step
   than holding out a tokamak.
5. The target is a diamagnetic confinement time, where the manuscript's is
   thermal, and for one device it is thermal. Both follow the standard set's own
   convention.
6. LHD supplies a large share of the rows, as JET and ASDEX Upgrade do in HDB5.
   Scoring each device equally is what keeps one machine from carrying the
   result.
7. The file is from 2012 and lacks Wendelstein 7-X.

## What has to be confirmed from the file

Each of these is a column name, a count or a missingness rate. None needs a
model, and the lock says what follows from each.

- the SHA-256 and byte count of `ISHCDB_26.txt`, and its delimiter and header
- that the column names are the documented ones
- the values `STELL` and `STDSET` take, and the rows per device in the standard
  set and outside it
- the units of `PTOT` and `NEBAR`
- whether `IOTA23` and `BT` carry a sign
- missingness per column and per device, and the rows per device after cleaning
- how many rows share a discharge
- whether `TAUETH` is present for Heliotron E
