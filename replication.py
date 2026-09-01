"""Replicating the reversal on rows STD5 never contained. Result 11.

Every number in Results 1 to 10 rests on one file: the ITPA standard analysis
set STD5, 6228 quasi-stationary ELMy H-mode slices. That is the honest ceiling
on all of it. A reviewer can always say the reversal is a property of *this*
selection of *these* rows, and nothing in the repository so far can answer.

The answer is available, and it is upstream of the file already used. The same
OSF project publishes the full database revision, ``DB5.2.3.csv``: 14153 rows
and 192 columns against STD5's 6228 and 15. Matching the two on
``(tokamak, shot, time)`` shows STD5 is an ELMy-H-mode quality selection out of
it, and that leaves two populations this repository has never analysed.

    disjoint_h    5358 rows, 3199 discharges, 12 machines with 30+ rows,
                  R from 0.29 to 3.46 m. H-mode rows that STD5 does not contain,
                  either because they are plain ``H`` phase or because they
                  failed one of the standard set's other quality criteria.
                  **Zero row overlap with STD5**, same confinement regime, and
                  almost the same machine count as the 13 STD5 can score.

    non_h         3860 rows, 5 machines. Ohmic, L-mode and radiative-improved
                  rows. A genuinely different confinement regime, on a machine
                  set that includes TEXTOR, a limiter tokamak contributing 1435
                  rows and absent from the H-mode analysis entirely.

The two arms answer different objections and neither answers both. ``disjoint_h``
has the machine count to support a leave-one-machine-out claim but shares the
regime and the devices. ``non_h`` changes the regime but has five machines, which
is too few for anything but a directional check, and is said so wherever it is
reported.

Why the non-H arm needs a different baseline
--------------------------------------------
IPB98(y,2) is an ELMy H-mode scaling. Scoring L-mode and ohmic rows against it
would be measuring the H-mode/L-mode confinement difference, not a model's
skill, and every model would beat it for reasons that have nothing to do with
this repository's argument. So the non-H arm is scored against **ITER89-P**
(Yushmanov et al., Nucl. Fusion 30 1999, 1990), the published L-mode power law,
which stands in exactly the same relation to that arm as IPB98(y,2) does to the
H-mode one.

What is being replicated
------------------------
Not the numbers. The row populations differ, so the RMSLE values will differ and
should. What is being replicated is the *structure* of Result 4: that grouped
cross-validation by discharge ranks a flexible model far above a log-linear
power law, that holding out an entire machine reverses that ranking, and that
the reversal is not an artifact of the STD5 selection criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
import hdb5

# --- The full database revision, pinned the same way STD5 is -----------------
#
# Same discipline and same reasoning as ``hdb5.HDB5_STD5_SHA256``: this file is
# third-party data fetched at run time from a host this repository does not
# control, so "reproducible" without a content pin would mean "runs again". The
# digest below was taken from two independent fresh downloads that agreed.
DB523_OSF_PROJECT = "https://osf.io/drwcq"
DB523_DOWNLOAD_URL = "https://osf.io/download/zhwa3/"
DEFAULT_DB523_FILENAME = "hdb5_db523.csv"
DB523_SHA256 = "7a48e34379663e3e298924990f05cd8f16b8581516bcfa7bb8f438f83ae80ab6"
DB523_N_BYTES = 11685289
# Shape of the raw CSV as downloaded. The first line is a numeric index header
# and the real column names are on the second, which is why the parsed frame has
# one more row than the 14153 records the database documents.
DB523_RAW_SHAPE = (14154, 192)

# The phase codes. ``PHASE`` distinguishes the confinement regime of each slice;
# everything beginning with ``H`` is some flavour of H-mode (``HGELM`` is type-I
# ELMy, ``HSELM`` type-III, and so on), and the three below are not.
NON_H_PHASES: tuple[str, ...] = ("OHM", "L", "RI")

# Identity of a slice, for matching against STD5. Time is rounded because the
# two exports carry it at different float precisions and an exact float compare
# would silently declare every row disjoint, which is the failure mode that
# would make this whole module report a fake replication.
_MATCH_TIME_DECIMALS = 6

REPLICATION_ARMS: tuple[str, ...] = ("disjoint_h", "non_h")

# DB5.2.3 stores SI units where the STD5 export stores the units the published
# scaling laws are written in. The three factors below were not assumed: they
# were measured on the 4595 rows the two files share, where the ratios come out
# at exactly 1e6, exactly 1e19 and 1e6 to six figures, while ``BT``, ``RGEO``,
# ``KAPPAA`` and ``MEFF`` come out at exactly 1. Getting this wrong is not a
# subtle error (IPB98(y,2) evaluates to 3e8 seconds on unconverted rows) but it
# is the kind that a pipeline happily carries all the way to a plot.
DB523_UNIT_SCALES: dict[str, float] = {
    "IP": 1e6,       # amperes -> megaamperes
    "NEL": 1e19,     # m^-3 -> 1e19 m^-3
    "PLTH": 1e6,     # watts -> megawatts
}


def default_db523_path() -> Path:
    return config.get_data_raw_dir() / DEFAULT_DB523_FILENAME


def verify_db523_file(path: Path | str) -> hdb5.DatasetFingerprint:
    """Fingerprint a file and raise unless it is the pinned DB5.2.3 revision."""
    fingerprint = hdb5.fingerprint_file(path)
    if fingerprint.sha256 == DB523_SHA256 and fingerprint.n_bytes == DB523_N_BYTES:
        return fingerprint
    raise hdb5.DatasetIntegrityError(
        f"DB5.2.3 integrity check failed for {fingerprint.path}.\n"
        f"  expected sha256 {DB523_SHA256} ({DB523_N_BYTES} bytes)\n"
        f"  observed sha256 {fingerprint.sha256} ({fingerprint.n_bytes} bytes)\n"
        f"Re-fetch from {DB523_DOWNLOAD_URL}. Every replication number under "
        "results/ is a statement about the pinned bytes, so a mismatch means "
        "those numbers must be regenerated before they can be compared."
    )


def download_db523(
    destination: Path | None = None, *, overwrite: bool = False, verify: bool = True
) -> Path:
    """Fetch the full DB5.2.3 revision from OSF into the raw data directory.

    Third-party scientific data (please cite Verdoolaege et al., Nucl. Fusion 61
    076006, 2021); fetched on demand rather than redistributed here, exactly as
    ``hdb5.download_hdb5_std5`` does, and verified on the staged temporary file
    so a download that does not match the pin never lands at the target path.
    """
    import urllib.request

    from storage import atomic_output_path

    target = Path(destination).expanduser().resolve() if destination else default_db523_path()
    if target.exists() and not overwrite:
        if verify:
            verify_db523_file(target)
        return target
    with atomic_output_path(target) as temp_path:
        with urllib.request.urlopen(DB523_DOWNLOAD_URL) as response:
            temp_path.write_bytes(response.read())
        if verify:
            verify_db523_file(temp_path)
    return target


def load_db523_raw(path: Path | str | None = None, *, verify: bool | None = None) -> pd.DataFrame:
    """Load DB5.2.3, skipping the numeric index header on the first line."""
    resolved = Path(path).expanduser().resolve() if path is not None else default_db523_path()
    if not resolved.exists():
        raise FileNotFoundError(
            f"DB5.2.3 not found: {resolved}. Fetch it with "
            f"`python3 -c 'import replication; replication.download_db523()'` or "
            f"download {DB523_DOWNLOAD_URL} to that path."
        )
    should_verify = resolved == default_db523_path() if verify is None else verify
    if should_verify:
        verify_db523_file(resolved)
    return pd.read_csv(resolved, low_memory=False, skiprows=[0])


def iter89p_tau_s(frame: pd.DataFrame) -> pd.Series:
    """Analytic ITER89-P L-mode confinement scaling (seconds).

    tau = 0.048 * Ip^0.85 * R^1.2 * a^0.3 * kappa^0.5
               * n20^0.1 * Bt^0.2 * M^0.5 * P^-0.5

    with Ip in MA, R and a in m, n20 the line-averaged density in 1e20 m^-3, Bt
    in T, M the isotope mass in amu and P the loss power in MW. The canonical
    frame carries density in 1e19 m^-3, hence the factor of ten.

    Reference: Yushmanov et al., Nucl. Fusion 30 1999 (1990).
    """
    return (
        0.048
        * np.power(frame["ip_ma"], 0.85)
        * np.power(frame["r_m"], 1.2)
        * np.power(frame["a_m"], 0.3)
        * np.power(frame["kappa"], 0.5)
        * np.power(frame["ne_line_1e19_m3"] / 10.0, 0.1)
        * np.power(frame["bt_t"], 0.2)
        * np.power(frame["m_eff_amu"], 0.5)
        * np.power(frame["p_loss_mw"], -0.5)
    )


def prepare_db523_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw DB5.2.3 slice through exactly STD5's cleaning path.

    Two things have to happen before STD5's cleaner can be used. The columns in
    :data:`DB523_UNIT_SCALES` are converted to the units the published laws are
    written in, and the inverse aspect ratio is derived as ``AMIN / RGEO``
    because DB5.2.3 stores minor radius and has no ``EPS`` column. That
    derivation reproduces STD5's own ``EPS`` to 3.6e-10 on the rows the two
    files share, which is the check that it is the right one.

    Everything after that, including the absolute values taken on the signed
    current and field, is STD5's code rather than a parallel implementation: a
    replication that cleaned its data differently would not be replicating
    anything.
    """
    frame = raw.copy()
    for column, scale in DB523_UNIT_SCALES.items():
        frame[column] = pd.to_numeric(frame[column], errors="coerce") / scale
    frame["EPS"] = pd.to_numeric(frame["AMIN"], errors="coerce") / pd.to_numeric(
        frame["RGEO"], errors="coerce"
    )
    cleaned = hdb5.map_to_canonical(frame)
    featured = hdb5.build_features(cleaned)
    featured["iter89p_tau_s"] = iter89p_tau_s(featured)
    featured["log_iter89p_tau_s"] = np.log(featured["iter89p_tau_s"])
    return featured


def _match_keys(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["TOK"].astype(str)
        + "::"
        + frame["SHOT"].astype(str)
        + "::"
        + pd.to_numeric(frame["TIME"], errors="coerce").round(_MATCH_TIME_DECIMALS).astype(str)
    )


@dataclass(frozen=True)
class ReplicationArm:
    """One replication population, with the provenance needed to read its numbers."""

    name: str
    dataset: pd.DataFrame
    baseline_column: str
    baseline_label: str
    n_rows: int
    n_discharges: int
    n_machines_scored: int
    n_rows_shared_with_std5: int
    r_min_m: float
    r_max_m: float

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "baseline": self.baseline_label,
            "n_rows": self.n_rows,
            "n_discharges": self.n_discharges,
            "n_machines_scored": self.n_machines_scored,
            "n_rows_shared_with_std5": self.n_rows_shared_with_std5,
            "r_min_m": self.r_min_m,
            "r_max_m": self.r_max_m,
        }


def build_replication_arms(
    *,
    db523_path: Path | str | None = None,
    std5_path: Path | str | None = None,
    min_rows: int = hdb5.MIN_HELD_OUT_ROWS,
) -> dict[str, ReplicationArm]:
    """The two replication populations, with STD5's rows removed from the H arm.

    ``n_rows_shared_with_std5`` is carried on every arm and asserted to be zero
    for ``disjoint_h`` in ``tests/test_replication.py``. It is the one property
    the whole result depends on, and it depends in turn on the row match
    working: a match that silently failed would leave STD5's own rows in the
    "disjoint" arm and reproduce Result 4 by construction.
    """
    raw = load_db523_raw(db523_path)
    std5 = hdb5.load_hdb5_dataframe(std5_path)
    std5_keys = set(_match_keys(std5))

    keys = _match_keys(raw)
    phase = raw["PHASE"].astype(str)
    selections = {
        "disjoint_h": phase.str.startswith("H") & ~keys.isin(std5_keys),
        "non_h": phase.isin(NON_H_PHASES),
    }
    baselines = {
        "disjoint_h": ("ipb98y2_tau_s", "IPB98(y,2)"),
        "non_h": ("iter89p_tau_s", "ITER89-P"),
    }

    arms: dict[str, ReplicationArm] = {}
    for name, mask in selections.items():
        subset = raw.loc[mask]
        dataset = prepare_db523_frame(subset)
        shared = int(_match_keys(subset).isin(std5_keys).sum())
        counts = dataset[hdb5.TOKAMAK_LABEL_COLUMN].value_counts()
        baseline_column, baseline_label = baselines[name]
        arms[name] = ReplicationArm(
            name=name,
            dataset=dataset,
            baseline_column=baseline_column,
            baseline_label=baseline_label,
            n_rows=int(len(dataset)),
            n_discharges=int(dataset[hdb5.GROUP_COLUMN].nunique()),
            n_machines_scored=int((counts >= min_rows).sum()),
            n_rows_shared_with_std5=shared,
            r_min_m=float(dataset["r_m"].min()),
            r_max_m=float(dataset["r_m"].max()),
        )
    return arms
