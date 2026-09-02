"""Result 15: the reversal's precondition, measured by varying it.

Result 13 ran this repository's audit on Kleiber's law and found two halves. The
extrapolation failure reproduced completely: tree ensembles lost to both power
laws at all 8 mass cuts and on 9 of 11 held-out orders. The *ranking reversal*
did not, and Result 13b says why, as a conjecture rather than a measurement:

    With a single predictor and a relationship that is close to a straight line
    in logs, a tree has far less to exploit, and the 41% cross-validated margin
    this README opens with is simply not available here to be reversed. **The
    reversal needs enough feature dimensionality for the flexible model to win
    interpolation first.** Nothing in Results 4 to 12 could have shown that,
    because one database cannot.

Two databases cannot show it either, if they differ in a dozen ways at once.
HDB5 has nine features and a reversal; the mammalian data has one feature and no
reversal; between them sit different sciences, different group structures,
different sample sizes and different amounts of noise. Feature count is one
candidate explanation among many.

This module isolates it. The Biomass And Allometry Database has, for the same
plants, a ladder of predictors of increasing dimension, so the experiment is to
fix the rows, fix the groups, fix the splits, fix the models, and vary only how
many features the models are allowed to see. Whatever moves is caused by
dimensionality, because nothing else moved.

    total plant mass                    the target, over 7 orders of magnitude
    basal diameter                      rung 1: the Kleiber analogue, one
                                        predictor, a pure power law
    + height                            rung 2
    + leaf area                         rung 3
    + leaf mass                         rung 4

    species, the unit a plant is a      ->  the analogue of tokamak
    member of
    diameter, the size axis             ->  the analogue of machine size
    West-Brown-Enquist, mass scales     ->  the analogue of IPB98(y,2) and of
    as diameter^(8/3)                       Kleiber's 3/4: a published exponent
                                            derived from theory rather than fitted

The split structure has to match Result 4's exactly or the comparison is
meaningless, and this is the trap the analysis fell into once. HDB5's "CV, by
discharge" holds out *shots*, so every machine in the held-out fold is also in
the training fold: it is interpolation within known machines. Grouping the
cross-validation by species here would instead be the analogue of
leave-one-tokamak-out, and comparing that against leave-one-species-out compares
a hard split with the same hard split. So the interpolation arm keeps the same
species on both sides, and the extrapolation arm holds an entire species out.
``tests/test_tree_allometry.py`` pins that distinction.

Like Result 13, this runs through ``scaling_audit``, the domain-agnostic module,
rather than a copy of ``hdb5``.

Data
----
The Biomass And Allometry Database (Falster et al. 2015, *Ecology* 96:1445),
release v1.0.1, 21084 plants from 176 studies, released under CC0. Third-party
data, fetched on demand and pinned by SHA-256 exactly as the HDB5 files are, not
redistributed here.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
import hdb5
from scaling_law import _clean_fp_state as clean_fp_state

# Pinned the same way, and for the same reason, as ``hdb5.HDB5_STD5_SHA256``.
BAAD_DOWNLOAD_URL = "https://github.com/dfalster/baad/releases/download/v1.0.1/baad_data.zip"
BAAD_SHA256 = "0375f012475658c039e3dcef398951e8f68cdeb5430eec8cd65548965496cfe2"
BAAD_N_BYTES = 1117744
DEFAULT_BAAD_FILENAME = "baad_data.zip"
BAAD_MEMBER = "baad_data/baad_data.csv"

# West, Brown and Enquist (1999) derive mass proportional to diameter^(8/3) from
# a branching-network model. It is the analogue of IPB98(y,2) and of Kleiber's
# 3/4: a published exponent that comes from theory rather than from this data,
# and is therefore something to test rather than assume.
WBE_EXPONENT = 8.0 / 3.0

GROUP_COLUMN = "species"
TARGET_COLUMN = "total_mass_kg"
SIZE_COLUMN = "diameter_m"

# The source columns, in the order the ladder adds them. Diameter first because
# it is the single predictor the classical law uses, which makes rung 1 the
# direct analogue of Result 13's one-predictor problem.
SOURCE_COLUMNS: dict[str, str] = {
    "m.to": TARGET_COLUMN,
    "d.ba": SIZE_COLUMN,
    "h.t": "height_m",
    "a.lf": "leaf_area_m2",
    "ma.ilf": "leaf_mass_kg",
}

LOG_FEATURE_ORDER: tuple[str, ...] = (
    "log_diameter_m",
    "log_height_m",
    "log_leaf_area_m2",
    "log_leaf_mass_kg",
)

# The ladder. Each rung is a prefix of the one above, so adding a feature is the
# only difference between consecutive rungs.
FEATURE_LADDER: dict[int, tuple[str, ...]] = {
    n: LOG_FEATURE_ORDER[:n] for n in range(1, len(LOG_FEATURE_ORDER) + 1)
}

# Below this a species' held-out score is dominated by which few individuals
# happen to be in it. Mirrors ``hdb5.MIN_HELD_OUT_ROWS``.
MIN_HELD_OUT_ROWS = 30


def default_baad_path() -> Path:
    return config.get_data_raw_dir() / DEFAULT_BAAD_FILENAME


def verify_baad_file(path: Path | str) -> hdb5.DatasetFingerprint:
    """Fingerprint the archive and raise unless it is the pinned release."""
    fingerprint = hdb5.fingerprint_file(path, read_shape=False)
    if fingerprint.sha256 == BAAD_SHA256 and fingerprint.n_bytes == BAAD_N_BYTES:
        return fingerprint
    raise hdb5.DatasetIntegrityError(
        f"BAAD dataset integrity check failed for {fingerprint.path}.\n"
        f"  expected sha256 {BAAD_SHA256} ({BAAD_N_BYTES} bytes)\n"
        f"  observed sha256 {fingerprint.sha256} ({fingerprint.n_bytes} bytes)\n"
        f"Re-fetch from {BAAD_DOWNLOAD_URL}. Every Result 15 number under "
        "results/ is a statement about the pinned bytes, so a mismatch means "
        "those numbers must be regenerated before they can be compared."
    )


def download_baad(
    destination: Path | None = None, *, overwrite: bool = False, verify: bool = True
) -> Path:
    """Fetch the tagged GitHub release into the raw data directory.

    Verified on the staged temporary file, so a download that does not match the
    pin never lands at the target path.
    """
    import urllib.request

    from storage import atomic_output_path

    target = Path(destination).expanduser().resolve() if destination else default_baad_path()
    if target.exists() and not overwrite:
        if verify:
            verify_baad_file(target)
        return target
    with atomic_output_path(target) as temp_path:
        with urllib.request.urlopen(BAAD_DOWNLOAD_URL) as response:
            temp_path.write_bytes(response.read())
        if verify:
            verify_baad_file(temp_path)
    return target


def load_baad_raw(path: Path | str | None = None, *, verify: bool | None = None) -> pd.DataFrame:
    """Read the plant table out of the archive without unpacking it to disk."""
    target = Path(path).expanduser().resolve() if path else default_baad_path()
    if verify is None:
        verify = True
    if verify:
        verify_baad_file(target)
    with zipfile.ZipFile(target) as archive:
        with archive.open(BAAD_MEMBER) as member:
            return pd.read_csv(io.BytesIO(member.read()), low_memory=False)


def prepare_dataset(path: Path | str | None = None, *, verify: bool | None = None) -> pd.DataFrame:
    """Complete, strictly positive rows on every ladder column, with log features.

    One row set for the whole ladder, not one per rung. That is the point of the
    design: if rung 1 were fitted on every plant with a diameter and rung 4 only
    on those that also have a leaf mass, the rungs would differ in their rows as
    well as their features and nothing could be attributed to dimensionality.
    So the intersection is taken once, here, and every rung sees exactly these
    plants.
    """
    raw = load_baad_raw(path, verify=verify)
    missing = [column for column in SOURCE_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"BAAD table is missing expected columns: {missing}")

    frame = raw[[*SOURCE_COLUMNS, GROUP_COLUMN]].rename(columns=SOURCE_COLUMNS)
    frame = frame.dropna()
    numeric = list(SOURCE_COLUMNS.values())
    frame = frame[(frame[numeric] > 0).all(axis=1)].reset_index(drop=True)

    for column in numeric:
        frame[f"log_{column}"] = np.log(frame[column].to_numpy(dtype=float))
    return frame


def eligible_species(dataset: pd.DataFrame, *, min_rows: int = MIN_HELD_OUT_ROWS) -> list[str]:
    """Species with enough plants to hold out, largest first by median diameter."""
    counts = dataset[GROUP_COLUMN].value_counts()
    keep = [str(name) for name, count in counts.items() if count >= min_rows]
    medians = dataset[dataset[GROUP_COLUMN].isin(keep)].groupby(GROUP_COLUMN)[SIZE_COLUMN].median()
    return [str(name) for name in medians.sort_values(ascending=False).index]


def species_size_medians(dataset: pd.DataFrame) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in dataset.groupby(GROUP_COLUMN)[SIZE_COLUMN].median().items()
    }


@dataclass(frozen=True)
class WBEBaseline:
    """The published exponent, and what refitting it freely gives instead.

    ``constrained_rmsle`` holds the exponent at 8/3 and fits only the intercept,
    which is the analogue of scoring IPB98(y,2) as published. ``free_exponent``
    lets the data choose, which is how Result 13 reports Kleiber's 0.75 against
    a refitted 0.687.
    """

    free_exponent: float
    free_intercept: float
    free_rmsle: float
    constrained_intercept: float
    constrained_rmsle: float


def fit_wbe(log_diameter: np.ndarray, log_mass: np.ndarray) -> WBEBaseline:
    """Fit mass against diameter freely, and again with the exponent held at 8/3."""
    design = np.column_stack([np.ones_like(log_diameter), log_diameter])
    # Same benign NumPy 2.x BLAS flags the solvers in scaling_law guard against:
    # matmul reports divide-by-zero on entirely finite inputs and a correct result.
    with clean_fp_state():
        coefficients, *_ = np.linalg.lstsq(design, log_mass, rcond=None)
        free_residual = log_mass - design @ coefficients

    constrained_intercept = float(np.mean(log_mass - WBE_EXPONENT * log_diameter))
    constrained_residual = log_mass - (constrained_intercept + WBE_EXPONENT * log_diameter)

    return WBEBaseline(
        free_exponent=float(coefficients[1]),
        free_intercept=float(coefficients[0]),
        free_rmsle=float(np.sqrt(np.mean(free_residual**2))),
        constrained_intercept=constrained_intercept,
        constrained_rmsle=float(np.sqrt(np.mean(constrained_residual**2))),
    )
