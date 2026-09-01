"""Result 13: the same audit on a scaling law from a different science.

Every other result here is about one database, which is the honest ceiling on
all of them. Result 11 widened it to rows the standard set does not contain, but
those rows still come from the same ITPA collection and the same devices, so the
finding could still be a property of tokamaks rather than of how scaling laws
are validated. This module answers that by leaving fusion entirely.

Mammalian basal metabolic rate against body mass is the same object as a
confinement scaling law, and it is the oldest one: Kleiber (1932) found that
metabolic rate scales as body mass to the 3/4, and that exponent is still the
published baseline a fitted model has to beat. The structural correspondence is
exact where it needs to be:

    IPB98(y,2), a published analytic law   ->  Kleiber's law, BMR proportional
                                               to mass^(3/4)
    tokamak, the unit a device is          ->  taxonomic order, the unit a
    a member of                                species is a member of
    machine size, 1.82x from the           ->  body mass, spanning two orders of
    database to ITER                           magnitude across order medians
    leave-one-tokamak-out                  ->  leave-one-order-out
    the ITER-matched size cut              ->  mass-ordered cuts

The analysis is deliberately run through ``scaling_audit``, the domain-agnostic
module, rather than through a copy of ``hdb5``. That is the point twice over: it
tests the reusable code on real external data, and it means the finding is
produced by the same procedure a reader would apply to their own dataset.

One way this problem is *simpler* than the tokamak one, and it is a feature. It
has a single predictor. There is no feature engineering to argue about and no
possibility that a model wins by finding a better combination of inputs: the
only thing separating the models is functional form. If the reversal appears
even here, it is not about feature space.

Data
----
Supplement 1 of the metabolic-rate compilation deposited on Figshare (article
3549807), a tab-separated table of basal and field metabolic rates with body
masses and per-record source citations. Third-party data, fetched on demand and
pinned by SHA-256 exactly as the HDB5 files are, not redistributed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
import hdb5

# Pinned the same way, and for the same reason, as ``hdb5.HDB5_STD5_SHA256``:
# every number Result 13 reports is a statement about these exact bytes, so a
# silent revision upstream must fail loudly rather than move a result.
ALLOMETRY_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/5616942"
ALLOMETRY_SHA256 = "ab22d9ae8f35e96d7fbf5d8e53053d647f0594d74f8917888169993ce668571e"
ALLOMETRY_N_BYTES = 56587
DEFAULT_ALLOMETRY_FILENAME = "allometry_bmr.txt"

# The file marks every absent measurement with this sentinel rather than leaving
# the field blank, so it parses as a valid number and would sail through any
# check that only tested for nulls.
MISSING_SENTINEL = -9999.0

# Kleiber's exponent. Published in 1932 and still the reference value; this is
# the analogue of IPB98(y,2)'s exponents, and like them it is used here as
# something to be tested rather than assumed.
KLEIBER_EXPONENT = 0.75

GROUP_COLUMN = "order"
TARGET_COLUMN = "bmr_ml_o2_per_hour"
MASS_COLUMN = "mass_g"
LOG_FEATURES: tuple[str, ...] = ("log_mass_g",)

# Below this an order's score is dominated by which handful of species happen to
# be in it. Mirrors ``hdb5.MIN_HELD_OUT_ROWS`` in intent.
MIN_HELD_OUT_ROWS = 10


def default_allometry_path() -> Path:
    return config.get_data_raw_dir() / DEFAULT_ALLOMETRY_FILENAME


def verify_allometry_file(path: Path | str) -> hdb5.DatasetFingerprint:
    """Fingerprint the file and raise unless it is the pinned deposit."""
    fingerprint = hdb5.fingerprint_file(path, read_shape=False)
    if fingerprint.sha256 == ALLOMETRY_SHA256 and fingerprint.n_bytes == ALLOMETRY_N_BYTES:
        return fingerprint
    raise hdb5.DatasetIntegrityError(
        f"Allometry dataset integrity check failed for {fingerprint.path}.\n"
        f"  expected sha256 {ALLOMETRY_SHA256} ({ALLOMETRY_N_BYTES} bytes)\n"
        f"  observed sha256 {fingerprint.sha256} ({fingerprint.n_bytes} bytes)\n"
        f"Re-fetch from {ALLOMETRY_DOWNLOAD_URL}. Every Result 13 number under "
        "results/ is a statement about the pinned bytes, so a mismatch means "
        "those numbers must be regenerated before they can be compared."
    )


def download_allometry(
    destination: Path | None = None, *, overwrite: bool = False, verify: bool = True
) -> Path:
    """Fetch the Figshare deposit into the raw data directory.

    Verified on the staged temporary file, so a download that does not match the
    pin never lands at the target path.
    """
    import urllib.request

    from storage import atomic_output_path

    target = (
        Path(destination).expanduser().resolve() if destination else default_allometry_path()
    )
    if target.exists() and not overwrite:
        if verify:
            verify_allometry_file(target)
        return target
    with atomic_output_path(target) as temp_path:
        with urllib.request.urlopen(ALLOMETRY_DOWNLOAD_URL) as response:
            temp_path.write_bytes(response.read())
        if verify:
            verify_allometry_file(temp_path)
    return target


def load_allometry_raw(
    path: Path | str | None = None, *, verify: bool | None = None
) -> pd.DataFrame:
    """Parse the deposit as it actually is, which is not as it appears.

    Two things about this file break a naive read and neither raises. It is tab
    separated with **carriage-return** line endings, so a default read returns a
    single row of 623 concatenated records rather than failing; and the column
    names carry trailing whitespace.
    """
    resolved = Path(path).expanduser().resolve() if path else default_allometry_path()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Allometry dataset not found at {resolved}. Run "
            f"`python3 -c 'import allometry; allometry.download_allometry()'`, or "
            f"download {ALLOMETRY_DOWNLOAD_URL} to that path."
        )
    if verify is None:
        verify = True
    if verify:
        verify_allometry_file(resolved)

    frame = pd.read_csv(resolved, sep="\t", lineterminator="\r")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def prepare_dataset(path: Path | str | None = None, *, verify: bool | None = None) -> pd.DataFrame:
    """Clean to the basal-metabolic-rate rows with a usable mass, in logs.

    The field-metabolic-rate columns are dropped rather than merged: they are a
    different measurement taken on a different set of animals at a different
    body mass, and pooling them would put two quantities under one target.
    """
    raw = load_allometry_raw(path, verify=verify)
    frame = pd.DataFrame(
        {
            GROUP_COLUMN: raw["Order"].astype(str).str.strip(),
            "species": raw["Species"].astype(str).str.strip(),
            TARGET_COLUMN: pd.to_numeric(raw["BMR (mlO2/hour)"], errors="coerce"),
            MASS_COLUMN: pd.to_numeric(raw["Body mass for BMR (gr)"], errors="coerce"),
        }
    )
    frame = frame.replace(MISSING_SENTINEL, np.nan).dropna(
        subset=[TARGET_COLUMN, MASS_COLUMN]
    )
    # Logs are taken, so non-positive values are dropped rather than clipped.
    frame = frame[(frame[TARGET_COLUMN] > 0) & (frame[MASS_COLUMN] > 0)]
    frame["log_mass_g"] = np.log(frame[MASS_COLUMN].to_numpy(dtype=float))
    frame["log_bmr"] = np.log(frame[TARGET_COLUMN].to_numpy(dtype=float))
    return frame.reset_index(drop=True)


def eligible_orders(dataset: pd.DataFrame, *, min_rows: int = MIN_HELD_OUT_ROWS) -> list[str]:
    """Orders with enough species to be scored, largest first by median mass."""
    counts = dataset.groupby(GROUP_COLUMN).size()
    keep = counts[counts >= min_rows].index
    medians = dataset[dataset[GROUP_COLUMN].isin(keep)].groupby(GROUP_COLUMN)[MASS_COLUMN].median()
    return [str(name) for name in medians.sort_values().index]


def order_mass_medians(dataset: pd.DataFrame) -> dict[str, float]:
    """Median body mass per order: the axis the ordered split extrapolates along."""
    medians = dataset.groupby(GROUP_COLUMN)[MASS_COLUMN].median()
    return {str(name): float(value) for name, value in medians.items()}


@dataclass(frozen=True)
class KleiberBaseline:
    """Kleiber's law with the exponent published and the coefficient fitted.

    The published law fixes the *exponent* at 3/4. Its coefficient is a
    units-dependent normalisation rather than a claim, and this compilation
    reports metabolic rate in ml O2 per hour against mass in grams, which is not
    the unit system Kleiber's constant is quoted in. So the exponent comes from
    1932 and the intercept is fitted.

    That makes this baseline blind in a way IPB98(y,2) is *not* in Results 4 to
    12: the intercept is fitted on the training fold only, so the baseline never
    sees the held-out order. Worth stating plainly, because it means the
    comparison here is if anything harder on the published law than the tokamak
    one is, not easier.
    """

    log_coefficient: float
    exponent: float = KLEIBER_EXPONENT

    def predict_log(self, log_mass: np.ndarray) -> np.ndarray:
        return self.log_coefficient + self.exponent * np.asarray(log_mass, dtype=float)


def fit_kleiber(log_mass: np.ndarray, log_bmr: np.ndarray) -> KleiberBaseline:
    """Least squares for the intercept alone, with the exponent held at 3/4."""
    residual = np.asarray(log_bmr, dtype=float) - KLEIBER_EXPONENT * np.asarray(
        log_mass, dtype=float
    )
    return KleiberBaseline(log_coefficient=float(np.mean(residual)))
