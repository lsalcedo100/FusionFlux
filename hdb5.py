"""Energy-confinement-time modeling on the real ITPA H-mode database (HDB5).

Unlike the synthetic ``neutron_yield`` pipeline, this path trains on real
experimental data: the ITPA Global H-mode Confinement Database, standard
analysis set ``STD5`` (version 5.2.3), published on the Open Science Framework
(https://osf.io/drwcq). Each row is a quasi-stationary time slice from a real
tokamak discharge.

The target is the thermal energy confinement time ``TAUTH`` (seconds). The
features are the engineering/operating parameters that the published IPB98(y,2)
and ITPA20 scaling laws regress against. The confinement time itself is never
used as an input, so there is no target leakage.

We report every model against the analytic IPB98(y,2) scaling law evaluated on
the same data, so "did the model actually learn something" is answered against a
real physics baseline rather than against the mean.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from storage import write_dataframe_csv_atomic, write_json_strict

# --- Real-data source -------------------------------------------------------

HDB5_OSF_PROJECT = "https://osf.io/drwcq"
HDB5_STD5_DOWNLOAD_URL = "https://osf.io/download/38htb/"
DEFAULT_HDB5_FILENAME = "hdb5_std5.csv"

# --- Data integrity ---------------------------------------------------------
#
# Every number in ``results/RESULTS.md`` is a statement about one specific file
# that this repository does not contain and does not control: it is fetched at
# run time from a third-party host. Without a pin, "reproducible" means "runs
# again", not "produces the same result". A silent upstream revision, a
# truncated download or a wrong OSF node would all reproduce cleanly and give
# different numbers, and nothing in the pipeline would notice.
#
# So the analysed file is pinned by content hash. The digest below was taken
# from a fresh download of HDB5_STD5_DOWNLOAD_URL and verified byte-identical to
# the copy the published results were computed from. Loading any other bytes
# through the default path is an error rather than a warning: a mismatch means
# the results are not comparable to the published ones, which is exactly the
# case that must not pass quietly.
#
# If upstream legitimately publishes a new revision, re-run
# ``python3 hdb5.py verify --print-only`` to read the new digest, update the
# three constants together, and regenerate every artifact under ``results/``.
# Do not update the pin without regenerating: the pin's whole purpose is that it
# and the reported numbers move as one.
HDB5_STD5_SHA256 = "67601c2da5c51f90cf6298ff499cccc74d09ac80c2b98c7dde0d8db3ebb9ac5b"
HDB5_STD5_N_BYTES = 879645
# Shape of the *raw* CSV as downloaded, before any cleaning. Checked alongside
# the digest because it is the property a human can verify against the source
# publication, and because it localises a mismatch: a wrong digest with the
# right shape is a revision, a wrong shape is a different dataset.
HDB5_STD5_RAW_SHAPE = (6228, 15)

# Read in blocks rather than slurping the file: the digest is also used on
# caller-supplied files, which carry no size guarantee.
_HASH_BLOCK_BYTES = 1 << 20

# Map the raw HDB5 column names to clean canonical names. Values are the raw
# columns; ``abs_value`` marks signed quantities (current/field carry a sign for
# direction that is physically irrelevant to confinement).
CANONICAL_COLUMN_SOURCES = {
    "tau_th_s": ("TAUTH", False),
    "ip_ma": ("IP", True),
    "bt_t": ("BT", True),
    "ne_line_1e19_m3": ("NEL", False),
    "p_loss_mw": ("PLTH", False),
    "r_m": ("RGEO", False),
    "kappa": ("KAPPAA", False),
    "inverse_aspect_ratio": ("EPS", False),
    "m_eff_amu": ("MEFF", False),
}
TOKAMAK_COLUMN = "TOK"
SHOT_COLUMN = "SHOT"

TARGET_COLUMN = "tau_th_s"
GROUP_COLUMN = "group_id"
TOKAMAK_LABEL_COLUMN = "tokamak"
DEVICE_COLUMN = "device"

# Two of the 18 ``TOK`` labels are the same physical tokamak after a wall
# retrofit: ``JETILW`` is JET with the ITER-like wall, ``AUGW`` is ASDEX Upgrade
# with tungsten. The distinction matters for confinement physics, so the
# database is right to separate them, but it means the 18 machine labels are 16
# devices. Anything that resamples "machines" to get an uncertainty on tokamaks
# in general has to decide which it means: treating JET and JETILW as
# independent draws counts one device twice.
#
# ``ASDEX`` is deliberately not folded into ``AUG``. ASDEX Upgrade is a separate,
# later machine rather than a rewall of ASDEX.
WALL_VARIANT_DEVICES = {"JETILW": "JET", "AUGW": "AUG"}

# Base engineering inputs (all strictly positive after cleaning).
BASE_ENGINEERING_COLUMNS = (
    "ip_ma",
    "bt_t",
    "ne_line_1e19_m3",
    "p_loss_mw",
    "r_m",
    "kappa",
    "inverse_aspect_ratio",
    "m_eff_amu",
    "a_m",
)

# Leak-free model features: logs of the positive engineering inputs plus the
# analytic IPB98 prediction as a physics prior. None of these use the target.
MODEL_FEATURE_COLUMNS = (
    "log_ip_ma",
    "log_bt_t",
    "log_ne_line_1e19_m3",
    "log_p_loss_mw",
    "log_r_m",
    "log_kappa",
    "log_inverse_aspect_ratio",
    "log_m_eff_amu",
    "log_a_m",
    "log_ipb98y2_tau_s",
)

RANDOM_STATE = config.RANDOM_STATE
N_CV_FOLDS = 5


@dataclass(frozen=True)
class ModelScore:
    model_name: str
    cv_rmsle: float
    cv_r2_log: float
    cv_mae_s: float


# --- Loading and cleaning ---------------------------------------------------


def default_hdb5_path() -> Path:
    return config.get_data_raw_dir() / DEFAULT_HDB5_FILENAME


class DatasetIntegrityError(RuntimeError):
    """The dataset on disk is not the one the published results were computed on.

    Deliberately not a subclass of ``ValueError``: callers that broadly catch
    input errors should not be able to swallow this one by accident.
    """


@dataclass(frozen=True)
class DatasetFingerprint:
    """Content identity of a dataset file, independent of where it came from."""

    path: str
    sha256: str
    n_bytes: int
    # None when the file could not be parsed as CSV; the digest is still valid
    # and is the thing worth reporting in that case.
    n_rows: int | None = None
    n_columns: int | None = None

    @property
    def matches_pin(self) -> bool:
        """Whether this is byte-for-byte the pinned STD5 revision."""
        return self.sha256 == HDB5_STD5_SHA256 and self.n_bytes == HDB5_STD5_N_BYTES

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "n_bytes": self.n_bytes,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "matches_pin": self.matches_pin,
            "pinned_sha256": HDB5_STD5_SHA256,
        }


def sha256_of_file(path: Path | str) -> str:
    """Streaming SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_file(path: Path | str, *, read_shape: bool = True) -> DatasetFingerprint:
    """Hash a file and, when it parses as CSV, record its shape alongside."""
    resolved = Path(path).expanduser().resolve()
    n_rows: int | None = None
    n_columns: int | None = None
    if read_shape:
        try:
            shape = pd.read_csv(resolved, low_memory=False).shape
        except Exception:  # noqa: BLE001 - shape is a nicety; the digest is the point
            pass
        else:
            n_rows, n_columns = int(shape[0]), int(shape[1])
    return DatasetFingerprint(
        path=str(resolved),
        sha256=sha256_of_file(resolved),
        n_bytes=resolved.stat().st_size,
        n_rows=n_rows,
        n_columns=n_columns,
    )


def verify_hdb5_file(path: Path | str) -> DatasetFingerprint:
    """Fingerprint a file and raise unless it is the pinned STD5 revision.

    The error message distinguishes the three ways this fails, because they call
    for different responses: a truncated or interrupted download should be
    re-fetched, a shape mismatch means the wrong file entirely, and a
    right-shape-wrong-digest means upstream revised the data and every number in
    ``results/`` needs regenerating before it can be trusted.
    """
    fingerprint = fingerprint_file(path)
    if fingerprint.matches_pin:
        return fingerprint

    observed_shape = (fingerprint.n_rows, fingerprint.n_columns)
    if fingerprint.n_rows is None:
        detail = "the file does not parse as CSV, so it is most likely a truncated or failed download"
    elif observed_shape != HDB5_STD5_RAW_SHAPE:
        detail = (
            f"shape is {observed_shape} but the pinned STD5 revision is "
            f"{HDB5_STD5_RAW_SHAPE}, so this is a different dataset rather than "
            "a revision of the expected one"
        )
    else:
        detail = (
            "shape matches but the contents differ, so upstream has revised the "
            "data; every result under results/ must be regenerated before it can "
            "be compared against the published numbers"
        )

    raise DatasetIntegrityError(
        f"HDB5 integrity check failed for {fingerprint.path}: {detail}.\n"
        f"  expected sha256 {HDB5_STD5_SHA256} ({HDB5_STD5_N_BYTES} bytes)\n"
        f"  observed sha256 {fingerprint.sha256} ({fingerprint.n_bytes} bytes)\n"
        "Re-fetch with `python3 hdb5.py download --overwrite`, or inspect with "
        "`python3 hdb5.py verify --print-only`. To analyse a deliberately "
        "different file, pass --dataset-path (unpinned files are reported, not "
        "enforced)."
    )


def download_hdb5_std5(
    destination: Path | None = None,
    *,
    overwrite: bool = False,
    verify: bool = True,
) -> Path:
    """Download the ITPA HDB5 STD5 dataset from OSF into the raw data directory.

    The database is third-party scientific data (please cite Verdoolaege et al.,
    Nucl. Fusion 61 076006, 2021); it is fetched on demand rather than
    redistributed in the repository.

    Verification happens on the staged temporary file, before the atomic rename.
    A download that does not match the pin therefore never lands at the target
    path at all, so a failed fetch cannot leave a plausible-looking wrong dataset
    behind for the next run to pick up silently.
    """
    import urllib.request

    from storage import atomic_output_path

    target = Path(destination).expanduser().resolve() if destination else default_hdb5_path()
    if target.exists() and not overwrite:
        if verify:
            verify_hdb5_file(target)
        return target
    with atomic_output_path(target) as temp_path:
        with urllib.request.urlopen(HDB5_STD5_DOWNLOAD_URL) as response:
            temp_path.write_bytes(response.read())
        if verify:
            verify_hdb5_file(temp_path)
    return target


def load_hdb5_dataframe(
    path: Path | str | None = None,
    *,
    verify: bool | None = None,
) -> pd.DataFrame:
    """Load the raw HDB5 CSV, verifying the pin when this is the canonical file.

    ``verify=None`` (the default) enforces the pin exactly when the resolved path
    is the default one, which is the file every published result was computed
    from. An explicitly supplied ``--dataset-path`` is deliberately *not*
    enforced: analysing a different revision or a subset is a legitimate thing to
    do, and the pin exists to stop that happening by accident, not to forbid it.
    """
    resolved = Path(path).expanduser().resolve() if path is not None else default_hdb5_path()
    if not resolved.exists():
        raise FileNotFoundError(
            f"HDB5 dataset not found: {resolved}. Download the STD5 set from "
            f"{HDB5_OSF_PROJECT} ({HDB5_STD5_DOWNLOAD_URL}) and save it there, "
            "or pass --dataset-path."
        )
    should_verify = resolved == default_hdb5_path() if verify is None else verify
    if should_verify:
        verify_hdb5_file(resolved)
    return pd.read_csv(resolved, low_memory=False)


def map_to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw HDB5 columns to clean canonical names and drop invalid rows."""
    missing = [
        source
        for source, _ in CANONICAL_COLUMN_SOURCES.values()
        if source not in raw.columns
    ]
    for identity_column in (TOKAMAK_COLUMN, SHOT_COLUMN):
        if identity_column not in raw.columns:
            missing.append(identity_column)
    if missing:
        raise ValueError(f"HDB5 input is missing expected columns: {sorted(set(missing))}")

    frame = pd.DataFrame(index=raw.index)
    for canonical, (source, take_abs) in CANONICAL_COLUMN_SOURCES.items():
        values = pd.to_numeric(raw[source], errors="coerce")
        frame[canonical] = values.abs() if take_abs else values

    frame[TOKAMAK_LABEL_COLUMN] = raw[TOKAMAK_COLUMN].astype(str)
    frame[GROUP_COLUMN] = (
        raw[TOKAMAK_COLUMN].astype(str) + "::" + raw[SHOT_COLUMN].astype(str)
    )
    frame["a_m"] = frame["inverse_aspect_ratio"] * frame["r_m"]

    positive_columns = [TARGET_COLUMN, *BASE_ENGINEERING_COLUMNS]
    finite_and_positive = frame[positive_columns].notna().all(axis=1) & (
        frame[positive_columns] > 0
    ).all(axis=1)
    cleaned = frame.loc[finite_and_positive].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No valid HDB5 rows remained after cleaning.")
    return cleaned


def ipb98y2_tau_s(frame: pd.DataFrame) -> pd.Series:
    """Analytic IPB98(y,2) energy-confinement scaling law (seconds).

    tau = 0.0562 * Ip^0.93 * Bt^0.15 * ne19^0.41 * P^-0.69
                 * R^1.97 * eps^0.58 * kappa^0.78 * M^0.19
    """
    return (
        0.0562
        * np.power(frame["ip_ma"], 0.93)
        * np.power(frame["bt_t"], 0.15)
        * np.power(frame["ne_line_1e19_m3"], 0.41)
        * np.power(frame["p_loss_mw"], -0.69)
        * np.power(frame["r_m"], 1.97)
        * np.power(frame["inverse_aspect_ratio"], 0.58)
        * np.power(frame["kappa"], 0.78)
        * np.power(frame["m_eff_amu"], 0.19)
    )


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leak-free log features and the analytic IPB98 prior."""
    featured = frame.copy()
    featured["ipb98y2_tau_s"] = ipb98y2_tau_s(featured)
    for column in (*BASE_ENGINEERING_COLUMNS, "ipb98y2_tau_s"):
        featured[f"log_{column}"] = np.log(featured[column])
    return featured


def prepare_dataset_from_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and feature-engineer an already-loaded raw HDB5 frame."""
    return build_features(map_to_canonical(raw))


def prepare_dataset(path: Path | str | None = None, *, verify: bool | None = None) -> pd.DataFrame:
    return prepare_dataset_from_frame(load_hdb5_dataframe(path, verify=verify))


def with_device_column(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add :data:`DEVICE_COLUMN`, folding wall variants back onto one device."""
    framed = dataset.copy()
    framed[DEVICE_COLUMN] = (
        framed[TOKAMAK_LABEL_COLUMN].map(WALL_VARIANT_DEVICES).fillna(framed[TOKAMAK_LABEL_COLUMN])
    )
    return framed


def dataset_provenance(path: Path | str | None = None) -> dict[str, object]:
    """Identity of the file an analysis actually read, for stamping into results.

    Every generated artifact under ``results/`` carries this. A number and the
    hash of the bytes it came from travel together, so a reader can tell whether
    a result predates an upstream revision without taking the repository's word
    for which revision it used.
    """
    resolved = Path(path).expanduser().resolve() if path is not None else default_hdb5_path()
    return fingerprint_file(resolved).to_json()


# --- Metrics ----------------------------------------------------------------


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    clipped = np.clip(y_pred, 1e-6, None)
    return float(np.sqrt(np.mean((np.log(clipped) - np.log(y_true)) ** 2)))


def _r2_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    clipped = np.clip(y_pred, 1e-6, None)
    log_true = np.log(y_true)
    residual = np.sum((log_true - np.log(clipped)) ** 2)
    total = np.sum((log_true - log_true.mean()) ** 2)
    return float(1.0 - residual / total) if total > 0 else float("nan")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


# --- Models -----------------------------------------------------------------


def build_model_zoo() -> dict[str, Pipeline]:
    """Regressors that predict log(tau); each is wrapped so callers see tau."""
    return {
        "mean_baseline": Pipeline([("model", DummyRegressor(strategy="mean"))]),
        # The IPB98 prior and log_a_m are exact linear combinations of the other
        # log features, so the design matrix is singular; the SVD solver is
        # numerically stable under that collinearity. (The richer feature set is
        # kept because the IPB98 prior measurably helps the tree models.)
        "ridge_loglinear": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=1.0, solver="svd"))]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))]
        ),
    }


@contextlib.contextmanager
def _suppress_benign_matmul_warnings() -> Iterator[None]:
    """Silence spurious NumPy 2.0 BLAS floating-point-state warnings.

    The singular (perfectly collinear) design matrix makes NumPy's ``matmul``
    emit "divide by zero"/"overflow"/"invalid value" RuntimeWarnings on some
    BLAS backends even though inputs are finite and results are correct and
    stable. Scoped narrowly to the fit/predict numerics.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        yield


def _grouped_cv_predictions(
    estimator: Pipeline,
    features: pd.DataFrame,
    log_target: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> np.ndarray:
    """Out-of-fold predictions of log(tau) using grouped CV."""
    predictions = np.empty_like(log_target, dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    with _suppress_benign_matmul_warnings():
        for train_idx, test_idx in splitter.split(features, log_target, groups):
            fold_model = clone_pipeline(estimator)
            fit_pipeline(fold_model, features.iloc[train_idx], log_target[train_idx])
            predictions[test_idx] = fold_model.predict(features.iloc[test_idx])
    return predictions


def clone_pipeline(estimator: Pipeline) -> Pipeline:
    from sklearn.base import clone

    return clone(estimator)


def fit_pipeline(
    estimator: Pipeline, features: pd.DataFrame, log_target: np.ndarray
) -> Pipeline:
    """Fit in parallel, then predict single-threaded, so the output is reproducible.

    Under a fixed seed ``RandomForestRegressor`` grows bit-identical trees no
    matter how many workers it uses, but ``predict`` accumulates the per-tree
    averages into one shared array under a lock, so the summation order depends
    on thread scheduling and the last bit of every prediction with it. That is
    far below anything physical, and it is still enough to make every number in
    ``results/`` change on a rerun of an unchanged analysis, which is exactly the
    churn the pinned dataset hash and the fixed seeds exist to rule out.

    Fitting stays parallel because that is where the time goes (roughly 4x on
    this database); prediction is over one held-out machine at a time and costs
    nothing single-threaded.
    """
    estimator.fit(features, log_target)
    # ``getattr`` rather than ``.steps`` because the zoo also carries estimators
    # that are not pipelines: ``PowerLawResidualHybrid`` arrives here through
    # ``extra_models``. Nothing it wraps is threaded, so it has nothing to pin.
    for _, step in getattr(estimator, "steps", ()):
        if getattr(step, "n_jobs", None) is not None:
            step.n_jobs = 1
    return estimator


def evaluate_models(
    dataset: pd.DataFrame,
    *,
    n_splits: int = N_CV_FOLDS,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
    include_controls: bool = False,
    extra_models: dict[str, Pipeline] | None = None,
) -> list[ModelScore]:
    """Grouped cross-validation by discharge: interpolation within known machines.

    ``feature_columns`` exists so this can be run on the same blind feature set
    that :func:`leave_one_tokamak_out` uses. Comparing the two while the feature
    set also changes would confound the split with the features, and the whole
    point of that comparison is that only the split changes.

    ``include_controls`` mirrors the same flag on :func:`leave_one_tokamak_out`,
    so the control models can be scored under both splits rather than only the
    one they were introduced to probe. ``extra_models`` is the general form of
    that flag, for scoring an arbitrary set of additional estimators (the
    flexibility ladder in ``analysis_extrapolation`` is the caller that needs it).
    """
    features = dataset[list(feature_columns)]
    tau = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    groups = dataset[GROUP_COLUMN].to_numpy()
    effective_splits = min(n_splits, int(pd.Series(groups).nunique()))

    scores: list[ModelScore] = []

    # Physics baseline: the analytic scaling law, no training.
    ipb98 = dataset["ipb98y2_tau_s"].to_numpy(dtype=float)
    scores.append(
        ModelScore("ipb98y2_analytic", _rmsle(tau, ipb98), _r2_log(tau, ipb98), _mae(tau, ipb98))
    )

    zoo = _assemble_zoo(include_controls=include_controls, extra_models=extra_models)
    for name, estimator in zoo.items():
        oof_log = _grouped_cv_predictions(estimator, features, log_tau, groups, effective_splits)
        oof_tau = np.exp(oof_log)
        scores.append(
            ModelScore(name, _rmsle(tau, oof_tau), _r2_log(tau, oof_tau), _mae(tau, oof_tau))
        )

    scores.sort(key=lambda score: score.cv_rmsle)
    return scores


# --- Training orchestration -------------------------------------------------


@dataclass(frozen=True)
class ConfinementArtifact:
    model: Pipeline
    feature_columns: tuple[str, ...]
    target_column: str
    model_name: str

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        log_tau = self.model.predict(features[list(self.feature_columns)])
        return np.clip(np.exp(log_tau), 0.0, None)


def train_confinement_model(
    path: Path | str | None = None,
    *,
    output_dir: Path | None = None,
    n_splits: int = N_CV_FOLDS,
) -> dict[str, object]:
    import joblib

    dataset = prepare_dataset(path)
    scores = evaluate_models(dataset, n_splits=n_splits)

    baseline = next(score for score in scores if score.model_name == "ipb98y2_analytic")
    trainable = [score for score in scores if score.model_name != "ipb98y2_analytic"]
    best = min(trainable, key=lambda score: score.cv_rmsle)

    features = dataset[list(MODEL_FEATURE_COLUMNS)]
    log_tau = np.log(dataset[TARGET_COLUMN].to_numpy(dtype=float))
    final_model = clone_pipeline(build_model_zoo()[best.model_name])
    with _suppress_benign_matmul_warnings():
        fit_pipeline(final_model, features, log_tau)

    artifact = ConfinementArtifact(
        model=final_model,
        feature_columns=MODEL_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        model_name=best.model_name,
    )

    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else config.get_data_processed_dir() / "hdb5_confinement"
    )
    model_path = resolved_output_dir / "confinement_model.joblib"
    metrics_path = resolved_output_dir / "confinement_metrics.csv"
    metadata_path = resolved_output_dir / "confinement_metadata.json"

    from storage import ensure_parent_directory

    ensure_parent_directory(model_path)
    joblib.dump(artifact, model_path)

    metrics_frame = pd.DataFrame(
        [
            {
                "model_name": score.model_name,
                "cv_rmsle": score.cv_rmsle,
                "cv_r2_log": score.cv_r2_log,
                "cv_mae_s": score.cv_mae_s,
                "is_selected": score.model_name == best.model_name,
                "is_physics_baseline": score.model_name == "ipb98y2_analytic",
            }
            for score in scores
        ]
    )
    write_dataframe_csv_atomic(metrics_path, metrics_frame)

    metadata = {
        "dataset_source": HDB5_OSF_PROJECT,
        "target_column": TARGET_COLUMN,
        "n_rows": int(len(dataset)),
        "n_groups": int(dataset[GROUP_COLUMN].nunique()),
        "tokamaks": sorted(dataset[TOKAMAK_LABEL_COLUMN].unique().tolist()),
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "selected_model": best.model_name,
        "selected_cv_rmsle": best.cv_rmsle,
        "selected_cv_r2_log": best.cv_r2_log,
        "physics_baseline_cv_rmsle": baseline.cv_rmsle,
        "physics_baseline_cv_r2_log": baseline.cv_r2_log,
        "beats_physics_baseline": bool(best.cv_rmsle < baseline.cv_rmsle),
        "model_path": str(model_path),
    }
    write_json_strict(metadata_path, metadata)
    return metadata


# --- Prediction -------------------------------------------------------------

# The engineering inputs a caller must supply to score a single operating point.
# ``a_m`` is derived from ``inverse_aspect_ratio * r_m`` exactly as in cleaning,
# so it is not requested directly.
SINGLE_CASE_INPUT_COLUMNS = (
    "ip_ma",
    "bt_t",
    "ne_line_1e19_m3",
    "p_loss_mw",
    "r_m",
    "kappa",
    "inverse_aspect_ratio",
    "m_eff_amu",
)


def default_model_path() -> Path:
    return config.get_data_processed_dir() / "hdb5_confinement" / "confinement_model.joblib"


def load_confinement_artifact(model_path: Path | str | None = None) -> ConfinementArtifact:
    """Load a saved :class:`ConfinementArtifact` from disk."""
    import joblib

    resolved = (
        Path(model_path).expanduser().resolve() if model_path is not None else default_model_path()
    )
    if not resolved.exists():
        raise FileNotFoundError(
            f"Confinement model not found: {resolved}. Run `python3 hdb5.py train` first, "
            "or pass --model-path."
        )
    artifact = joblib.load(resolved)
    if not isinstance(artifact, ConfinementArtifact):
        raise TypeError(
            f"{resolved} does not contain a ConfinementArtifact (got {type(artifact).__name__})."
        )
    return artifact


def build_single_case_frame(inputs: dict[str, float]) -> pd.DataFrame:
    """Build a one-row featured frame from raw engineering inputs.

    Applies the same derivation (``a_m``), positivity, and feature engineering
    used in training so a single prediction goes through identical preprocessing.
    """
    missing = [column for column in SINGLE_CASE_INPUT_COLUMNS if inputs.get(column) is None]
    if missing:
        raise ValueError(f"Missing required inputs: {sorted(missing)}")

    frame = pd.DataFrame([{column: float(inputs[column]) for column in SINGLE_CASE_INPUT_COLUMNS}])
    invalid = [
        column
        for column in SINGLE_CASE_INPUT_COLUMNS
        if not (np.isfinite(frame[column]) & (frame[column] > 0)).all()
    ]
    if invalid:
        raise ValueError(f"Inputs must be finite and strictly positive: {sorted(invalid)}")

    frame["a_m"] = frame["inverse_aspect_ratio"] * frame["r_m"]
    return build_features(frame)


def predict_single_case(
    inputs: dict[str, float], *, model_path: Path | str | None = None
) -> dict[str, object]:
    """Predict confinement time for one operating point using a saved artifact.

    This is the low-level path: it returns the saved model's point estimate and
    nothing else. It does not attach an interval, does not measure how far the
    operating point sits from the training data, and does not check whether the
    answer lies above the range the artifact can even emit. Handed ITER's
    parameters it will return a tree ensemble's number, roughly 0.4 s, against
    the 3.6 s the analytic law gives, with no complaint: Results 4b, 4c and 5
    are the reasons that number is wrong and none of them are consulted here.

    Use :func:`predictor.predict` instead unless you specifically want one saved
    artifact's raw output. It reports the same estimate alongside a calibrated
    interval, the extrapolation distance, and an explicit refusal when the point
    is beyond what this study measured.
    """
    artifact = load_confinement_artifact(model_path)
    featured = build_single_case_frame(inputs)
    predicted = float(artifact.predict(featured)[0])
    return {
        "predicted_tau_th_s": predicted,
        "ipb98y2_tau_s": float(featured["ipb98y2_tau_s"].iloc[0]),
        "model_name": artifact.model_name,
        "model_path": str(
            Path(model_path).expanduser().resolve() if model_path is not None else default_model_path()
        ),
    }


# --- Leave-one-tokamak-out extrapolation ------------------------------------

# A machine needs enough held-out rows for its RMSLE to mean anything.
MIN_HELD_OUT_ROWS = 30

# The IPB98 prior is a fixed log-linear combination of the eight engineering log
# features (see ``results/RESULTS.md``, Result 1), and its exponents were fitted
# on this same database, held-out machine included. Keeping it as a feature
# therefore leaks the held-out machine into every model that uses it, which is
# exactly what an extrapolation test must exclude. Engineering parameters only.
BLIND_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in MODEL_FEATURE_COLUMNS if column != "log_ipb98y2_tau_s"
)


@dataclass(frozen=True)
class MachineScore:
    """One model's error on one entirely held-out machine."""

    model_name: str
    tokamak: str
    n_held_out_rows: int
    rmsle: float
    r2_log: float
    mae_s: float
    # False for ``ipb98y2_analytic``: it was fitted on this database, so it saw
    # the held-out machine and is a reference point rather than a fair baseline.
    is_blind: bool


@dataclass(frozen=True)
class ExtrapolationDiagnostic:
    """How far outside the training data a held-out machine actually sits.

    Separates two explanations for a model failing on an unseen machine: that
    its functional form is wrong, or that the machine simply lies outside the
    training range. Tree ensembles average training targets, so their output is
    bounded by ``[min(y_train), max(y_train))]`` and they cannot reach a machine
    above that range no matter how good the features are. A log-linear power law
    has no such bound.
    """

    tokamak: str
    n_held_out_rows: int
    # Distance of the held-out mean log-feature vector from the training mean,
    # in training-covariance units.
    feature_mahalanobis: float
    n_features_outside_train_range: int
    # Fraction of held-out rows whose true tau lies outside the training target
    # range. Tree ensembles structurally cannot predict these.
    target_above_train_max_fraction: float
    target_below_train_min_fraction: float
    # log(max y_held_out) - log(max y_train). Positive means the machine reaches
    # confinement times no tree in the forest can output.
    log_target_headroom: float


def build_control_models() -> dict[str, Pipeline]:
    """Controls that separate *constrained* from merely *able to extrapolate*.

    Ridge beating the trees on an unseen machine has two candidate explanations
    that the main zoo cannot tell apart: the power-law form is physically right,
    or ridge is simply the only model in the zoo that extrapolates at all (a
    tree ensemble averages training targets, so it is bounded by the training
    range by construction).

    ``ridge_log_quadratic`` is the discriminating case. It is flexible, with
    curvature and every pairwise interaction in log space, but it is still a
    polynomial and so still extrapolates. If flexibility per se were the
    problem it should degrade like the trees; if the log-linear power-law form
    is what matters it should degrade like plain ridge.
    """
    from sklearn.preprocessing import PolynomialFeatures

    return {
        "ridge_log_quadratic": Pipeline(
            [
                ("expand", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0, solver="svd")),
            ]
        ),
    }


def _assemble_zoo(
    *,
    include_controls: bool = False,
    extra_models: dict[str, Pipeline] | None = None,
) -> dict[str, Pipeline]:
    """The model zoo plus any controls and caller-supplied extras.

    Shared by both splits so a model can never be scored under one and silently
    missing from the other, which would make the two columns of Result 4
    incomparable for that row.
    """
    zoo = dict(build_model_zoo())
    if include_controls:
        zoo.update(build_control_models())
    if extra_models:
        overlapping = set(extra_models) & set(zoo)
        if overlapping:
            raise ValueError(
                f"extra_models would silently replace existing models: {sorted(overlapping)}"
            )
        zoo.update(extra_models)
    return zoo


def eligible_tokamaks(dataset: pd.DataFrame, *, min_rows: int = MIN_HELD_OUT_ROWS) -> list[str]:
    """Machines with enough rows to hold out, in descending order of size."""
    counts = dataset[TOKAMAK_LABEL_COLUMN].value_counts()
    return [str(name) for name, count in counts.items() if int(count) >= min_rows]


def _mahalanobis_of_mean(train_features: np.ndarray, held_out_features: np.ndarray) -> float:
    """Distance between held-out and training feature means, in training units.

    The training covariance is singular by construction: ``log a_m`` is exactly
    ``log r_m + log inverse_aspect_ratio`` because ``a_m`` is derived that way in
    cleaning. A plain inverse would blow up, so this uses the pseudo-inverse,
    which measures the distance within the subspace the data actually spans and
    ignores the null directions where every row sits at the same value anyway.
    """
    difference = held_out_features.mean(axis=0) - train_features.mean(axis=0)
    covariance = np.cov(train_features, rowvar=False)
    covariance = np.atleast_2d(covariance)
    quadratic = float(difference @ np.linalg.pinv(covariance) @ difference)
    return float(np.sqrt(max(quadratic, 0.0)))


def extrapolation_diagnostic(
    dataset: pd.DataFrame,
    tokamak: str,
    *,
    feature_columns: tuple[str, ...] = BLIND_FEATURE_COLUMNS,
) -> ExtrapolationDiagnostic:
    """Measure how far one machine sits outside the rest of the database."""
    held_mask = (dataset[TOKAMAK_LABEL_COLUMN] == tokamak).to_numpy()
    if not held_mask.any():
        raise ValueError(f"No rows for tokamak {tokamak!r}.")
    if held_mask.all():
        raise ValueError(f"Tokamak {tokamak!r} is the only machine in the dataset.")

    columns = list(feature_columns)
    train_features = dataset.loc[~held_mask, columns].to_numpy(dtype=float)
    held_features = dataset.loc[held_mask, columns].to_numpy(dtype=float)

    train_tau = dataset.loc[~held_mask, TARGET_COLUMN].to_numpy(dtype=float)
    held_tau = dataset.loc[held_mask, TARGET_COLUMN].to_numpy(dtype=float)

    train_minimum = train_features.min(axis=0)
    train_maximum = train_features.max(axis=0)
    held_median = np.median(held_features, axis=0)
    outside = int(np.sum((held_median < train_minimum) | (held_median > train_maximum)))

    return ExtrapolationDiagnostic(
        tokamak=tokamak,
        n_held_out_rows=int(held_mask.sum()),
        feature_mahalanobis=_mahalanobis_of_mean(train_features, held_features),
        n_features_outside_train_range=outside,
        target_above_train_max_fraction=float(np.mean(held_tau > train_tau.max())),
        target_below_train_min_fraction=float(np.mean(held_tau < train_tau.min())),
        log_target_headroom=float(np.log(held_tau.max()) - np.log(train_tau.max())),
    )


def leave_one_tokamak_out(
    dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = BLIND_FEATURE_COLUMNS,
    min_rows: int = MIN_HELD_OUT_ROWS,
    include_ipb98_reference: bool = True,
    include_controls: bool = False,
    extra_models: dict[str, Pipeline] | None = None,
) -> pd.DataFrame:
    """Score every model on each machine in turn, trained on all the others.

    Grouped CV by discharge measures interpolation inside machines the model has
    already seen. This measures the case a scaling law actually exists for:
    predicting a device that was not in the training set at all.
    """
    machines = eligible_tokamaks(dataset, min_rows=min_rows)
    if not machines:
        raise ValueError(
            f"No tokamak has at least {min_rows} rows; nothing can be held out."
        )

    columns = list(feature_columns)
    features = dataset[columns]
    tau = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    labels = dataset[TOKAMAK_LABEL_COLUMN].to_numpy()

    records: list[dict[str, object]] = []
    for machine in machines:
        held_mask = labels == machine
        held_index = np.flatnonzero(held_mask)
        train_index = np.flatnonzero(~held_mask)
        held_tau = tau[held_index]

        if include_ipb98_reference:
            reference = dataset["ipb98y2_tau_s"].to_numpy(dtype=float)[held_index]
            records.append(
                asdict(
                    MachineScore(
                        model_name="ipb98y2_analytic",
                        tokamak=machine,
                        n_held_out_rows=len(held_index),
                        rmsle=_rmsle(held_tau, reference),
                        r2_log=_r2_log(held_tau, reference),
                        mae_s=_mae(held_tau, reference),
                        is_blind=False,
                    )
                )
            )

        zoo = _assemble_zoo(include_controls=include_controls, extra_models=extra_models)
        for name, estimator in zoo.items():
            model = clone_pipeline(estimator)
            with _suppress_benign_matmul_warnings():
                fit_pipeline(model, features.iloc[train_index], log_tau[train_index])
                predicted = np.exp(model.predict(features.iloc[held_index]))
            records.append(
                asdict(
                    MachineScore(
                        model_name=name,
                        tokamak=machine,
                        n_held_out_rows=len(held_index),
                        rmsle=_rmsle(held_tau, predicted),
                        r2_log=_r2_log(held_tau, predicted),
                        mae_s=_mae(held_tau, predicted),
                        is_blind=True,
                    )
                )
            )

    return pd.DataFrame(records)


def summarize_leave_one_tokamak_out(per_machine: pd.DataFrame) -> pd.DataFrame:
    """Mean and median RMSLE per model across held-out machines.

    Every machine counts once regardless of size. That is deliberate (the claim
    is about machines, not rows) but it means the summary is dominated by the
    many small devices; ``per_machine`` is where the story actually is.
    """
    summary = (
        per_machine.groupby(["model_name", "is_blind"], as_index=False)
        .agg(
            mean_rmsle=("rmsle", "mean"),
            median_rmsle=("rmsle", "median"),
            worst_rmsle=("rmsle", "max"),
            n_machines=("tokamak", "nunique"),
        )
        .sort_values("mean_rmsle")
        .reset_index(drop=True)
    )
    return summary


def extrapolation_report(
    dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = BLIND_FEATURE_COLUMNS,
    min_rows: int = MIN_HELD_OUT_ROWS,
    include_controls: bool = False,
    extra_models: dict[str, Pipeline] | None = None,
) -> pd.DataFrame:
    """Per-machine RMSLE joined to how far outside the training data it sits."""
    per_machine = leave_one_tokamak_out(
        dataset,
        feature_columns=feature_columns,
        min_rows=min_rows,
        include_controls=include_controls,
        extra_models=extra_models,
    )
    diagnostics = pd.DataFrame(
        [
            asdict(extrapolation_diagnostic(dataset, machine, feature_columns=feature_columns))
            for machine in eligible_tokamaks(dataset, min_rows=min_rows)
        ]
    )
    return per_machine.merge(
        diagnostics.drop(columns=["n_held_out_rows"]), on="tokamak", how="left"
    )


# --- Size-ordered extrapolation: the ITER direction -------------------------
#
# Leave-one-tokamak-out holds out a machine but leaves 12 others spanning much
# of its parameter range, so it measures "predict a machine you have not seen"
# while still interpolating in size. A next-step device is not that case. ITER's
# major radius is 6.2 m; the largest row in this database is JT-60U at 3.40 m.
# The relevant question is what happens when the target sits *beyond the size
# range entirely*, and leave-one-out cannot ask it.
#
# So order the machines by size and cut. Train on the machines below the cut,
# predict every machine above it. Sweeping the cut sweeps the size ratio being
# asked for, and one rung of that sweep matches the ITER jump almost exactly:
# training on everything up to DIII-D (max R 1.865 m) and predicting up to
# JT-60U (3.40 m) is a factor of 1.823, against ITER's 6.2 / 3.40 = 1.824. The
# database contains, inside itself, an extrapolation the same size as the one
# that separates it from ITER.
ITER_MAJOR_RADIUS_M = 6.2

# Cuts need enough machines below them for a fit to mean anything, and enough
# rows above them for the held-out score to mean anything.
MIN_TRAIN_MACHINES = 3
MIN_TEST_ROWS = MIN_HELD_OUT_ROWS

# Spherical tokamaks (START, MAST, NSTX) sit at inverse aspect ratios around
# 0.7, roughly double the conventional 0.3. They are also among the smallest
# machines, so they land in the training set of every size cut and a critic can
# fairly say the extrapolation being measured is as much in shape as in size.
# ``conventional_aspect_ratio_only`` drops them as a control.
MAX_CONVENTIONAL_INVERSE_ASPECT_RATIO = 0.5


@dataclass(frozen=True)
class MachineSize:
    """One machine's position along the size axis the cut is taken on."""

    tokamak: str
    n_rows: int
    r_median_m: float
    r_max_m: float
    inverse_aspect_ratio_median: float


@dataclass(frozen=True)
class SizeSplit:
    """Train on every machine below a size cut, predict every machine above it."""

    n_train_machines: int
    train_machines: tuple[str, ...]
    test_machines: tuple[str, ...]
    n_train_rows: int
    n_test_rows: int
    # Largest major radius the model saw, and the largest it is asked about.
    train_r_max_m: float
    test_r_max_m: float
    # test_r_max / train_r_max: the size extrapolation the split demands.
    size_ratio: float
    # Fraction of held-out rows whose true tau exceeds the training maximum.
    # These are the rows Result 4c says no tree ensemble can reach at all.
    target_above_train_max_fraction: float

    def to_json(self) -> dict[str, object]:
        return {
            "n_train_machines": self.n_train_machines,
            "train_machines": list(self.train_machines),
            "test_machines": list(self.test_machines),
            "n_train_rows": self.n_train_rows,
            "n_test_rows": self.n_test_rows,
            "train_r_max_m": self.train_r_max_m,
            "test_r_max_m": self.test_r_max_m,
            "size_ratio": self.size_ratio,
            "target_above_train_max_fraction": self.target_above_train_max_fraction,
        }


def machine_sizes(dataset: pd.DataFrame) -> list[MachineSize]:
    """Every machine's size, ascending by median major radius.

    Median rather than mean: a machine's rows are not uniformly distributed over
    its operating space, and the median is the robust summary of "how big is this
    device". The cut is taken on the median, but the *ratio* a split demands is
    reported from the maxima, because what matters for extrapolation is the edge
    of the range the model saw rather than its centre.
    """
    grouped = dataset.groupby(TOKAMAK_LABEL_COLUMN)
    sizes = [
        MachineSize(
            tokamak=str(name),
            n_rows=int(len(rows)),
            r_median_m=float(rows["r_m"].median()),
            r_max_m=float(rows["r_m"].max()),
            inverse_aspect_ratio_median=float(rows["inverse_aspect_ratio"].median()),
        )
        for name, rows in grouped
    ]
    sizes.sort(key=lambda size: size.r_median_m)
    return sizes


def iter_size_ratio(dataset: pd.DataFrame) -> float:
    """How far beyond this database ITER sits, as a major-radius ratio."""
    return ITER_MAJOR_RADIUS_M / float(dataset["r_m"].max())


def size_ordered_splits(
    dataset: pd.DataFrame,
    *,
    min_train_machines: int = MIN_TRAIN_MACHINES,
    min_test_rows: int = MIN_TEST_ROWS,
    conventional_aspect_ratio_only: bool = False,
) -> list[SizeSplit]:
    """Every usable size cut, from the smallest training set to the largest.

    Cut ``k`` trains on the ``k`` smallest machines and predicts all the rest.
    Unlike leave-one-out this holds out *several* machines at once, which is the
    point: the held-out set is everything the model has no size precedent for.
    """
    sizes = machine_sizes(dataset)
    if conventional_aspect_ratio_only:
        sizes = [
            size
            for size in sizes
            if size.inverse_aspect_ratio_median <= MAX_CONVENTIONAL_INVERSE_ASPECT_RATIO
        ]
    keep = {size.tokamak for size in sizes}
    frame = dataset[dataset[TOKAMAK_LABEL_COLUMN].isin(keep)]

    ordered = [size.tokamak for size in sizes]
    labels = frame[TOKAMAK_LABEL_COLUMN].to_numpy()
    radius = frame["r_m"].to_numpy(dtype=float)
    tau = frame[TARGET_COLUMN].to_numpy(dtype=float)

    splits: list[SizeSplit] = []
    for cut in range(min_train_machines, len(ordered)):
        train_machines = tuple(ordered[:cut])
        test_machines = tuple(ordered[cut:])
        train_mask = np.isin(labels, list(train_machines))
        test_mask = ~train_mask
        if int(test_mask.sum()) < min_test_rows:
            continue
        train_r_max = float(radius[train_mask].max())
        test_r_max = float(radius[test_mask].max())
        splits.append(
            SizeSplit(
                n_train_machines=cut,
                train_machines=train_machines,
                test_machines=test_machines,
                n_train_rows=int(train_mask.sum()),
                n_test_rows=int(test_mask.sum()),
                train_r_max_m=train_r_max,
                test_r_max_m=test_r_max,
                size_ratio=test_r_max / train_r_max,
                target_above_train_max_fraction=float(
                    np.mean(tau[test_mask] > tau[train_mask].max())
                ),
            )
        )
    return splits


def iter_matched_split(
    dataset: pd.DataFrame,
    splits: list[SizeSplit],
) -> SizeSplit:
    """The cut whose size ratio is closest to the jump from this data to ITER.

    This is the rung the headline number comes from. Picking it by proximity to
    the ITER ratio rather than by eye keeps it a property of the data: if the
    database gains a larger machine, the matched rung moves on its own.
    """
    if not splits:
        raise ValueError("No usable size splits; cannot match the ITER ratio.")
    target = iter_size_ratio(dataset)
    return min(splits, key=lambda split: abs(np.log(split.size_ratio) - np.log(target)))


def score_size_split(
    dataset: pd.DataFrame,
    split: SizeSplit,
    *,
    feature_columns: tuple[str, ...] = BLIND_FEATURE_COLUMNS,
    include_ipb98_reference: bool = True,
    include_controls: bool = False,
    extra_models: dict[str, Pipeline] | None = None,
    per_machine: bool = False,
    min_rows: int = MIN_HELD_OUT_ROWS,
) -> pd.DataFrame:
    """Fit every model below the cut and score it above the cut.

    Pooled over all held-out rows by default. With ``per_machine`` it also
    returns one row per held-out machine with at least ``min_rows`` rows, so a
    pooled score dominated by one large device can be checked against the
    individual machines behind it.
    """
    columns = list(feature_columns)
    labels = dataset[TOKAMAK_LABEL_COLUMN].to_numpy()
    train_mask = np.isin(labels, list(split.train_machines))
    test_mask = np.isin(labels, list(split.test_machines))

    features = dataset[columns]
    tau = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    train_index = np.flatnonzero(train_mask)
    test_index = np.flatnonzero(test_mask)

    def _scopes() -> list[tuple[str, np.ndarray]]:
        scopes: list[tuple[str, np.ndarray]] = [("__pooled__", test_index)]
        if per_machine:
            for machine in split.test_machines:
                machine_index = np.flatnonzero(labels == machine)
                if machine_index.size >= min_rows:
                    scopes.append((machine, machine_index))
        return scopes

    records: list[dict[str, object]] = []

    def _record(model_name: str, scope: str, index: np.ndarray, predicted: np.ndarray, blind: bool) -> None:
        truth = tau[index]
        records.append(
            {
                "model_name": model_name,
                "scope": scope,
                "n_train_machines": split.n_train_machines,
                "size_ratio": split.size_ratio,
                "n_held_out_rows": int(index.size),
                "rmsle": _rmsle(truth, predicted),
                "r2_log": _r2_log(truth, predicted),
                "mae_s": _mae(truth, predicted),
                "is_blind": blind,
            }
        )

    if include_ipb98_reference:
        reference = dataset["ipb98y2_tau_s"].to_numpy(dtype=float)
        for scope, index in _scopes():
            _record("ipb98y2_analytic", scope, index, reference[index], False)

    zoo = _assemble_zoo(include_controls=include_controls, extra_models=extra_models)
    for name, estimator in zoo.items():
        model = clone_pipeline(estimator)
        with _suppress_benign_matmul_warnings():
            fit_pipeline(model, features.iloc[train_index], log_tau[train_index])
            for scope, index in _scopes():
                predicted = np.exp(model.predict(features.iloc[index]))
                _record(name, scope, index, predicted, True)

    return pd.DataFrame(records)


def size_extrapolation_report(
    dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = BLIND_FEATURE_COLUMNS,
    min_train_machines: int = MIN_TRAIN_MACHINES,
    min_test_rows: int = MIN_TEST_ROWS,
    include_controls: bool = False,
    extra_models: dict[str, Pipeline] | None = None,
    conventional_aspect_ratio_only: bool = False,
) -> tuple[pd.DataFrame, list[SizeSplit]]:
    """Score every model at every size cut. Returns the scores and the splits."""
    splits = size_ordered_splits(
        dataset,
        min_train_machines=min_train_machines,
        min_test_rows=min_test_rows,
        conventional_aspect_ratio_only=conventional_aspect_ratio_only,
    )
    if not splits:
        raise ValueError("No size cut leaves enough machines to train on and rows to score.")
    frames = [
        score_size_split(
            dataset,
            split,
            feature_columns=feature_columns,
            include_controls=include_controls,
            extra_models=extra_models,
        )
        for split in splits
    ]
    return pd.concat(frames, ignore_index=True), splits


# --- The constrained hybrid: a power law plus a shrunk residual correction ---
#
# Result 4 diagnoses a failure and stops there. The trees win by 41% under
# grouped CV and lose to a log-linear power law on every one of 13 held-out
# machines, and Result 4d attributes that to functional form: the power law is
# the only form on the ladder whose error is bounded away from the data.
#
# The obvious cure is to stop choosing. Fit the power law, then learn a
# correction on its *log residuals* and damp that correction hard. At zero
# damping the model is whatever the correction is; at full damping it is exactly
# the power law. In between it inherits the power law's extrapolation behaviour
# (the base term is unbounded and log-linear, so it keeps growing with size the
# way the physics says it should) and spends its flexibility on whatever
# in-range structure the power law leaves on the table.
#
# The damping factor is a single explicit multiplier rather than a correction
# hyperparameter, because the two correction families regularize along entirely
# different axes (a ridge penalty on curvature terms, a depth and a learning
# rate on trees) and no shared sweep exists over those. Multiplying the fitted
# correction by ``shrinkage`` is the one knob that means the same thing for
# both, which is what makes the two frontiers comparable.
#
# The two correction families differ in exactly the way Result 4c cares about:
#
#   ridge on a degree-2 log expansion   unbounded. Curvature terms are free to
#                                       diverge away from the data, so this can
#                                       damage the base model's extrapolation
#                                       and only the penalty stops it.
#   depth-2 gradient-boosted trees      bounded by the residual range it was
#                                       trained on, by the same argument as
#                                       Result 4c. It cannot rescue the base
#                                       model far from the data and it cannot
#                                       wreck it either.
#
# Whether either helps is a measurement, not a design claim, and both outcomes
# are worth reporting: a hybrid that beats plain ridge under CV while keeping
# most of its leave-one-machine-out score is a point neither pure model reaches,
# and a hybrid that cannot do that says the residual left over after the power
# law is not learnable across machines at all.

# Deliberately strong. These are not tuned for CV score; they are set so the
# correction starts out heavily constrained and ``shrinkage`` does the rest.
DEFAULT_RIDGE_CORRECTION_ALPHA = 1000.0
DEFAULT_GBM_CORRECTION_DEPTH = 2
DEFAULT_GBM_CORRECTION_LEARNING_RATE = 0.05
DEFAULT_GBM_CORRECTION_ITERATIONS = 200
DEFAULT_GBM_CORRECTION_L2 = 10.0

# The shrinkage sweep. 0.0 is the pure power law and 1.0 is the undamped
# correction, so the endpoints are both interpretable models rather than
# arbitrary rungs.
SHRINKAGE_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


class PowerLawResidualHybrid(RegressorMixin, BaseEstimator):
    """Log-linear ridge, plus ``shrinkage`` times a correction on its residuals.

    ``RegressorMixin`` comes first deliberately. scikit-learn resolves estimator
    tags along the MRO, so with ``BaseEstimator`` leading, the mixin's
    ``__sklearn_tags__`` never runs and ``sklearn.base.is_regressor`` returns
    False for what is plainly a regressor. Nothing in this repository dispatches
    on that tag, so it would not have changed a number here; it would have
    broken the first caller that handed this to a scikit-learn utility.

    Both stages are fitted on the same rows. That is deliberate: the correction
    is meant to absorb structure the power law systematically misses, not noise
    the power law happens to leave on held-out rows, and it is held back by its
    own penalty rather than by a sample split. The penalty is the thing being
    swept, so making it do all the work keeps the sweep interpretable.

    At ``shrinkage=0`` this is exactly ``ridge_loglinear`` from the zoo, which
    is what makes the sweep a frontier anchored at a model already reported
    rather than a new family that has to be argued for separately.
    """

    def __init__(
        self,
        correction: str = "ridge",
        *,
        base_alpha: float = 1.0,
        shrinkage: float = 1.0,
        ridge_correction_alpha: float = DEFAULT_RIDGE_CORRECTION_ALPHA,
        polynomial_degree: int = 2,
        gbm_max_depth: int = DEFAULT_GBM_CORRECTION_DEPTH,
        gbm_learning_rate: float = DEFAULT_GBM_CORRECTION_LEARNING_RATE,
        gbm_max_iter: int = DEFAULT_GBM_CORRECTION_ITERATIONS,
        gbm_l2_regularization: float = DEFAULT_GBM_CORRECTION_L2,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.correction = correction
        self.base_alpha = base_alpha
        self.shrinkage = shrinkage
        self.ridge_correction_alpha = ridge_correction_alpha
        self.polynomial_degree = polynomial_degree
        self.gbm_max_depth = gbm_max_depth
        self.gbm_learning_rate = gbm_learning_rate
        self.gbm_max_iter = gbm_max_iter
        self.gbm_l2_regularization = gbm_l2_regularization
        self.random_state = random_state

    def _build_base(self) -> Pipeline:
        # Identical to ``ridge_loglinear`` in the zoo, including the SVD solver,
        # so that shrinkage=0 reproduces that row exactly rather than
        # approximately. The design matrix is singular by Result 1.
        return Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=self.base_alpha, solver="svd"))]
        )

    def _build_correction(self) -> Pipeline:
        if self.correction == "ridge":
            from sklearn.preprocessing import PolynomialFeatures

            return Pipeline(
                [
                    (
                        "expand",
                        PolynomialFeatures(degree=self.polynomial_degree, include_bias=False),
                    ),
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=self.ridge_correction_alpha, solver="svd")),
                ]
            )
        if self.correction == "gbm":
            return Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_depth=self.gbm_max_depth,
                            learning_rate=self.gbm_learning_rate,
                            max_iter=self.gbm_max_iter,
                            l2_regularization=self.gbm_l2_regularization,
                            # Off so the fit is a deterministic function of the
                            # training rows: with early stopping on, the number
                            # of boosting rounds would depend on an internal
                            # validation split and two rungs of the shrinkage
                            # sweep could differ by more than shrinkage.
                            early_stopping=False,
                            random_state=self.random_state,
                        ),
                    )
                ]
            )
        raise ValueError(
            f"Unknown correction {self.correction!r}; expected 'ridge' or 'gbm'."
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "PowerLawResidualHybrid":
        target = np.asarray(y, dtype=float)
        self.base_ = self._build_base()
        with _suppress_benign_matmul_warnings():
            self.base_.fit(X, target)
            base_prediction = self.base_.predict(X)
        self.residual_ = target - base_prediction
        self.correction_ = self._build_correction()
        with _suppress_benign_matmul_warnings():
            self.correction_.fit(X, self.residual_)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        with _suppress_benign_matmul_warnings():
            base_prediction = self.base_.predict(X)
            if self.shrinkage == 0.0:
                # Short-circuited so that shrinkage=0 is bit-identical to the
                # base model rather than base + 0 * (something possibly
                # non-finite far outside the training range).
                return np.asarray(base_prediction, dtype=float)
            correction = self.correction_.predict(X)
        return np.asarray(base_prediction + self.shrinkage * correction, dtype=float)


def build_hybrid_models(
    shrinkage_grid: tuple[float, ...] = SHRINKAGE_GRID,
    *,
    corrections: tuple[str, ...] = ("ridge", "gbm"),
) -> dict[str, Pipeline]:
    """One hybrid per (correction family, shrinkage) rung of the sweep.

    Wrapped in a ``Pipeline`` so these drop into the same zoo, the same
    ``clone_pipeline`` and the same three splits as every other model, with no
    special-casing anywhere downstream.
    """
    models: dict[str, Pipeline] = {}
    for correction in corrections:
        for shrinkage in shrinkage_grid:
            name = f"hybrid_{correction}_s{shrinkage:g}".replace(".", "p")
            models[name] = Pipeline(
                [("model", PowerLawResidualHybrid(correction=correction, shrinkage=shrinkage))]
            )
    return models


def hybrid_model_name(correction: str, shrinkage: float) -> str:
    """The zoo key for one rung, so callers never rebuild the naming by hand."""
    return f"hybrid_{correction}_s{shrinkage:g}".replace(".", "p")


# --- Split-conformal intervals and where their calibration goes ------------
#
# Every number above is a point error. For a next-step device the point error is
# not the deliverable: nobody builds a machine on a single predicted number, and
# the question asked of a model is "what range should we plan for", which is an
# interval. Result 4 says the model is wrong on a new machine. The sharper
# statement, and the one that decides whether the interval is usable, is whether
# it is *confidently* wrong.
#
# Split conformal prediction is the natural tool because it assumes almost
# nothing about the model. Fit on part of the training data, take the absolute
# log residuals on a held-back calibration part, and use their (1-alpha)
# quantile as a half-width. Under exchangeability of the calibration and test
# rows, the resulting interval covers at least 1-alpha of test rows in finite
# samples, whatever the model is.
#
# That proviso is the entire point here. Exchangeability holds for grouped CV by
# discharge: calibration discharges and test discharges are both held-out
# discharges from the same machines, so the guarantee applies and 90% intervals
# should cover about 90%. It fails for leave-one-tokamak-out by construction:
# the calibration rows come from machines the model trained on and the test rows
# come from a machine it has never seen. Nothing then guarantees anything, and
# the size of the resulting shortfall is a direct measurement of how far the
# distribution moved.
#
# So the collapse below is not a bug in the conformal method and not a failure
# to calibrate properly. It is the assumption being false, measured. What is
# worth reporting is how far it falls, and whether it falls in step with the
# same Mahalanobis distance that Result 4b showed the point errors track.
#
# Interval *width* is reported alongside coverage throughout, because coverage
# alone is trivial to win: an interval wide enough to be useless covers
# everything. A model is only doing well if it holds coverage at a width that
# still says something.

# 90% intervals. Reported as the nominal level everywhere so the gap to the
# empirical number is the quantity being read.
DEFAULT_CONFORMAL_ALPHA = 0.10

# Share of *discharges*, not rows, held back to calibrate. Splitting by
# discharge matters for the same reason the CV split does: several time slices
# from one shot are not independent, so a row-level calibration split would put
# near-duplicates of calibration rows into the fit and return half-widths that
# are too small.
DEFAULT_CALIBRATION_FRACTION = 0.25

# A conformal half-width at level 1-alpha needs at least ceil((n+1)*(1-alpha))
# calibration points to exist at all; below that the finite-sample construction
# returns an infinite interval rather than a wrong one. At alpha=0.10 that is 19
# points, and this floor is set well above it so the reported half-widths are
# quantiles of a usable sample rather than of a handful.
MIN_CALIBRATION_ROWS = 100

CONFORMAL_SEED = 20240811


@dataclass(frozen=True)
class CoverageScore:
    """Empirical coverage of one model's nominal 1-alpha intervals, one scope."""

    model_name: str
    # "grouped_cv", "leave_one_tokamak_out" or "size_cut".
    split: str
    # A machine name, or "__pooled__" for every scored row in the arm.
    scope: str
    n_rows: int
    nominal_coverage: float
    empirical_coverage: float
    # Conformal half-width in log units: the interval is
    # [exp(log_pred - h), exp(log_pred + h)], so exp(h) is the multiplicative
    # factor and the interval is that factor up and down from the point
    # prediction. Reported because coverage without width is not a result.
    median_half_width_log: float
    median_interval_factor: float
    rmsle: float
    is_blind: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def split_conformal_half_width(
    calibration_abs_residuals: np.ndarray,
    alpha: float = DEFAULT_CONFORMAL_ALPHA,
) -> float:
    """The finite-sample-valid conformal quantile of absolute log residuals.

    The rank is ``ceil((n + 1) * (1 - alpha))`` rather than the plain empirical
    quantile. That +1 is the whole finite-sample guarantee: it accounts for the
    test point itself being one of the n+1 exchangeable scores. Using
    ``numpy.quantile`` instead would undercover slightly at every n, by an
    amount that shrinks as 1/n and so would be invisible on the large arms here
    and material on the small ones.

    Returns ``inf`` when the sample is too small for the level to be attainable,
    which is the honest answer: no finite interval has the guarantee there.
    """
    scores = np.asarray(calibration_abs_residuals, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def _calibration_mask_by_group(
    groups: np.ndarray,
    *,
    calibration_fraction: float,
    seed: int,
) -> np.ndarray:
    """Boolean mask over rows selecting whole discharges to calibrate on."""
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Need at least two discharges to split off a calibration set.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    n_calibration = int(round(unique_groups.size * calibration_fraction))
    n_calibration = min(max(n_calibration, 1), unique_groups.size - 1)
    calibration_groups = shuffled[:n_calibration]
    return np.isin(groups, calibration_groups)


def _conformal_arm(
    dataset: pd.DataFrame,
    *,
    train_index: np.ndarray,
    test_index: np.ndarray,
    zoo: dict[str, Pipeline],
    feature_columns: tuple[str, ...],
    alpha: float,
    calibration_fraction: float,
    seed: int,
    include_ipb98_reference: bool,
) -> pd.DataFrame:
    """Fit, calibrate and score intervals for one train/test index pair.

    Returns one row per (model, test row): whether the interval covered, how
    wide it was, and the log residual. Aggregation into coverage is left to the
    caller because the three arms group differently (by machine, by fold, or
    pooled) and collapsing here would throw away what they need.
    """
    columns = list(feature_columns)
    features = dataset[columns]
    tau = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    groups = dataset[GROUP_COLUMN].to_numpy()
    labels = dataset[TOKAMAK_LABEL_COLUMN].to_numpy()

    calibration_selector = _calibration_mask_by_group(
        groups[train_index], calibration_fraction=calibration_fraction, seed=seed
    )
    calibration_index = train_index[calibration_selector]
    proper_index = train_index[~calibration_selector]
    if calibration_index.size < MIN_CALIBRATION_ROWS:
        raise ValueError(
            f"Only {calibration_index.size} calibration rows; need {MIN_CALIBRATION_ROWS}."
        )

    frames: list[pd.DataFrame] = []

    def _collect(
        model_name: str,
        calibration_log_prediction: np.ndarray,
        test_log_prediction: np.ndarray,
        blind: bool,
    ) -> None:
        half_width = split_conformal_half_width(
            np.abs(log_tau[calibration_index] - calibration_log_prediction), alpha
        )
        test_residual = log_tau[test_index] - test_log_prediction
        frames.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "is_blind": blind,
                    "row": test_index,
                    "tokamak": labels[test_index],
                    "covered": np.abs(test_residual) <= half_width,
                    "half_width_log": half_width,
                    "abs_log_residual": np.abs(test_residual),
                }
            )
        )

    if include_ipb98_reference:
        # No fitting, so the analytic law is calibrated on exactly the same
        # calibration rows as everything else and differs only in where its
        # predictions come from.
        analytic = np.log(dataset["ipb98y2_tau_s"].to_numpy(dtype=float))
        _collect(
            "ipb98y2_analytic", analytic[calibration_index], analytic[test_index], False
        )

    for name, estimator in zoo.items():
        model = clone_pipeline(estimator)
        with _suppress_benign_matmul_warnings():
            fit_pipeline(model, features.iloc[proper_index], log_tau[proper_index])
            calibration_prediction = model.predict(features.iloc[calibration_index])
            test_prediction = model.predict(features.iloc[test_index])
        _collect(name, calibration_prediction, test_prediction, True)

    return pd.concat(frames, ignore_index=True)


def _summarize_coverage(
    per_row: pd.DataFrame,
    *,
    split: str,
    alpha: float,
    by_machine: bool,
) -> pd.DataFrame:
    """Collapse per-row coverage flags into pooled and per-machine scores."""
    nominal = 1.0 - alpha

    def _score(frame: pd.DataFrame, scope: str) -> dict[str, object]:
        return asdict(
            CoverageScore(
                model_name=str(frame["model_name"].iloc[0]),
                split=split,
                scope=scope,
                n_rows=int(len(frame)),
                nominal_coverage=nominal,
                empirical_coverage=float(frame["covered"].mean()),
                median_half_width_log=float(frame["half_width_log"].median()),
                median_interval_factor=float(np.exp(frame["half_width_log"].median())),
                rmsle=float(np.sqrt(np.mean(frame["abs_log_residual"].to_numpy() ** 2))),
                is_blind=bool(frame["is_blind"].iloc[0]),
            )
        )

    records: list[dict[str, object]] = []
    for _, frame in per_row.groupby("model_name", sort=False):
        records.append(_score(frame, "__pooled__"))
        if by_machine:
            for machine, machine_frame in frame.groupby("tokamak", sort=False):
                records.append(_score(machine_frame, str(machine)))
    return pd.DataFrame(records)


def conformal_coverage_grouped_cv(
    dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = BLIND_FEATURE_COLUMNS,
    n_splits: int = N_CV_FOLDS,
    alpha: float = DEFAULT_CONFORMAL_ALPHA,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    include_controls: bool = False,
    extra_models: dict[str, Pipeline] | None = None,
    seed: int = CONFORMAL_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage under the in-distribution split, where the guarantee applies.

    Calibration and test rows are both held-out discharges from machines the
    model trained on, so they are exchangeable and empirical coverage should sit
    at the nominal level. This arm is the control: it establishes that the
    intervals are constructed correctly, so that the shortfall in the
    leave-one-machine-out arm cannot be blamed on the construction.
    """
    groups = dataset[GROUP_COLUMN].to_numpy()
    effective_splits = min(n_splits, int(pd.Series(groups).nunique()))
    zoo = _assemble_zoo(include_controls=include_controls, extra_models=extra_models)
    splitter = GroupKFold(n_splits=effective_splits)
    features = dataset[list(feature_columns)]
    log_tau = np.log(dataset[TARGET_COLUMN].to_numpy(dtype=float))

    frames = [
        _conformal_arm(
            dataset,
            train_index=train_index,
            test_index=test_index,
            zoo=zoo,
            feature_columns=feature_columns,
            alpha=alpha,
            calibration_fraction=calibration_fraction,
            # Offset per fold so the calibration discharges are not the same
            # draw every time, which would tie all five folds to one split.
            seed=seed + fold,
            include_ipb98_reference=True,
        )
        for fold, (train_index, test_index) in enumerate(
            splitter.split(features, log_tau, groups)
        )
    ]
    per_row = pd.concat(frames, ignore_index=True)
    return per_row, _summarize_coverage(
        per_row, split="grouped_cv", alpha=alpha, by_machine=True
    )


def conformal_coverage_leave_one_tokamak_out(
    dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = BLIND_FEATURE_COLUMNS,
    min_rows: int = MIN_HELD_OUT_ROWS,
    alpha: float = DEFAULT_CONFORMAL_ALPHA,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    include_controls: bool = False,
    extra_models: dict[str, Pipeline] | None = None,
    seed: int = CONFORMAL_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage on a machine the model never saw, where the guarantee does not.

    The interval is calibrated exactly as in the CV arm, on held-out discharges
    from the training machines, and then applied to a machine outside that set.
    That is precisely what a team with a new device would do, and it is the
    situation conformal prediction does not cover.
    """
    machines = eligible_tokamaks(dataset, min_rows=min_rows)
    if not machines:
        raise ValueError(f"No tokamak has at least {min_rows} rows; nothing can be held out.")
    zoo = _assemble_zoo(include_controls=include_controls, extra_models=extra_models)
    labels = dataset[TOKAMAK_LABEL_COLUMN].to_numpy()

    frames = [
        _conformal_arm(
            dataset,
            train_index=np.flatnonzero(labels != machine),
            test_index=np.flatnonzero(labels == machine),
            zoo=zoo,
            feature_columns=feature_columns,
            alpha=alpha,
            calibration_fraction=calibration_fraction,
            seed=seed + index,
            include_ipb98_reference=True,
        )
        for index, machine in enumerate(machines)
    ]
    per_row = pd.concat(frames, ignore_index=True)
    return per_row, _summarize_coverage(
        per_row, split="leave_one_tokamak_out", alpha=alpha, by_machine=True
    )


def conformal_coverage_size_split(
    dataset: pd.DataFrame,
    split: SizeSplit,
    *,
    feature_columns: tuple[str, ...] = BLIND_FEATURE_COLUMNS,
    alpha: float = DEFAULT_CONFORMAL_ALPHA,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    include_controls: bool = False,
    extra_models: dict[str, Pipeline] | None = None,
    seed: int = CONFORMAL_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage across a size cut: calibrate on small machines, predict large.

    The leave-one-machine-out arm breaks exchangeability in machine identity.
    This breaks it in the direction that matters for a next-step device, and at
    the ITER-matched rung it breaks it by the same factor.
    """
    labels = dataset[TOKAMAK_LABEL_COLUMN].to_numpy()
    per_row = _conformal_arm(
        dataset,
        train_index=np.flatnonzero(np.isin(labels, list(split.train_machines))),
        test_index=np.flatnonzero(np.isin(labels, list(split.test_machines))),
        zoo=_assemble_zoo(include_controls=include_controls, extra_models=extra_models),
        feature_columns=feature_columns,
        alpha=alpha,
        calibration_fraction=calibration_fraction,
        seed=seed,
        include_ipb98_reference=True,
    )
    return per_row, _summarize_coverage(
        per_row, split="size_cut", alpha=alpha, by_machine=True
    )


# --- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an energy-confinement-time model on the real ITPA HDB5 dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train on the HDB5 STD5 dataset.")
    train.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the HDB5 STD5 CSV (defaults to data/raw/hdb5_std5.csv).",
    )
    train.add_argument(
        "--cv-folds",
        type=int,
        default=N_CV_FOLDS,
        help="Number of grouped cross-validation folds.",
    )
    train.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Fetch the default HDB5 dataset from OSF first if it is not present.",
    )

    predict = subparsers.add_parser(
        "predict", help="Predict confinement time for one operating point."
    )
    predict.add_argument("--ip-ma", type=float, required=True, help="Plasma current (MA).")
    predict.add_argument("--bt-t", type=float, required=True, help="Toroidal field (T).")
    predict.add_argument(
        "--ne-line-1e19-m3",
        type=float,
        required=True,
        help="Line-averaged density (1e19 m^-3).",
    )
    predict.add_argument("--p-loss-mw", type=float, required=True, help="Loss power (MW).")
    predict.add_argument("--r-m", type=float, required=True, help="Major radius (m).")
    predict.add_argument("--kappa", type=float, required=True, help="Elongation.")
    predict.add_argument(
        "--inverse-aspect-ratio", type=float, required=True, help="Inverse aspect ratio (a/R)."
    )
    predict.add_argument("--m-eff-amu", type=float, required=True, help="Effective ion mass (amu).")
    predict.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a saved confinement_model.joblib (defaults to the trained artifact).",
    )

    evaluate = subparsers.add_parser("evaluate", help="Cross-validate models, print a report.")
    evaluate.add_argument("--dataset-path", type=str, default=None)
    evaluate.add_argument("--cv-folds", type=int, default=N_CV_FOLDS)

    extrapolate = subparsers.add_parser(
        "extrapolate",
        help="Leave-one-tokamak-out: score every model on machines it never saw.",
    )
    extrapolate.add_argument("--dataset-path", type=str, default=None)
    extrapolate.add_argument(
        "--min-rows",
        type=int,
        default=MIN_HELD_OUT_ROWS,
        help="Skip machines with fewer held-out rows than this.",
    )
    extrapolate.add_argument(
        "--keep-ipb98-feature",
        action="store_true",
        help=(
            "Keep log_ipb98y2_tau_s as a model feature. Off by default: its "
            "exponents were fitted on this database including the held-out "
            "machine, so it leaks."
        ),
    )
    extrapolate.add_argument(
        "--include-controls",
        action="store_true",
        help=(
            "Also fit ridge_log_quadratic, a flexible model that still "
            "extrapolates, to separate functional form from extrapolation."
        ),
    )
    extrapolate.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Write per-machine and summary CSVs here instead of only printing.",
    )

    size_extrapolate = subparsers.add_parser(
        "size-extrapolate",
        help="Train on the small machines, predict the large ones: the ITER direction.",
    )
    size_extrapolate.add_argument("--dataset-path", type=str, default=None)
    size_extrapolate.add_argument(
        "--min-train-machines",
        type=int,
        default=MIN_TRAIN_MACHINES,
        help="Smallest number of machines a training set may contain.",
    )
    size_extrapolate.add_argument(
        "--conventional-aspect-ratio-only",
        action="store_true",
        help=(
            "Drop the spherical tokamaks, so the extrapolation being measured is "
            "in size rather than in plasma shape."
        ),
    )
    size_extrapolate.add_argument(
        "--include-controls",
        action="store_true",
        help="Also fit the polynomial controls from build_control_models.",
    )
    size_extrapolate.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Write the sweep and the ITER-matched cut here instead of only printing.",
    )

    conformal = subparsers.add_parser(
        "conformal",
        help="Split-conformal interval coverage under each split: in range and out.",
    )
    conformal.add_argument("--dataset-path", type=str, default=None)
    conformal.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_CONFORMAL_ALPHA,
        help="Miss rate; intervals are nominal 1 - alpha (default 0.10, so 90%%).",
    )
    conformal.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
        help="Share of discharges held back from the fit to calibrate on.",
    )
    conformal.add_argument(
        "--include-controls",
        action="store_true",
        help="Also score the polynomial controls from build_control_models.",
    )
    conformal.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Write the per-machine coverage table here instead of only printing.",
    )

    download = subparsers.add_parser("download", help="Fetch the HDB5 STD5 dataset from OSF.")
    download.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if the dataset already exists.",
    )
    download.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the content-hash check. The download is then unpinned; results from it are not comparable.",
    )

    verify = subparsers.add_parser(
        "verify",
        help="Check the dataset on disk against the pinned content hash.",
    )
    verify.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="File to fingerprint (defaults to the canonical HDB5 path).",
    )
    verify.add_argument(
        "--print-only",
        action="store_true",
        help="Report the fingerprint and exit 0 even on a mismatch, for reading a new digest.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        path = download_hdb5_std5(overwrite=args.overwrite, verify=not args.no_verify)
        print(
            json.dumps(
                {"downloaded_to": str(path), "provenance": dataset_provenance(path)},
                indent=2,
            )
        )
        return
    if args.command == "verify":
        target = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else default_hdb5_path()
        if args.print_only:
            print(json.dumps(fingerprint_file(target).to_json(), indent=2))
            return
        fingerprint = verify_hdb5_file(target)
        print(json.dumps({"verified": True, **fingerprint.to_json()}, indent=2))
        return
    if args.command == "evaluate":
        dataset = prepare_dataset(args.dataset_path)
        scores = evaluate_models(dataset, n_splits=args.cv_folds)
        report = [
            {
                "model_name": score.model_name,
                "cv_rmsle": round(score.cv_rmsle, 4),
                "cv_r2_log": round(score.cv_r2_log, 4),
                "cv_mae_s": round(score.cv_mae_s, 4),
            }
            for score in scores
        ]
        print(json.dumps(report, indent=2))
        return
    if args.command == "extrapolate":
        dataset = prepare_dataset(args.dataset_path)
        feature_columns = (
            MODEL_FEATURE_COLUMNS if args.keep_ipb98_feature else BLIND_FEATURE_COLUMNS
        )
        per_machine = extrapolation_report(
            dataset,
            feature_columns=feature_columns,
            min_rows=args.min_rows,
            include_controls=args.include_controls,
        )
        summary = summarize_leave_one_tokamak_out(per_machine)
        if args.output_dir is not None:
            output_dir = Path(args.output_dir).expanduser().resolve()
            write_dataframe_csv_atomic(output_dir / "extrapolation_per_machine.csv", per_machine)
            write_dataframe_csv_atomic(output_dir / "extrapolation_summary.csv", summary)
        print(
            json.dumps(
                {
                    "feature_columns": list(feature_columns),
                    "ipb98_feature_included": bool(args.keep_ipb98_feature),
                    "n_machines": int(per_machine["tokamak"].nunique()),
                    "summary": summary.to_dict(orient="records"),
                },
                indent=2,
            )
        )
        return
    if args.command == "size-extrapolate":
        dataset = prepare_dataset(args.dataset_path)
        sweep, splits = size_extrapolation_report(
            dataset,
            min_train_machines=args.min_train_machines,
            include_controls=args.include_controls,
            conventional_aspect_ratio_only=args.conventional_aspect_ratio_only,
        )
        matched = iter_matched_split(dataset, splits)
        matched_scores = score_size_split(
            dataset,
            matched,
            include_controls=args.include_controls,
            per_machine=True,
        )
        if args.output_dir is not None:
            output_dir = Path(args.output_dir).expanduser().resolve()
            write_dataframe_csv_atomic(output_dir / "size_extrapolation_sweep.csv", sweep)
            write_dataframe_csv_atomic(
                output_dir / "size_extrapolation_iter_matched.csv", matched_scores
            )
        pooled = matched_scores[matched_scores["scope"] == "__pooled__"]
        print(
            json.dumps(
                {
                    "iter_size_ratio": iter_size_ratio(dataset),
                    "iter_matched_split": matched.to_json(),
                    "n_splits": len(splits),
                    "iter_matched_scores": pooled.drop(columns=["scope"]).to_dict(orient="records"),
                },
                indent=2,
            )
        )
        return
    if args.command == "conformal":
        dataset = prepare_dataset(args.dataset_path)
        shared = {
            "alpha": args.alpha,
            "calibration_fraction": args.calibration_fraction,
            "include_controls": args.include_controls,
        }
        _, cv_summary = conformal_coverage_grouped_cv(dataset, **shared)
        _, lomo_summary = conformal_coverage_leave_one_tokamak_out(dataset, **shared)
        matched = iter_matched_split(dataset, size_ordered_splits(dataset))
        _, size_summary = conformal_coverage_size_split(dataset, matched, **shared)
        summary = pd.concat(
            [cv_summary, lomo_summary, size_summary], ignore_index=True
        )
        if args.output_dir is not None:
            output_dir = Path(args.output_dir).expanduser().resolve()
            write_dataframe_csv_atomic(output_dir / "conformal_coverage.csv", summary)
        pooled = summary[summary["scope"] == "__pooled__"]
        print(
            json.dumps(
                {
                    "nominal_coverage": 1.0 - args.alpha,
                    "calibration_fraction": args.calibration_fraction,
                    "pooled": pooled.drop(columns=["scope"]).to_dict(orient="records"),
                },
                indent=2,
            )
        )
        return
    if args.command == "predict":
        result = predict_single_case(
            {
                "ip_ma": args.ip_ma,
                "bt_t": args.bt_t,
                "ne_line_1e19_m3": args.ne_line_1e19_m3,
                "p_loss_mw": args.p_loss_mw,
                "r_m": args.r_m,
                "kappa": args.kappa,
                "inverse_aspect_ratio": args.inverse_aspect_ratio,
                "m_eff_amu": args.m_eff_amu,
            },
            model_path=args.model_path,
        )
        print(json.dumps(result, indent=2))
        return
    if args.download_if_missing and args.dataset_path is None:
        download_hdb5_std5()
    metadata = train_confinement_model(args.dataset_path, n_splits=args.cv_folds)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    # Import the module under its real name so saved artifacts pickle their
    # classes as ``hdb5.ConfinementArtifact`` rather than ``__main__.*``,
    # keeping the joblib file loadable from any importing process.
    import sys

    import hdb5

    try:
        hdb5.main()
    except hdb5.DatasetIntegrityError as error:
        # The message explains which of the three failure modes this is and what
        # to do about each; a traceback would bury it. Caught only here, so
        # library callers and tests still see the exception itself.
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
