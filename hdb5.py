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
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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


def download_hdb5_std5(destination: Path | None = None, *, overwrite: bool = False) -> Path:
    """Download the ITPA HDB5 STD5 dataset from OSF into the raw data directory.

    The database is third-party scientific data (please cite Verdoolaege et al.,
    Nucl. Fusion 61 076006, 2021); it is fetched on demand rather than
    redistributed in the repository.
    """
    import urllib.request

    from storage import atomic_output_path

    target = Path(destination).expanduser().resolve() if destination else default_hdb5_path()
    if target.exists() and not overwrite:
        return target
    with atomic_output_path(target) as temp_path:
        with urllib.request.urlopen(HDB5_STD5_DOWNLOAD_URL) as response:
            temp_path.write_bytes(response.read())
    return target


def load_hdb5_dataframe(path: Path | str | None = None) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve() if path is not None else default_hdb5_path()
    if not resolved.exists():
        raise FileNotFoundError(
            f"HDB5 dataset not found: {resolved}. Download the STD5 set from "
            f"{HDB5_OSF_PROJECT} ({HDB5_STD5_DOWNLOAD_URL}) and save it there, "
            "or pass --dataset-path."
        )
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


def prepare_dataset(path: Path | str | None = None) -> pd.DataFrame:
    return build_features(map_to_canonical(load_hdb5_dataframe(path)))


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
            fold_model.fit(features.iloc[train_idx], log_target[train_idx])
            predictions[test_idx] = fold_model.predict(features.iloc[test_idx])
    return predictions


def clone_pipeline(estimator: Pipeline) -> Pipeline:
    from sklearn.base import clone

    return clone(estimator)


def evaluate_models(dataset: pd.DataFrame, *, n_splits: int = N_CV_FOLDS) -> list[ModelScore]:
    features = dataset[list(MODEL_FEATURE_COLUMNS)]
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

    for name, estimator in build_model_zoo().items():
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
        final_model.fit(features, log_tau)

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
    """Predict confinement time for one operating point using a saved artifact."""
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

    download = subparsers.add_parser("download", help="Fetch the HDB5 STD5 dataset from OSF.")
    download.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if the dataset already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        path = download_hdb5_std5(overwrite=args.overwrite)
        print(json.dumps({"downloaded_to": str(path)}, indent=2))
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
    import hdb5

    hdb5.main()
