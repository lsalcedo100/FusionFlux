from __future__ import annotations

import shutil
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Union, cast
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline

import config
from artifact_model import FusionFluxModelArtifact
from config import (
    GROUP_COLUMN,
    HIGH_YIELD_PERCENTILE,
    HOLDOUT_TEST_SIZE,
    LOW_LAWSON_RATIO_THRESHOLD,
    MAX_CV_FOLDS,
    MIN_CV_FOLDS,
    MIN_GROUPED_HOLDOUT_GROUPS,
    MIN_TEST_SAMPLES,
    MIN_TOTAL_SAMPLES,
    MIN_TRAIN_SAMPLES,
    ORIGINAL_ROW_INDEX_COLUMN,
    PHYSICS_MISMATCH_FLAG_MODE,
    PREDICTED_YIELD_THRESHOLD,
    RANDOM_STATE,
    RAW_CSV_ROW_NUMBER_COLUMN,
    SUPPORTED_PHYSICS_MISMATCH_FLAG_MODES,
    TARGET_COLUMN,
    TARGET_LOG_COLUMN,
)
from features import (
    DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS,
    align_to_feature_schema,
    build_preprocessing_contract,
    ensure_project_directories,
    get_model_feature_columns,
    prepare_dataset,
)
from inference import (
    ARTIFACT_SCHEMA_VERSION,
    LATEST_TRAINING_RUN_FILENAME,
    TRAINING_METADATA_FILENAME,
    TRAINING_MODEL_FILENAME,
    _current_runtime_versions,
    _serialize_training_metadata_paths,
    _write_latest_training_run_manifest,
)
from storage import ensure_parent_directory, write_dataframe_csv_atomic, write_json_strict

ModelFactory = Callable[[], TransformedTargetRegressor]
MetricValue = Union[float, int]
MetricSummaryValue = Union[MetricValue, str]
MODEL_SELECTION_COLUMNS = ["cv_rmse_mean", "cv_mae_mean", "model"]
TRAINING_RUNS_DIRNAME = "runs"


@dataclass(frozen=True)
class PhysicsMismatchFlagSummary:
    flag_mode: str
    high_yield_threshold: float
    high_yield_threshold_source: str
    high_yield_percentile: float | None
    predicted_yield_threshold: float | None
    low_lawson_ratio_threshold: float

    def to_metadata_dict(self, *, flagged_case_count: int) -> dict[str, object]:
        return {
            "flag_mode": self.flag_mode,
            "high_yield_threshold": self.high_yield_threshold,
            "high_yield_threshold_source": self.high_yield_threshold_source,
            "high_yield_percentile": self.high_yield_percentile,
            "predicted_yield_threshold": self.predicted_yield_threshold,
            "low_lawson_ratio_threshold": self.low_lawson_ratio_threshold,
            "flagged_case_count": int(flagged_case_count),
        }


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_training_run_id() -> str:
    return f"train_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"


def _build_training_artifact_paths() -> dict[str, Path | str]:
    run_id = _build_training_run_id()
    runs_dir = config.get_data_processed_dir() / TRAINING_RUNS_DIRNAME
    run_dir = runs_dir / run_id
    staging_run_dir = runs_dir / ".staging" / run_id
    plots_dir = run_dir / "plots"
    models_dir = run_dir / "models"
    staging_plots_dir = staging_run_dir / "plots"
    staging_models_dir = staging_run_dir / "models"
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "staging_run_dir": staging_run_dir,
        "plots_dir": plots_dir,
        "staging_plots_dir": staging_plots_dir,
        "models_dir": models_dir,
        "staging_models_dir": staging_models_dir,
        "processed_dataset_path": run_dir / "fusion_dataset_processed.csv",
        "staging_processed_dataset_path": staging_run_dir / "fusion_dataset_processed.csv",
        "metrics_path": run_dir / "metrics.csv",
        "staging_metrics_path": staging_run_dir / "metrics.csv",
        "prediction_path": run_dir / "test_predictions.csv",
        "staging_prediction_path": staging_run_dir / "test_predictions.csv",
        "mismatch_path": run_dir / "physics_mismatch_flags.csv",
        "staging_mismatch_path": staging_run_dir / "physics_mismatch_flags.csv",
        "synthetic_dataset_path": run_dir / "synthetic_training_input.csv",
        "staging_synthetic_dataset_path": staging_run_dir / "synthetic_training_input.csv",
        "feature_importance_path": run_dir / "feature_importance.csv",
        "staging_feature_importance_path": staging_run_dir / "feature_importance.csv",
        "importance_plot_path": plots_dir / "feature_importance.png",
        "staging_importance_plot_path": staging_plots_dir / "feature_importance.png",
        "model_path": models_dir / TRAINING_MODEL_FILENAME,
        "staging_model_path": staging_models_dir / TRAINING_MODEL_FILENAME,
        "metadata_path": run_dir / TRAINING_METADATA_FILENAME,
        "staging_metadata_path": staging_run_dir / TRAINING_METADATA_FILENAME,
    }


def _cleanup_staged_training_run(artifact_paths: dict[str, Path | str]) -> None:
    staging_run_dir = cast(Path, artifact_paths["staging_run_dir"])
    if staging_run_dir.exists():
        shutil.rmtree(staging_run_dir, ignore_errors=True)
    for maybe_empty_dir in (staging_run_dir.parent, cast(Path, artifact_paths["run_dir"]).parent):
        try:
            maybe_empty_dir.rmdir()
        except OSError:
            continue


def _publish_staged_training_run(artifact_paths: dict[str, Path | str]) -> None:
    staging_run_dir = cast(Path, artifact_paths["staging_run_dir"])
    run_dir = cast(Path, artifact_paths["run_dir"])
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(f"Training run directory already exists: {run_dir}")
    staging_run_dir.rename(run_dir)
    try:
        staging_run_dir.parent.rmdir()
    except OSError:
        pass


def _clip_negative_predictions(predictions: np.ndarray) -> tuple[np.ndarray, int]:
    clipped_predictions = np.asarray(predictions, dtype=float)
    negative_mask = clipped_predictions < 0
    clipped_count = int(np.count_nonzero(negative_mask))
    if clipped_count == 0:
        return clipped_predictions, 0
    clipped_predictions = clipped_predictions.copy()
    clipped_predictions[negative_mask] = 0.0
    return clipped_predictions, clipped_count


def _group_holdout_total_score(total_rows: int, *, target_test_rows: int) -> tuple[int, int, int]:
    return (
        abs(total_rows - target_test_rows),
        0 if total_rows >= target_test_rows else 1,
        -total_rows,
    )


def _select_group_holdout_positions(
    group_counts: list[int],
    *,
    target_test_rows: int,
    max_test_rows: int,
) -> tuple[int, tuple[int, ...]]:
    reachable_bits = 1
    reachable_mask = (1 << (max_test_rows + 1)) - 1
    parent_totals = array("i", [-1]) * (max_test_rows + 1)
    chosen_positions = array("i", [-1]) * (max_test_rows + 1)

    for position, group_row_count in enumerate(group_counts):
        new_bits = ((reachable_bits << group_row_count) & reachable_mask) & ~reachable_bits
        pending_bits = new_bits
        while pending_bits:
            next_total_bit = pending_bits & -pending_bits
            total_rows = next_total_bit.bit_length() - 1
            parent_totals[total_rows] = total_rows - group_row_count
            chosen_positions[total_rows] = position
            pending_bits ^= next_total_bit
        reachable_bits |= new_bits

    candidate_bits = reachable_bits >> MIN_TEST_SAMPLES
    if not candidate_bits:
        raise ValueError(
            "Grouped holdout could not find a test split with enough rows while keeping groups intact. "
            "Provide more shots before training."
        )

    best_total: int | None = None
    pending_candidate_bits = candidate_bits
    while pending_candidate_bits:
        next_total_bit = pending_candidate_bits & -pending_candidate_bits
        total_rows = next_total_bit.bit_length() - 1 + MIN_TEST_SAMPLES
        if total_rows <= max_test_rows and (
            best_total is None
            or _group_holdout_total_score(total_rows, target_test_rows=target_test_rows)
            < _group_holdout_total_score(best_total, target_test_rows=target_test_rows)
        ):
            best_total = total_rows
        pending_candidate_bits ^= next_total_bit

    if best_total is None:
        raise ValueError(
            "Grouped holdout could not find a test split with enough rows while keeping groups intact. "
            "Provide more shots before training."
        )

    selected_positions: list[int] = []
    total_rows = best_total
    while total_rows > 0:
        position = int(chosen_positions[total_rows])
        if position < 0:
            raise RuntimeError("Failed to reconstruct grouped holdout split.")
        selected_positions.append(position)
        total_rows = int(parent_totals[total_rows])

    selected_positions.reverse()
    return best_total, tuple(selected_positions)


def _select_group_holdout_indices(
    df: pd.DataFrame,
    *,
    target_test_rows: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    group_sizes = df.groupby(GROUP_COLUMN, sort=False).size()
    group_names = list(group_sizes.index)
    group_counts = group_sizes.to_numpy(dtype=int)

    rng = np.random.default_rng(random_state)
    shuffled_order = rng.permutation(len(group_names))
    shuffled_groups = [group_names[index] for index in shuffled_order]
    shuffled_counts = [int(group_counts[index]) for index in shuffled_order]

    max_test_rows = len(df) - MIN_TRAIN_SAMPLES
    _, selected_positions = _select_group_holdout_positions(
        shuffled_counts,
        target_test_rows=target_test_rows,
        max_test_rows=max_test_rows,
    )
    selected_groups = {shuffled_groups[position] for position in selected_positions}
    test_mask = df[GROUP_COLUMN].isin(selected_groups).to_numpy(dtype=bool)
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)
    return train_idx, test_idx


def select_split_indices(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray, str]:
    sample_count = len(df)
    if sample_count < MIN_TOTAL_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TOTAL_SAMPLES} samples to produce a trustworthy holdout; found {sample_count}."
        )

    test_size = max(HOLDOUT_TEST_SIZE, MIN_TEST_SAMPLES / sample_count)
    test_count = int(np.ceil(sample_count * test_size))
    train_count = sample_count - test_count
    if train_count < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TRAIN_SAMPLES} training rows after holdout; got {train_count} from {sample_count} samples."
        )

    unique_groups = df[GROUP_COLUMN].nunique(dropna=True) if GROUP_COLUMN in df.columns else 0
    has_repeated_groups = GROUP_COLUMN in df.columns and 0 < unique_groups < len(df)
    if has_repeated_groups:
        if unique_groups < MIN_GROUPED_HOLDOUT_GROUPS:
            raise ValueError(
                f"Need at least {MIN_GROUPED_HOLDOUT_GROUPS} unique {GROUP_COLUMN} values for grouped holdout; "
                f"found {unique_groups}."
            )
        train_idx, test_idx = _select_group_holdout_indices(
            df,
            target_test_rows=test_count,
            random_state=random_state,
        )
        if len(train_idx) < MIN_TRAIN_SAMPLES or len(test_idx) < MIN_TEST_SAMPLES:
            raise ValueError(
                "Grouped holdout left too few rows for training or evaluation. Provide more shots before training."
            )
        return train_idx, test_idx, "group_row_target_split"

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    return np.asarray(train_idx), np.asarray(test_idx), "random_split"


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                feature_columns,
            )
        ]
    )


def build_model_registry(feature_columns: list[str]) -> dict[str, ModelFactory]:
    return {
        "baseline": lambda: TransformedTargetRegressor(
            regressor=Pipeline([("prep", build_preprocessor(feature_columns)), ("model", DummyRegressor(strategy="median"))]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "random_forest": lambda: TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("prep", build_preprocessor(feature_columns)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=400,
                            max_depth=14,
                            min_samples_leaf=2,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "hist_gradient_boosting": lambda: TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("prep", build_preprocessor(feature_columns)),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_depth=8,
                            learning_rate=0.05,
                            max_iter=350,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }


def compute_metrics(y_true: pd.Series, predictions: np.ndarray, *, context: str) -> dict[str, MetricValue]:
    if len(y_true) == 0:
        raise ValueError(f"{context} targets must contain at least one value.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"{context} predictions must be finite.")

    mse = mean_squared_error(y_true, predictions)
    if not np.isfinite(mse):
        raise ValueError(f"{context} metrics became non-finite; refusing to train on an unstable split.")

    distinct_target_values = int(y_true.nunique(dropna=True))
    r2 = float("nan")
    if len(y_true) >= 2 and distinct_target_values >= 2:
        candidate_r2 = float(r2_score(y_true, predictions, force_finite=False))
        if np.isfinite(candidate_r2):
            r2 = candidate_r2
    metrics: dict[str, MetricValue] = {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
    }

    high_yield_threshold = float(np.quantile(y_true, HIGH_YIELD_PERCENTILE))
    high_yield_mask = y_true >= high_yield_threshold
    if high_yield_mask.any():
        metrics["high_yield_mae"] = float(mean_absolute_error(y_true[high_yield_mask], predictions[high_yield_mask]))
        metrics["high_yield_count"] = int(high_yield_mask.sum())
    else:
        metrics["high_yield_mae"] = float("nan")
        metrics["high_yield_count"] = 0
    return metrics


def extract_feature_importance(
    model: TransformedTargetRegressor,
    feature_columns: list[str],
    *,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    model_name: str,
) -> tuple[pd.DataFrame, str]:
    pipeline = cast(Pipeline, model.regressor_)
    preprocessor: ColumnTransformer = pipeline.named_steps["prep"]
    estimator = pipeline.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        transformed_feature_names = list(preprocessor.get_feature_names_out())
        clean_feature_names = [name.split("__", 1)[-1] for name in transformed_feature_names]
        importance_df = pd.DataFrame(
            {
                "feature": clean_feature_names,
                "importance": np.asarray(estimator.feature_importances_, dtype=float),
                "source_model_name": model_name,
                "importance_method": "intrinsic_feature_importances",
            }
        )
        importance_method = "intrinsic_feature_importances"
    else:
        permutation_result = permutation_importance(
            model,
            X_reference[feature_columns],
            y_reference,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="neg_root_mean_squared_error",
        )
        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": np.asarray(permutation_result.importances_mean, dtype=float),
                "source_model_name": model_name,
                "importance_method": "permutation_importance",
            }
        )
        importance_method = "permutation_importance"

    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df, importance_method


def extract_cross_validated_feature_importance(
    model_factory: ModelFactory,
    feature_columns: list[str],
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
) -> tuple[pd.DataFrame, str]:
    fold_importances: list[pd.DataFrame] = []
    base_importance_method: str | None = None

    for fit_idx, validation_idx in cv_splits:
        model = model_factory()
        model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        fold_importance_df, fold_importance_method = extract_feature_importance(
            model,
            feature_columns,
            X_reference=X_train.iloc[validation_idx],
            y_reference=y_train.iloc[validation_idx],
            model_name=model_name,
        )
        if base_importance_method is None:
            base_importance_method = fold_importance_method
        elif base_importance_method != fold_importance_method:
            raise ValueError("Feature importance method changed across cross-validation folds.")
        fold_importance_lookup = (
            fold_importance_df.loc[:, ["feature", "importance"]]
            .drop_duplicates(subset=["feature"], keep="last")
            .set_index("feature")["importance"]
            .astype(float)
        )
        fold_importances.append(
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance": fold_importance_lookup.reindex(feature_columns, fill_value=0.0).to_numpy(dtype=float),
                    "importance_when_present": fold_importance_lookup.reindex(feature_columns).to_numpy(dtype=float),
                    "present_in_fold": [feature in fold_importance_lookup.index for feature in feature_columns],
                }
            )
        )

    if not fold_importances or base_importance_method is None:
        raise ValueError("Cross-validated feature importance requires at least one fold.")

    importance_df = (
        pd.concat(fold_importances, ignore_index=True)
        .groupby("feature", as_index=False)
        .agg(
            importance=("importance", "mean"),
            importance_mean_when_present=("importance_when_present", "mean"),
            cv_folds_present=("present_in_fold", "sum"),
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["cv_fold_count"] = len(cv_splits)
    importance_df["cv_folds_present"] = importance_df["cv_folds_present"].astype(int)
    importance_df["cv_folds_missing"] = importance_df["cv_fold_count"] - importance_df["cv_folds_present"]
    importance_method = f"cross_validated_{base_importance_method}"
    importance_df["source_model_name"] = model_name
    importance_df["importance_method"] = importance_method
    return (
        importance_df.loc[
            :,
            [
                "feature",
                "importance",
                "importance_mean_when_present",
                "cv_fold_count",
                "cv_folds_present",
                "cv_folds_missing",
                "source_model_name",
                "importance_method",
            ],
        ],
        importance_method,
    )


def save_residual_plots(
    y_true: pd.Series,
    predictions: np.ndarray,
    output_path: Path,
    model_name: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    residuals = y_true - predictions
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, predictions, alpha=0.7, edgecolor="none")
    min_axis = min(float(y_true.min()), float(predictions.min()))
    max_axis = max(float(y_true.max()), float(predictions.max()))
    axes[0].plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="black")
    axes[0].set_title(f"Actual vs Predicted ({model_name})")
    axes[0].set_xlabel("Actual Neutron Yield")
    axes[0].set_ylabel("Predicted Neutron Yield")

    axes[1].scatter(predictions, residuals, alpha=0.7, edgecolor="none")
    axes[1].axhline(0.0, linestyle="--", color="black")
    axes[1].set_title(f"Residuals ({model_name})")
    axes[1].set_xlabel("Predicted Neutron Yield")
    axes[1].set_ylabel("Residual")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_feature_importance_plot(
    importance_df: pd.DataFrame,
    output_path: Path,
    *,
    model_name: str,
    importance_method: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    top_features = importance_df.head(12).iloc[::-1]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features["feature"], top_features["importance"], color="#2f6f9f")
    title_model_name = model_name.replace("_", " ").title()
    if "permutation_importance" in importance_method:
        ax.set_title(f"{title_model_name} Permutation Importance")
    else:
        ax.set_title(f"{title_model_name} Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _resolve_physics_mismatch_flag_summary(predictions: np.ndarray) -> PhysicsMismatchFlagSummary:
    if PHYSICS_MISMATCH_FLAG_MODE not in SUPPORTED_PHYSICS_MISMATCH_FLAG_MODES:
        raise ValueError(
            "config.PHYSICS_MISMATCH_FLAG_MODE must be one of "
            f"{list(SUPPORTED_PHYSICS_MISMATCH_FLAG_MODES)}, got {PHYSICS_MISMATCH_FLAG_MODE!r}."
        )

    if PHYSICS_MISMATCH_FLAG_MODE == "predicted_percentile":
        return PhysicsMismatchFlagSummary(
            flag_mode=PHYSICS_MISMATCH_FLAG_MODE,
            high_yield_threshold=float(np.quantile(predictions, HIGH_YIELD_PERCENTILE)),
            high_yield_threshold_source=f"top_{int(HIGH_YIELD_PERCENTILE * 100)}pct_holdout_predictions",
            high_yield_percentile=HIGH_YIELD_PERCENTILE,
            predicted_yield_threshold=None,
            low_lawson_ratio_threshold=LOW_LAWSON_RATIO_THRESHOLD,
        )

    if PREDICTED_YIELD_THRESHOLD is None:
        raise ValueError(
            "config.PREDICTED_YIELD_THRESHOLD must be set when "
            "config.PHYSICS_MISMATCH_FLAG_MODE='predicted_yield_threshold'."
        )
    return PhysicsMismatchFlagSummary(
        flag_mode=PHYSICS_MISMATCH_FLAG_MODE,
        high_yield_threshold=float(PREDICTED_YIELD_THRESHOLD),
        high_yield_threshold_source="fixed_predicted_yield_threshold",
        high_yield_percentile=None,
        predicted_yield_threshold=float(PREDICTED_YIELD_THRESHOLD),
        low_lawson_ratio_threshold=LOW_LAWSON_RATIO_THRESHOLD,
    )


def flag_physics_mismatches(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
) -> tuple[pd.DataFrame, PhysicsMismatchFlagSummary]:
    summary = _resolve_physics_mismatch_flag_summary(predictions)
    flag_mask = (predictions >= summary.high_yield_threshold) & (
        test_frame["lawson_ratio"].to_numpy() < summary.low_lawson_ratio_threshold
    )
    flagged = test_frame.loc[flag_mask].copy()
    flagged["predicted_neutron_yield"] = predictions[flag_mask]
    flagged["physics_mismatch_flag_mode"] = summary.flag_mode
    flagged["physics_mismatch_high_yield_threshold"] = summary.high_yield_threshold
    flagged["physics_mismatch_high_yield_threshold_source"] = summary.high_yield_threshold_source
    flagged["physics_mismatch_high_yield_percentile"] = (
        summary.high_yield_percentile if summary.high_yield_percentile is not None else np.nan
    )
    flagged["physics_mismatch_predicted_yield_threshold"] = (
        summary.predicted_yield_threshold if summary.predicted_yield_threshold is not None else np.nan
    )
    flagged["physics_mismatch_low_lawson_ratio_threshold"] = summary.low_lawson_ratio_threshold
    write_dataframe_csv_atomic(output_path, flagged, index=False)
    return flagged, summary


def validate_training_frame(df: pd.DataFrame, candidate_feature_columns: list[str]) -> None:
    if not candidate_feature_columns:
        raise ValueError("No model features are available after dataset preparation.")
    if len(df) < MIN_TOTAL_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TOTAL_SAMPLES} samples to produce a trustworthy holdout; found {len(df)}."
        )
    if df[TARGET_COLUMN].nunique(dropna=True) < 2:
        raise ValueError("Training target must contain at least two distinct values.")
    if GROUP_COLUMN in df.columns and df[GROUP_COLUMN].nunique(dropna=True) == 1 and len(df) > 1:
        raise ValueError(f"Need more than one unique {GROUP_COLUMN} value to build a trustworthy holdout.")


def build_cv_splits(
    train_frame: pd.DataFrame,
    split_strategy: str,
    random_state: int = RANDOM_STATE,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, int]:
    if split_strategy == "group_row_target_split":
        group_count = int(train_frame[GROUP_COLUMN].nunique(dropna=True))
        fold_count = min(MAX_CV_FOLDS, group_count)
        if fold_count < MIN_CV_FOLDS:
            raise ValueError(
                f"Need at least {MIN_CV_FOLDS} unique {GROUP_COLUMN} values in the training fold for grouped CV; "
                f"found {group_count}."
            )
        splitter = GroupKFold(n_splits=fold_count)
        splits = list(splitter.split(train_frame, groups=train_frame[GROUP_COLUMN]))
        return splits, "group_k_fold", fold_count

    fold_count = min(MAX_CV_FOLDS, len(train_frame))
    if fold_count < MIN_CV_FOLDS:
        raise ValueError(f"Need at least {MIN_CV_FOLDS} training rows for cross-validation; found {len(train_frame)}.")
    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=random_state)
    return list(splitter.split(train_frame)), "k_fold", fold_count


def cross_validate_model(
    model_factory: ModelFactory,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    fold_rmse: list[float] = []
    fold_mae: list[float] = []
    for fold_index, (fit_idx, validation_idx) in enumerate(cv_splits, start=1):
        model = model_factory()
        model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        predictions, _ = _clip_negative_predictions(model.predict(X_train.iloc[validation_idx]))
        metrics = compute_metrics(y_train.iloc[validation_idx], predictions, context=f"cross-validation fold {fold_index}")
        fold_rmse.append(float(metrics["rmse"]))
        fold_mae.append(float(metrics["mae"]))

    return {
        "cv_rmse_mean": float(np.mean(fold_rmse)),
        "cv_rmse_std": float(np.std(fold_rmse, ddof=0)),
        "cv_mae_mean": float(np.mean(fold_mae)),
        "cv_mae_std": float(np.std(fold_mae, ddof=0)),
    }


def train_models(
    dataset_path: str | Path | None = None,
    *,
    allow_synthetic: bool = False,
    assume_temperature_unit: str | None = None,
    shot_prediction_cutoff_rows: int = DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS,
    generate_reports: bool = True,
) -> dict:
    ensure_project_directories()
    artifact_paths = _build_training_artifact_paths()
    published_run = False
    try:
        prepared = prepare_dataset(
            dataset_path,
            allow_synthetic=allow_synthetic,
            processed_output_path=cast(Path, artifact_paths["staging_processed_dataset_path"]),
            assume_temperature_unit=assume_temperature_unit,
            shot_prediction_cutoff_rows=shot_prediction_cutoff_rows,
            synthetic_output_path=(
                cast(Path, artifact_paths["staging_synthetic_dataset_path"])
                if allow_synthetic and dataset_path is None
                else None
            ),
        )
        df = prepared.dataframe.copy()
        validate_training_frame(df, prepared.candidate_feature_columns)

        train_idx, test_idx, split_strategy = select_split_indices(df)
        holdout_feature_columns = get_model_feature_columns(df.iloc[train_idx])
        if not holdout_feature_columns:
            raise ValueError("No model features are available after selecting the holdout feature schema.")

        saved_feature_columns = get_model_feature_columns(df)
        if not saved_feature_columns:
            raise ValueError("No model features are available for the refit-on-full-data saved model.")
        saved_model_only_feature_columns = [
            column for column in saved_feature_columns if column not in holdout_feature_columns
        ]

        X_holdout = align_to_feature_schema(df, holdout_feature_columns)
        y = df[TARGET_COLUMN]

        X_train, X_test = X_holdout.iloc[train_idx], X_holdout.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        cv_splits, cv_strategy, cv_fold_count = build_cv_splits(df.iloc[train_idx], split_strategy)

        holdout_models = build_model_registry(holdout_feature_columns)
        metrics_summary: list[dict[str, MetricSummaryValue]] = []
        predictions_by_model: dict[str, np.ndarray] = {}

        for model_name, model_factory in holdout_models.items():
            cv_metrics = cross_validate_model(model_factory, X_train, y_train, cv_splits)
            model = model_factory()
            model.fit(X_train, y_train)
            predictions, _ = _clip_negative_predictions(model.predict(X_test))

            model_metrics = compute_metrics(y_test, predictions, context=f"{model_name} holdout")
            metrics_summary.append(
                {
                    "model": model_name,
                    **cv_metrics,
                    "holdout_mae": model_metrics["mae"],
                    "holdout_rmse": model_metrics["rmse"],
                    "holdout_r2": model_metrics["r2"],
                    "holdout_high_yield_mae": model_metrics["high_yield_mae"],
                    "holdout_high_yield_count": model_metrics["high_yield_count"],
                }
            )
            predictions_by_model[model_name] = predictions

        metrics_df = pd.DataFrame(metrics_summary).sort_values(MODEL_SELECTION_COLUMNS).reset_index(drop=True)
        metrics_output_path = cast(Path, artifact_paths["staging_metrics_path"])
        ensure_parent_directory(metrics_output_path)
        write_dataframe_csv_atomic(metrics_output_path, metrics_df, index=False)

        best_model_name = str(metrics_df.iloc[0]["model"])
        best_predictions = predictions_by_model[best_model_name]

        prediction_frame = df.iloc[test_idx].copy()
        if TARGET_LOG_COLUMN in prediction_frame.columns:
            prediction_frame = prediction_frame.drop(columns=[TARGET_LOG_COLUMN])
        prediction_frame = prediction_frame.rename(columns={TARGET_COLUMN: "actual_neutron_yield"})
        prediction_frame["predicted_neutron_yield"] = best_predictions
        prediction_frame["residual"] = y_test.values - best_predictions
        prediction_output_path = cast(Path, artifact_paths["staging_prediction_path"])
        ensure_parent_directory(prediction_output_path)
        write_dataframe_csv_atomic(prediction_output_path, prediction_frame, index=False)

        mismatch_output_path = cast(Path, artifact_paths["staging_mismatch_path"])
        ensure_parent_directory(mismatch_output_path)
        flagged_cases, mismatch_summary = flag_physics_mismatches(
            prediction_frame,
            best_predictions,
            mismatch_output_path,
        )

        residual_plot_path: Path | None = None
        importance_output_path: Path | None = None
        importance_plot_path: Path | None = None
        importance_method: str | None = None
        if generate_reports:
            residual_plot_path = cast(Path, artifact_paths["staging_plots_dir"]) / f"{best_model_name}_residuals.png"
            ensure_parent_directory(residual_plot_path)
            save_residual_plots(y_test, best_predictions, residual_plot_path, best_model_name)

            importance_df, importance_method = extract_cross_validated_feature_importance(
                holdout_models[best_model_name],
                holdout_feature_columns,
                X_train=X_train,
                y_train=y_train,
                cv_splits=cv_splits,
                model_name=best_model_name,
            )
            importance_output_path = cast(Path, artifact_paths["staging_feature_importance_path"])
            ensure_parent_directory(importance_output_path)
            write_dataframe_csv_atomic(importance_output_path, importance_df, index=False)
            importance_plot_path = cast(Path, artifact_paths["staging_importance_plot_path"])
            ensure_parent_directory(importance_plot_path)
            save_feature_importance_plot(
                importance_df,
                importance_plot_path,
                model_name=best_model_name,
                importance_method=cast(str, importance_method),
            )

        production_model = build_model_registry(saved_feature_columns)[best_model_name]()
        production_model.fit(align_to_feature_schema(df, saved_feature_columns), y)
        run_id = cast(str, artifact_paths["run_id"])
        saved_model = FusionFluxModelArtifact(
            production_model,
            schema_version=ARTIFACT_SCHEMA_VERSION,
            training_run_id=run_id,
            feature_columns=saved_feature_columns,
            model_name=best_model_name,
            preprocessing_contract=build_preprocessing_contract(),
        )
        model_output_path = cast(Path, artifact_paths["staging_model_path"])
        ensure_parent_directory(model_output_path)
        joblib.dump(saved_model, model_output_path)

        runtime_versions = _current_runtime_versions()
        final_model_path = cast(Path, artifact_paths["model_path"])
        final_metadata_path = cast(Path, artifact_paths["metadata_path"])
        final_processed_dataset_path = cast(Path, artifact_paths["processed_dataset_path"])
        final_residual_plot_path = (
            cast(Path, artifact_paths["plots_dir"]) / f"{best_model_name}_residuals.png" if generate_reports else None
        )
        final_importance_output_path = (
            cast(Path, artifact_paths["feature_importance_path"]) if generate_reports else None
        )
        final_importance_plot_path = (
            cast(Path, artifact_paths["importance_plot_path"]) if generate_reports else None
        )
        resolved_source_dataset_path = (
            cast(Path, artifact_paths["synthetic_dataset_path"]) if prepared.synthetic_data_used else prepared.raw_path
        )

        metadata = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "training_run_id": run_id,
            "created_at_utc": _timestamp_utc(),
            "raw_dataset_path": str(resolved_source_dataset_path),
            "processed_dataset_path": str(final_processed_dataset_path),
            "artifact_run_directory": str(cast(Path, artifact_paths["run_dir"])),
            "prepared_dataset_candidate_feature_columns": prepared.candidate_feature_columns,
            "feature_columns": saved_feature_columns,
            "holdout_feature_columns": holdout_feature_columns,
            "best_model_name": best_model_name,
            "runtime_versions": runtime_versions,
            "split_strategy": split_strategy,
            "cv_strategy": cv_strategy,
            "cv_fold_count": cv_fold_count,
            "train_row_count": int(len(train_idx)),
            "test_row_count": int(len(test_idx)),
            "full_data_row_count": int(len(df)),
            "audit_summary": prepared.audit_summary,
            "column_mapping": prepared.column_mapping,
            "preprocessing": build_preprocessing_contract(),
            "dataset_source": {
                "kind": prepared.dataset_source_kind,
                "synthetic_data_used": prepared.synthetic_data_used,
                "synthetic_generation": (
                    {
                        "random_state": prepared.synthetic_random_state,
                        "row_count": prepared.synthetic_row_count,
                    }
                    if prepared.synthetic_data_used
                    else None
                ),
                "requested_dataset_path": (
                    str(prepared.requested_dataset_path) if prepared.requested_dataset_path is not None else None
                ),
                "resolved_dataset_path": str(resolved_source_dataset_path),
            },
            "dataset_preparation": {
                "assume_temperature_unit": assume_temperature_unit,
                "shot_prediction_cutoff_rows": shot_prediction_cutoff_rows,
                "row_identity_columns": [ORIGINAL_ROW_INDEX_COLUMN, RAW_CSV_ROW_NUMBER_COLUMN],
            },
            "feature_schema": {
                "holdout_selection_basis": "training_split_only",
                "holdout_feature_columns": holdout_feature_columns,
                "saved_model_feature_columns": saved_feature_columns,
                "saved_model_schema_source": "full_prepared_dataset",
                "saved_model_only_feature_columns": saved_model_only_feature_columns,
            },
            "model_selection": {
                "basis": "cross_validation",
                "primary_metric": "cv_rmse_mean",
                "tie_breakers": ["cv_mae_mean", "model"],
                "selected_model_name": best_model_name,
                "candidate_models": metrics_df["model"].astype(str).tolist(),
            },
            "holdout_evaluation": {
                "split_strategy": split_strategy,
                "cv_strategy": cv_strategy,
                "cv_fold_count": cv_fold_count,
                "train_row_count": int(len(train_idx)),
                "test_row_count": int(len(test_idx)),
                "feature_columns": holdout_feature_columns,
                "metrics_artifact_path": str(cast(Path, artifact_paths["metrics_path"])),
                "prediction_artifact_path": str(cast(Path, artifact_paths["prediction_path"])),
                "mismatch_artifact_path": str(cast(Path, artifact_paths["mismatch_path"])),
                "residual_plot_path": str(final_residual_plot_path) if final_residual_plot_path is not None else None,
                "selected_model_fit_scope": "training_split_only",
                "report_generation_enabled": generate_reports,
                "physics_mismatch_flagging": mismatch_summary.to_metadata_dict(flagged_case_count=len(flagged_cases)),
            },
            "model_explainability": {
                "enabled": generate_reports,
                "source_model_name": best_model_name,
                "feature_columns": holdout_feature_columns if generate_reports else None,
                "importance_method": importance_method,
                "fit_scope": "cross_validation_training_folds" if generate_reports else None,
                "artifact_scope": "selected_model_family_cv_folds" if generate_reports else None,
            },
            "saved_model": {
                "path": str(final_model_path),
                "artifact_type": "FusionFluxModelArtifact",
                "model_name": best_model_name,
                "feature_columns": saved_feature_columns,
                "feature_schema_source": "full_prepared_dataset",
                "fit_scope": "full_prepared_dataset",
                "row_count": int(len(df)),
                "training_run_id": run_id,
            },
            "artifacts": {
                "metrics_path": str(cast(Path, artifact_paths["metrics_path"])),
                "prediction_path": str(cast(Path, artifact_paths["prediction_path"])),
                "mismatch_path": str(cast(Path, artifact_paths["mismatch_path"])),
                "feature_importance_path": (
                    str(final_importance_output_path) if final_importance_output_path is not None else None
                ),
                "residual_plot_path": str(final_residual_plot_path) if final_residual_plot_path is not None else None,
                "importance_plot_path": (
                    str(final_importance_plot_path) if final_importance_plot_path is not None else None
                ),
            },
        }
        write_json_strict(
            cast(Path, artifact_paths["staging_metadata_path"]),
            _serialize_training_metadata_paths(metadata, metadata_path=final_metadata_path),
        )
        _publish_staged_training_run(artifact_paths)
        published_run = True
        _write_latest_training_run_manifest(run_id=run_id, model_path=final_model_path, metadata_path=final_metadata_path)

        return {
            "metrics_path": str(cast(Path, artifact_paths["metrics_path"])),
            "prediction_path": str(cast(Path, artifact_paths["prediction_path"])),
            "mismatch_path": str(cast(Path, artifact_paths["mismatch_path"])),
            "feature_importance_path": str(final_importance_output_path) if final_importance_output_path is not None else None,
            "residual_plot_path": str(final_residual_plot_path) if final_residual_plot_path is not None else None,
            "importance_plot_path": str(final_importance_plot_path) if final_importance_plot_path is not None else None,
            "model_path": str(final_model_path),
            "metadata_path": str(final_metadata_path),
            "latest_manifest_path": str(config.get_data_processed_dir() / LATEST_TRAINING_RUN_FILENAME),
            "best_model_name": best_model_name,
            "flagged_case_count": int(len(flagged_cases)),
            "dataset_source_kind": prepared.dataset_source_kind,
            "synthetic_data_used": prepared.synthetic_data_used,
            "saved_model_fit_scope": "full_prepared_dataset",
            "report_generation_enabled": generate_reports,
        }
    except Exception:
        if not published_run:
            _cleanup_staged_training_run(artifact_paths)
        raise


__all__ = [
    "HIGH_YIELD_PERCENTILE",
    "LOW_LAWSON_RATIO_THRESHOLD",
    "MODEL_SELECTION_COLUMNS",
    "MetricSummaryValue",
    "MetricValue",
    "ModelFactory",
    "PHYSICS_MISMATCH_FLAG_MODE",
    "TRAINING_RUNS_DIRNAME",
    "PhysicsMismatchFlagSummary",
    "build_cv_splits",
    "build_model_registry",
    "build_preprocessor",
    "compute_metrics",
    "cross_validate_model",
    "extract_cross_validated_feature_importance",
    "extract_feature_importance",
    "flag_physics_mismatches",
    "save_feature_importance_plot",
    "save_residual_plots",
    "select_split_indices",
    "train_models",
    "validate_training_frame",
]
