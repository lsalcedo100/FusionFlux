from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    DATA_PROCESSED_DIR,
    GROUP_COLUMN,
    HOLDOUT_TEST_SIZE,
    HIGH_YIELD_PERCENTILE,
    LOW_LAWSON_RATIO_THRESHOLD,
    MAX_CV_FOLDS,
    MIN_CV_FOLDS,
    MIN_GROUPED_HOLDOUT_GROUPS,
    MIN_TEST_SAMPLES,
    MIN_TOTAL_SAMPLES,
    MIN_TRAIN_SAMPLES,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from features import OPTIONAL_PHYSICS_COLUMNS, engineer_features, ensure_project_directories, prepare_dataset
from lawson import calculate_lawson_status, to_kev
from validation import validate_physics_inputs

ModelFactory = Callable[[], TransformedTargetRegressor]
MetricValue = Union[float, int]
MetricSummaryValue = Union[MetricValue, str]
MODEL_SELECTION_COLUMNS = ["cv_rmse_mean", "cv_mae_mean", "model"]


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
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, groups=df[GROUP_COLUMN]))
        if len(train_idx) < MIN_TRAIN_SAMPLES or len(test_idx) < MIN_TEST_SAMPLES:
            raise ValueError(
                "Grouped holdout left too few rows for training or evaluation. Provide more shots before training."
            )
        return train_idx, test_idx, "group_shuffle_split"

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
    if len(y_true) < 2 or y_true.nunique(dropna=True) < 2:
        raise ValueError(f"{context} targets must contain at least two distinct values.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"{context} predictions must be finite.")

    mse = mean_squared_error(y_true, predictions)
    r2 = float(r2_score(y_true, predictions, force_finite=False))
    if not np.isfinite(mse) or not np.isfinite(r2):
        raise ValueError(f"{context} metrics became non-finite; refusing to train on an unstable split.")
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
    model: TransformedTargetRegressor, feature_columns: list[str]
) -> pd.DataFrame:
    pipeline = model.regressor_
    preprocessor: ColumnTransformer = pipeline.named_steps["prep"]
    estimator = pipeline.named_steps["model"]

    transformed_feature_names = list(preprocessor.get_feature_names_out())
    clean_feature_names = [name.split("__", 1)[-1] for name in transformed_feature_names]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    else:
        importances = np.zeros(len(clean_feature_names))

    importance_df = pd.DataFrame(
        {"feature": clean_feature_names, "importance": np.asarray(importances, dtype=float)}
    ).sort_values("importance", ascending=False)
    return importance_df


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


def save_feature_importance_plot(importance_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    top_features = importance_df.head(12).iloc[::-1]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features["feature"], top_features["importance"], color="#2f6f9f")
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def flag_physics_mismatches(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    high_yield_threshold = float(np.quantile(predictions, HIGH_YIELD_PERCENTILE))
    flag_mask = (predictions >= high_yield_threshold) & (
        test_frame["lawson_ratio"].to_numpy() < LOW_LAWSON_RATIO_THRESHOLD
    )
    flagged = test_frame.loc[flag_mask].copy()
    flagged["predicted_neutron_yield"] = predictions[flag_mask]
    flagged.to_csv(output_path, index=False)
    return flagged


def validate_training_frame(df: pd.DataFrame, feature_columns: list[str]) -> None:
    if not feature_columns:
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
    if split_strategy == "group_shuffle_split":
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
        predictions = model.predict(X_train.iloc[validation_idx])
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
) -> dict:
    ensure_project_directories()
    prepared = prepare_dataset(dataset_path, allow_synthetic=allow_synthetic)
    df = prepared.dataframe.copy()
    feature_columns = prepared.feature_columns
    validate_training_frame(df, feature_columns)

    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    train_idx, test_idx, split_strategy = select_split_indices(df)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    cv_splits, cv_strategy, cv_fold_count = build_cv_splits(df.iloc[train_idx], split_strategy)

    models = build_model_registry(feature_columns)
    metrics_summary: list[dict[str, MetricSummaryValue]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    fitted_models: dict[str, TransformedTargetRegressor] = {}

    for model_name, model_factory in models.items():
        cv_metrics = cross_validate_model(model_factory, X_train, y_train, cv_splits)
        model = model_factory()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

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
        fitted_models[model_name] = model

    metrics_df = pd.DataFrame(metrics_summary).sort_values(MODEL_SELECTION_COLUMNS).reset_index(drop=True)
    metrics_output_path = DATA_PROCESSED_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_output_path, index=False)

    best_model_name = str(metrics_df.iloc[0]["model"])
    best_predictions = predictions_by_model[best_model_name]

    prediction_frame = X_test.copy()
    prediction_frame["actual_neutron_yield"] = y_test.values
    prediction_frame["predicted_neutron_yield"] = best_predictions
    prediction_frame["residual"] = y_test.values - best_predictions
    prediction_output_path = DATA_PROCESSED_DIR / "test_predictions.csv"
    prediction_frame.to_csv(prediction_output_path, index=False)

    mismatch_output_path = DATA_PROCESSED_DIR / "physics_mismatch_flags.csv"
    flagged_cases = flag_physics_mismatches(prediction_frame, best_predictions, mismatch_output_path)

    residual_plot_path = PLOTS_DIR / f"{best_model_name}_residuals.png"
    save_residual_plots(y_test, best_predictions, residual_plot_path, best_model_name)

    importance_df = extract_feature_importance(fitted_models["random_forest"], feature_columns)
    importance_output_path = DATA_PROCESSED_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_output_path, index=False)
    importance_plot_path = PLOTS_DIR / "random_forest_feature_importance.png"
    save_feature_importance_plot(importance_df, importance_plot_path)

    production_model = models[best_model_name]()
    production_model.fit(X, y)
    model_output_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(production_model, model_output_path)

    metadata = {
        "raw_dataset_path": str(prepared.raw_path),
        "processed_dataset_path": str(prepared.processed_path),
        "feature_columns": feature_columns,
        "best_model_name": best_model_name,
        "split_strategy": split_strategy,
        "cv_strategy": cv_strategy,
        "cv_fold_count": cv_fold_count,
        "train_row_count": int(len(train_idx)),
        "test_row_count": int(len(test_idx)),
        "full_data_row_count": int(len(df)),
        "audit_summary": prepared.audit_summary,
        "column_mapping": prepared.column_mapping,
        "dataset_source": {
            "kind": prepared.dataset_source_kind,
            "synthetic_data_used": prepared.synthetic_data_used,
            "requested_dataset_path": (
                str(prepared.requested_dataset_path) if prepared.requested_dataset_path is not None else None
            ),
            "resolved_dataset_path": str(prepared.raw_path),
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
            "metrics_artifact_path": str(metrics_output_path),
            "prediction_artifact_path": str(prediction_output_path),
            "mismatch_artifact_path": str(mismatch_output_path),
            "residual_plot_path": str(residual_plot_path),
            "selected_model_fit_scope": "training_split_only",
        },
        "saved_model": {
            "path": str(model_output_path),
            "model_name": best_model_name,
            "fit_scope": "full_prepared_dataset",
            "row_count": int(len(df)),
        },
        "feature_defaults": {
            column: float(df[column].median()) if column in df.columns and pd.api.types.is_numeric_dtype(df[column]) else None
            for column in feature_columns
        },
    }
    metadata_path = DATA_PROCESSED_DIR / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "metrics_path": str(metrics_output_path),
        "prediction_path": str(prediction_output_path),
        "mismatch_path": str(mismatch_output_path),
        "feature_importance_path": str(importance_output_path),
        "residual_plot_path": str(residual_plot_path),
        "importance_plot_path": str(importance_plot_path),
        "model_path": str(model_output_path),
        "metadata_path": str(metadata_path),
        "best_model_name": best_model_name,
        "flagged_case_count": int(len(flagged_cases)),
        "dataset_source_kind": prepared.dataset_source_kind,
        "synthetic_data_used": prepared.synthetic_data_used,
        "saved_model_fit_scope": "full_prepared_dataset",
    }


def predict_single_case(
    density_m3: float,
    temperature: float,
    confinement_time_s: float,
    temp_unit: str,
    fuel_purity: float | None,
    energy_input_mj: float | None,
    pressure_pa: float | None,
    ip_ma: float | None,
    bt_t: float | None,
    r_m: float | None,
    a_m: float | None,
    kappa: float | None,
    ne_20: float | None,
    m_amu: float | None,
    pin_mw: float | None,
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict:
    ensure_project_directories()
    validated_inputs = validate_physics_inputs(
        {
            "fuel_density_m3": density_m3,
            "temperature_keV": to_kev(temperature, temp_unit),
            "confinement_time_s": confinement_time_s,
            "fuel_purity": fuel_purity,
            "energy_input_MJ": energy_input_mj,
            "pressure_Pa": pressure_pa,
            "Ip_MA": ip_ma,
            "Bt_T": bt_t,
            "R_m": r_m,
            "a_m": a_m,
            "kappa": kappa,
            "ne_20": ne_20,
            "M_amu": m_amu,
            "Pin_MW": pin_mw,
        },
        required_fields=("fuel_density_m3", "temperature_keV", "confinement_time_s"),
        optional_fields=OPTIONAL_PHYSICS_COLUMNS,
    )
    resolved_model_path = Path(model_path).expanduser().resolve() if model_path else MODELS_DIR / "best_model.joblib"
    resolved_metadata_path = (
        Path(metadata_path).expanduser().resolve() if metadata_path else DATA_PROCESSED_DIR / "training_metadata.json"
    )

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model not found: {resolved_model_path}")
    if not resolved_metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {resolved_metadata_path}")

    model = joblib.load(resolved_model_path)
    metadata = json.loads(resolved_metadata_path.read_text())

    feature_defaults = metadata["feature_defaults"]
    row = {column: value for column, value in feature_defaults.items()}
    row["fuel_density_m3"] = validated_inputs["fuel_density_m3"]
    row["temperature_keV"] = validated_inputs["temperature_keV"]
    row["confinement_time_s"] = validated_inputs["confinement_time_s"]

    optional_updates = {
        "fuel_purity": validated_inputs["fuel_purity"],
        "energy_input_MJ": validated_inputs["energy_input_MJ"],
        "pressure_Pa": validated_inputs["pressure_Pa"],
        "Ip_MA": validated_inputs["Ip_MA"],
        "Bt_T": validated_inputs["Bt_T"],
        "R_m": validated_inputs["R_m"],
        "a_m": validated_inputs["a_m"],
        "kappa": validated_inputs["kappa"],
        "ne_20": validated_inputs["ne_20"],
        "M_amu": validated_inputs["M_amu"],
        "Pin_MW": validated_inputs["Pin_MW"],
    }
    for column, value in optional_updates.items():
        if value is not None:
            row[column] = value

    inference_df = pd.DataFrame([row])
    inference_df = engineer_features(inference_df)

    feature_columns = metadata["feature_columns"]
    predicted_yield = float(model.predict(inference_df[feature_columns])[0])
    lawson_result = calculate_lawson_status(
        density_m3=density_m3,
        temperature=temperature,
        confinement_time_s=confinement_time_s,
        temp_unit=temp_unit,
    )

    return {
        "predicted_neutron_yield": predicted_yield,
        "triple_product": lawson_result.triple_product,
        "lawson_ratio": lawson_result.lawson_ratio,
        "status": lawson_result.status,
        "model_name": metadata["best_model_name"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run the fusion predictor.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the ML pipeline.")
    train_parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the input CSV dataset. Required unless --allow-synthetic is set.",
    )
    train_parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Generate and train on synthetic demo data when --dataset-path is omitted.",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict a single fusion operating point.")
    predict_parser.add_argument("--density-m3", type=float, required=True)
    predict_parser.add_argument("--temperature", type=float, required=True)
    predict_parser.add_argument("--temp-unit", type=str, default="keV", choices=["keV", "eV", "K"])
    predict_parser.add_argument("--confinement-time-s", type=float, required=True)
    predict_parser.add_argument("--fuel-purity", type=float, default=None)
    predict_parser.add_argument("--energy-input-mj", type=float, default=None)
    predict_parser.add_argument("--pressure-pa", type=float, default=None)
    predict_parser.add_argument("--ip-ma", type=float, default=None)
    predict_parser.add_argument("--bt-t", type=float, default=None)
    predict_parser.add_argument("--r-m", type=float, default=None)
    predict_parser.add_argument("--a-m", type=float, default=None)
    predict_parser.add_argument("--kappa", type=float, default=None)
    predict_parser.add_argument("--ne-20", type=float, default=None)
    predict_parser.add_argument("--m-amu", type=float, default=None)
    predict_parser.add_argument("--pin-mw", type=float, default=None)
    predict_parser.add_argument("--model-path", type=str, default=None)
    predict_parser.add_argument("--metadata-path", type=str, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        artifacts = train_models(dataset_path=args.dataset_path, allow_synthetic=args.allow_synthetic)
        print(json.dumps(artifacts, indent=2))
        return

    if args.command == "predict":
        prediction = predict_single_case(
            density_m3=args.density_m3,
            temperature=args.temperature,
            confinement_time_s=args.confinement_time_s,
            temp_unit=args.temp_unit,
            fuel_purity=args.fuel_purity,
            energy_input_mj=args.energy_input_mj,
            pressure_pa=args.pressure_pa,
            ip_ma=args.ip_ma,
            bt_t=args.bt_t,
            r_m=args.r_m,
            a_m=args.a_m,
            kappa=args.kappa,
            ne_20=args.ne_20,
            m_amu=args.m_amu,
            pin_mw=args.pin_mw,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
        )
        print(json.dumps(prediction, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
