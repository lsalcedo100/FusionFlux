from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from helpers import (
    _write_dataset,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

import features
import train_model
import training
from config import ORIGINAL_ROW_INDEX_COLUMN, RAW_CSV_ROW_NUMBER_COLUMN


def test_train_models_rejects_tiny_dataset(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20],
            "temperature_keV": [12.0],
            "confinement_time_s": [1.0],
            "neutron_yield": [100.0],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "tiny.csv")

    with pytest.raises(ValueError, match="trustworthy holdout"):
        train_model.train_models(dataset_path)


def test_train_models_rejects_single_group_dataset(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "shot_id": [17] * 20,
            "fuel_density_m3": np.linspace(0.9e20, 1.3e20, 20),
            "temperature_keV": np.linspace(9.0, 18.0, 20),
            "confinement_time_s": np.linspace(0.8, 2.2, 20),
            "neutron_yield": np.linspace(50.0, 250.0, 20),
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "single_group.csv")

    with pytest.raises(ValueError, match="shot_id"):
        train_model.train_models(dataset_path)


def test_train_models_requires_explicit_dataset_source(
    isolated_project_dirs: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="--dataset-path.*--allow-synthetic"):
        train_model.train_models()


def test_train_models_does_not_create_run_dir_on_early_validation_failure(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20],
            "temperature_keV": [12.0, 13.0],
            "confinement_time_s": [1.0, 1.1],
            "neutron_yield": [100.0, 110.0],
            "pressure_Pa": ["bad", "still_bad"],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "early_failure.csv")

    with pytest.raises(ValueError, match="pressure_Pa.*numeric when provided"):
        train_model.train_models(dataset_path)

    runs_dir = isolated_project_dirs["processed"] / train_model.TRAINING_RUNS_DIRNAME
    assert not runs_dir.exists()


def test_train_models_cleans_up_staged_run_on_late_failure(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = features.create_synthetic_dataset(tmp_path / "late_failure.csv", n_rows=60, random_state=13)

    def fail_dump(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("late save failed")

    monkeypatch.setattr(train_model.joblib, "dump", fail_dump)

    with pytest.raises(RuntimeError, match="late save failed"):
        train_model.train_models(dataset_path)

    runs_dir = isolated_project_dirs["processed"] / train_model.TRAINING_RUNS_DIRNAME
    latest_manifest_path = isolated_project_dirs["processed"] / train_model.LATEST_TRAINING_RUN_FILENAME

    assert not latest_manifest_path.exists()
    assert list(isolated_project_dirs["processed"].rglob("fusion_dataset_processed.csv")) == []
    assert not runs_dir.exists()


def test_train_models_allow_synthetic_records_source_metadata(
    isolated_project_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_create_synthetic_dataset = features.create_synthetic_dataset

    def create_small_synthetic_dataset(output_path: Path | None = None, n_rows: int = 600, random_state: int = 42) -> Path:
        return original_create_synthetic_dataset(output_path=output_path, n_rows=60, random_state=random_state)

    monkeypatch.setattr(features, "create_synthetic_dataset", create_small_synthetic_dataset)

    artifacts = train_model.train_models(allow_synthetic=True)
    metadata = json.loads(Path(artifacts["metadata_path"]).read_text())

    assert artifacts["dataset_source_kind"] == "synthetic_generated"
    assert artifacts["synthetic_data_used"] is True
    assert metadata["dataset_source"]["kind"] == "synthetic_generated"
    assert metadata["dataset_source"]["synthetic_data_used"] is True
    assert metadata["dataset_source"]["synthetic_generation"] == {"random_state": 42, "row_count": 60}
    assert metadata["schema_version"] == train_model.ARTIFACT_SCHEMA_VERSION
    assert metadata["preprocessing"]["sha256"] == features.build_preprocessing_contract()["sha256"]
    assert metadata["preprocessing"]["contract_version"] == features.PREPROCESSING_CONTRACT_VERSION
    # Source/bytecode fingerprints were removed; the contract is purely structural.
    assert "source_sha256" not in metadata["preprocessing"]
    assert "logic_fingerprints" not in metadata["preprocessing"]
    assert metadata["runtime_versions"]["python"]
    assert metadata["runtime_versions"]["pandas"]
    assert metadata["runtime_versions"]["scikit_learn"]
    assert metadata["runtime_versions"]["joblib"]
    assert metadata["saved_model"]["fit_scope"] == "full_prepared_dataset"
    assert "/runs/" in artifacts["model_path"]
    assert "/runs/" in artifacts["metadata_path"]
    assert "NaN" not in Path(artifacts["metadata_path"]).read_text()
    assert metadata["artifact_run_directory"] == "."
    assert metadata["saved_model"]["path"] == "models/best_model.joblib"
    assert metadata["dataset_source"]["resolved_dataset_path"] == "synthetic_training_input.csv"
    assert not (isolated_project_dirs["raw"] / "synthetic_nuclear_fusion_experiment.csv").exists()
    assert (Path(artifacts["metadata_path"]).parent / "synthetic_training_input.csv").exists()
    latest_manifest = json.loads((isolated_project_dirs["processed"] / train_model.LATEST_TRAINING_RUN_FILENAME).read_text())
    assert not Path(latest_manifest["model_path"]).is_absolute()
    assert not Path(latest_manifest["metadata_path"]).is_absolute()
    resolved_model_path, resolved_metadata_path = train_model._resolve_prediction_artifact_paths(None, None)
    assert resolved_model_path == Path(artifacts["model_path"]).resolve()
    assert resolved_metadata_path == Path(artifacts["metadata_path"]).resolve()


def test_train_models_refits_selected_model_on_full_dataset_before_saving(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": np.linspace(1.0e20, 2.4e20, 15),
            "temperature_keV": np.linspace(8.0, 22.0, 15),
            "confinement_time_s": np.linspace(0.8, 2.2, 15),
            "neutron_yield": np.arange(1.0, 16.0),
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "refit_training.csv")

    def build_dummy_registry(feature_columns: list[str]) -> dict[str, train_model.ModelFactory]:
        def make_dummy_model() -> TransformedTargetRegressor:
            return TransformedTargetRegressor(
                regressor=Pipeline(
                    [
                        ("prep", train_model.build_preprocessor(feature_columns)),
                        ("model", DummyRegressor(strategy="median")),
                    ]
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            )

        return {
            "baseline": make_dummy_model,
            "random_forest": make_dummy_model,
            "hist_gradient_boosting": make_dummy_model,
        }

    monkeypatch.setattr(training, "build_model_registry", build_dummy_registry)
    monkeypatch.setattr(
        training,
        "select_split_indices",
        lambda df, random_state=train_model.RANDOM_STATE: (np.arange(12), np.arange(12, 15), "random_split"),
    )

    artifacts = train_model.train_models(dataset_path)
    saved_model = joblib.load(artifacts["model_path"])
    metadata = json.loads(Path(artifacts["metadata_path"]).read_text())
    prepared = features.prepare_dataset(dataset_path)

    inference_row = prepared.dataframe[metadata["feature_columns"]].iloc[[0]]
    saved_prediction = float(saved_model.predict(inference_row)[0])
    expected_full_prediction = float(np.expm1(np.median(np.log1p(frame["neutron_yield"].to_numpy(dtype=float)))))
    training_only_prediction = float(
        np.expm1(np.median(np.log1p(frame.iloc[:12]["neutron_yield"].to_numpy(dtype=float))))
    )

    assert saved_prediction == pytest.approx(expected_full_prediction)
    assert saved_prediction != pytest.approx(training_only_prediction)
    assert metadata["model_selection"]["basis"] == "cross_validation"
    assert metadata["model_selection"]["primary_metric"] == "cv_rmse_log_mean"
    assert metadata["saved_model"]["fit_scope"] == "full_prepared_dataset"
    assert metadata["saved_model"]["row_count"] == len(frame)
    assert metadata["holdout_evaluation"]["selected_model_fit_scope"] == "training_split_only"
    assert metadata["model_explainability"]["fit_scope"] == "cross_validation_training_folds"
    assert metadata["model_explainability"]["artifact_scope"] == "selected_model_family_cv_folds"


def test_train_models_persists_full_dataset_feature_schema_and_keeps_holdout_explainability_scoped_to_training(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": np.linspace(1.0e20, 2.4e20, 15),
            "temperature_keV": np.linspace(8.0, 22.0, 15),
            "confinement_time_s": np.linspace(0.8, 2.2, 15),
            "neutron_yield": np.linspace(100.0, 240.0, 15),
            "pressure_Pa": [np.nan] * 12 + [1.0e5, 1.1e5, 1.2e5],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "training_only_feature_schema.csv")

    def build_dummy_registry(feature_columns: list[str]) -> dict[str, train_model.ModelFactory]:
        def make_dummy_model() -> TransformedTargetRegressor:
            return TransformedTargetRegressor(
                regressor=Pipeline(
                    [
                        ("prep", train_model.build_preprocessor(feature_columns)),
                        ("model", DummyRegressor(strategy="median")),
                    ]
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            )

        return {
            "baseline": make_dummy_model,
            "random_forest": make_dummy_model,
            "hist_gradient_boosting": make_dummy_model,
        }

    captured_reference_indices: list[int] = []
    original_extract_feature_importance = train_model.extract_feature_importance

    def capture_extract_feature_importance(
        model: TransformedTargetRegressor,
        feature_columns: list[str],
        *,
        X_reference: pd.DataFrame,
        y_reference: pd.Series,
        model_name: str,
    ) -> tuple[pd.DataFrame, str]:
        captured_reference_indices.extend(int(index) for index in X_reference.index.tolist())
        assert set(X_reference.index).issubset(set(range(12)))
        return original_extract_feature_importance(
            model,
            feature_columns,
            X_reference=X_reference,
            y_reference=y_reference,
            model_name=model_name,
        )

    monkeypatch.setattr(training, "build_model_registry", build_dummy_registry)
    monkeypatch.setattr(
        training,
        "select_split_indices",
        lambda df, random_state=train_model.RANDOM_STATE: (np.arange(12), np.arange(12, 15), "random_split"),
    )
    monkeypatch.setattr(training, "extract_feature_importance", capture_extract_feature_importance)

    artifacts = train_model.train_models(dataset_path)
    metadata = json.loads(Path(artifacts["metadata_path"]).read_text())

    assert "pressure_Pa" in metadata["prepared_dataset_candidate_feature_columns"]
    assert "log_pressure_Pa" in metadata["prepared_dataset_candidate_feature_columns"]
    assert "pressure_Pa" in metadata["feature_columns"]
    assert "log_pressure_Pa" in metadata["feature_columns"]
    assert "pressure_Pa" not in metadata["holdout_feature_columns"]
    assert "log_pressure_Pa" not in metadata["holdout_feature_columns"]
    assert metadata["feature_schema"]["saved_model_schema_source"] == "full_prepared_dataset"
    assert metadata["feature_schema"]["saved_model_only_feature_columns"] == ["pressure_Pa", "log_pressure_Pa"]
    assert Path(artifacts["feature_importance_path"]).exists()
    assert captured_reference_indices
    assert set(captured_reference_indices).isdisjoint({12, 13, 14})
    assert metadata["model_explainability"]["fit_scope"] == "cross_validation_training_folds"
    assert metadata["model_explainability"]["artifact_scope"] == "selected_model_family_cv_folds"


def test_extract_cross_validated_feature_importance_zero_fills_features_missing_in_some_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoOpModel:
        def fit(self, _X: pd.DataFrame, _y: pd.Series) -> "NoOpModel":
            return self

    feature_columns = ["fuel_density_m3", "pressure_Pa", "temperature_keV"]
    X_train = pd.DataFrame(
        {
            "fuel_density_m3": np.linspace(1.0e20, 1.5e20, 6),
            "pressure_Pa": [np.nan, np.nan, 1.0e5, 1.1e5, 1.2e5, 1.3e5],
            "temperature_keV": np.linspace(10.0, 15.0, 6),
        }
    )
    y_train = pd.Series(np.linspace(100.0, 160.0, 6))
    cv_splits = [
        (np.array([2, 3, 4, 5]), np.array([0, 1])),
        (np.array([0, 1, 4, 5]), np.array([2, 3])),
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
    ]
    fold_outputs = iter(
        [
            pd.DataFrame(
                {
                    "feature": ["fuel_density_m3", "temperature_keV"],
                    "importance": [0.9, 0.1],
                    "source_model_name": ["random_forest", "random_forest"],
                    "importance_method": ["intrinsic_feature_importances", "intrinsic_feature_importances"],
                }
            ),
            pd.DataFrame(
                {
                    "feature": ["fuel_density_m3", "pressure_Pa", "temperature_keV"],
                    "importance": [0.35, 0.6, 0.05],
                    "source_model_name": ["random_forest", "random_forest", "random_forest"],
                    "importance_method": [
                        "intrinsic_feature_importances",
                        "intrinsic_feature_importances",
                        "intrinsic_feature_importances",
                    ],
                }
            ),
            pd.DataFrame(
                {
                    "feature": ["fuel_density_m3", "pressure_Pa", "temperature_keV"],
                    "importance": [0.5, 0.3, 0.2],
                    "source_model_name": ["random_forest", "random_forest", "random_forest"],
                    "importance_method": [
                        "intrinsic_feature_importances",
                        "intrinsic_feature_importances",
                        "intrinsic_feature_importances",
                    ],
                }
            ),
        ]
    )

    def fake_extract_feature_importance(
        _model: object,
        _feature_columns: list[str],
        *,
        X_reference: pd.DataFrame,
        y_reference: pd.Series,
        model_name: str,
    ) -> tuple[pd.DataFrame, str]:
        assert len(X_reference) == len(y_reference)
        assert model_name == "random_forest"
        return next(fold_outputs), "intrinsic_feature_importances"

    monkeypatch.setattr(training, "extract_feature_importance", fake_extract_feature_importance)

    importance_df, importance_method = train_model.extract_cross_validated_feature_importance(
        lambda: NoOpModel(),
        feature_columns,
        X_train=X_train,
        y_train=y_train,
        cv_splits=cv_splits,
        model_name="random_forest",
    )

    pressure_row = importance_df.set_index("feature").loc["pressure_Pa"]

    assert importance_method == "cross_validated_intrinsic_feature_importances"
    assert set(importance_df["feature"]) == set(feature_columns)
    assert pressure_row["importance"] == pytest.approx(0.3)
    assert pressure_row["importance_mean_when_present"] == pytest.approx(0.45)
    assert pressure_row["cv_fold_count"] == 3
    assert pressure_row["cv_folds_present"] == 2
    assert pressure_row["cv_folds_missing"] == 1


def test_train_parser_help_mentions_explicit_synthetic_flag() -> None:
    parser = train_model.build_parser()
    train_subparser = next(
        action.choices["train"]
        for action in parser._actions
        if hasattr(action, "choices") and action.choices is not None and "train" in action.choices
    )
    help_text = train_subparser.format_help()

    assert "--allow-synthetic" in help_text
    assert "--assume-temperature-unit" in help_text
    assert "--shot-prediction-cutoff-rows" in help_text
    assert "--skip-report-generation" in help_text
    assert "Generate and train on synthetic demo data" in help_text
    assert "--dataset-path" in help_text


def test_train_cli_wires_cutoff_flag_and_optional_report_generation(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    frame = pd.DataFrame(
        {
            "shot_id": np.repeat(np.arange(15), 3),
            "time_s": list(np.tile([0.0, 1.0, 2.0], 15)),
            "fuel_density_m3": np.linspace(1.0e20, 2.2e20, 45),
            "temperature_keV": np.linspace(8.0, 20.0, 45),
            "confinement_time_s": np.linspace(0.8, 2.0, 45),
            "neutron_yield": np.linspace(50.0, 250.0, 45),
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "cli_training.csv")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "train",
            "--dataset-path",
            str(dataset_path),
            "--shot-prediction-cutoff-rows",
            "1",
            "--skip-report-generation",
        ],
    )

    train_model.main()
    artifacts = json.loads(capsys.readouterr().out)
    metadata = json.loads(Path(artifacts["metadata_path"]).read_text())
    processed = pd.read_csv(artifacts["prediction_path"])

    assert artifacts["report_generation_enabled"] is False
    assert metadata["dataset_preparation"]["shot_prediction_cutoff_rows"] == 1
    assert metadata["holdout_evaluation"]["report_generation_enabled"] is False
    assert metadata["model_explainability"]["enabled"] is False
    assert metadata["model_explainability"]["importance_method"] is None
    assert artifacts["feature_importance_path"] is None
    assert artifacts["importance_plot_path"] is None
    assert artifacts["residual_plot_path"] is None
    assert metadata["artifacts"]["feature_importance_path"] is None
    assert metadata["artifacts"]["importance_plot_path"] is None
    assert metadata["artifacts"]["residual_plot_path"] is None
    assert processed["time_s"].eq(0.0).all()


def test_prediction_artifacts_retain_row_identity(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "HIGH_YIELD_PERCENTILE", 0.0)
    monkeypatch.setattr(training, "LOW_LAWSON_RATIO_THRESHOLD", 1.0e9)
    frame = pd.DataFrame(
        {
            "shot_id": np.arange(15),
            "fuel_density_m3": np.linspace(1.0e20, 2.4e20, 15),
            "temperature_keV": np.linspace(8.0, 22.0, 15),
            "confinement_time_s": np.linspace(0.8, 2.2, 15),
            "neutron_yield": np.linspace(100.0, 400.0, 15),
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "identity.csv")

    artifacts = train_model.train_models(dataset_path)
    prediction_frame = pd.read_csv(artifacts["prediction_path"])
    mismatch_frame = pd.read_csv(artifacts["mismatch_path"])

    for artifact_frame in (prediction_frame, mismatch_frame):
        assert {"shot_id", ORIGINAL_ROW_INDEX_COLUMN, RAW_CSV_ROW_NUMBER_COLUMN}.issubset(artifact_frame.columns)
        assert (artifact_frame[RAW_CSV_ROW_NUMBER_COLUMN] == artifact_frame[ORIGINAL_ROW_INDEX_COLUMN] + 2).all()
        assert set(artifact_frame["shot_id"]).issubset(set(frame["shot_id"]))
    assert mismatch_frame["physics_mismatch_flag_mode"].eq("predicted_percentile").all()
    assert mismatch_frame["physics_mismatch_high_yield_threshold_source"].eq("top_0pct_holdout_predictions").all()
    assert mismatch_frame["physics_mismatch_low_lawson_ratio_threshold"].eq(1.0e9).all()

    metadata = json.loads(Path(artifacts["metadata_path"]).read_text())
    assert metadata["holdout_evaluation"]["physics_mismatch_flagging"]["flag_mode"] == "predicted_percentile"
    assert metadata["holdout_evaluation"]["physics_mismatch_flagging"]["high_yield_percentile"] == 0.0


def test_train_models_handles_constant_target_group_folds(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    rows: list[dict[str, float | int]] = []
    for shot_id, target in enumerate([100.0, 150.0, 200.0, 250.0, 300.0], start=1):
        for offset in range(3):
            rows.append(
                {
                    "shot_id": shot_id,
                    "fuel_density_m3": 1.0e20 + shot_id * 1.0e19 + offset * 1.0e18,
                    "temperature_keV": 10.0 + shot_id + offset * 0.1,
                    "confinement_time_s": 1.0 + shot_id * 0.1 + offset * 0.05,
                    "neutron_yield": target,
                }
            )
    dataset_path = _write_dataset(tmp_path, pd.DataFrame(rows), "constant_target_groups.csv")

    artifacts = train_model.train_models(dataset_path)
    metrics = pd.read_csv(artifacts["metrics_path"])

    assert Path(artifacts["model_path"]).exists()
    assert metrics["holdout_rmse"].ge(0).all()
    assert metrics["holdout_r2"].isna().all()


def test_select_split_indices_targets_rows_for_uneven_group_sizes() -> None:
    rows: list[dict[str, float | int]] = []
    for shot_id, row_count in enumerate([10, 10, 1, 1, 1, 1, 1], start=1):
        for row_index in range(row_count):
            rows.append(
                {
                    "shot_id": shot_id,
                    "fuel_density_m3": 1.0e20 + shot_id * 1.0e18 + row_index,
                    "temperature_keV": 10.0 + shot_id,
                    "confinement_time_s": 1.0 + row_index * 0.01,
                    "neutron_yield": 100.0 + shot_id,
                }
            )
    frame = pd.DataFrame(rows)

    train_idx, test_idx, split_strategy = train_model.select_split_indices(frame, random_state=42)

    train_shots = set(frame.iloc[train_idx]["shot_id"])
    test_shots = set(frame.iloc[test_idx]["shot_id"])

    assert split_strategy == "group_row_target_split"
    assert train_shots.isdisjoint(test_shots)
    assert len(test_idx) == 5
    assert len(train_idx) == len(frame) - 5


def test_select_split_indices_handles_many_repeated_groups() -> None:
    rows: list[dict[str, float | int]] = []
    group_sizes = [2 + (shot_id % 4) for shot_id in range(80)]
    for shot_id, row_count in enumerate(group_sizes, start=1):
        for row_index in range(row_count):
            rows.append(
                {
                    "shot_id": shot_id,
                    "fuel_density_m3": 1.0e20 + shot_id * 1.0e18 + row_index,
                    "temperature_keV": 10.0 + shot_id * 0.1,
                    "confinement_time_s": 1.0 + row_index * 0.01,
                    "neutron_yield": 100.0 + shot_id,
                }
            )
    frame = pd.DataFrame(rows)

    target_test_rows = int(
        np.ceil(
            len(frame)
            * max(
                train_model.HOLDOUT_TEST_SIZE,
                train_model.MIN_TEST_SAMPLES / len(frame),
            )
        )
    )
    train_idx, test_idx, split_strategy = train_model.select_split_indices(frame, random_state=42)

    train_shots = set(frame.iloc[train_idx]["shot_id"])
    test_shots = set(frame.iloc[test_idx]["shot_id"])

    assert split_strategy == "group_row_target_split"
    assert train_shots.isdisjoint(test_shots)
    assert train_model.MIN_TEST_SAMPLES <= len(test_idx) <= len(frame) - train_model.MIN_TRAIN_SAMPLES
    assert abs(len(test_idx) - target_test_rows) <= max(group_sizes)
