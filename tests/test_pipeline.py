from __future__ import annotations

import inspect
import json
import re
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

import config
import features
import inference
import storage
import train_model
import training
import validation
from artifact_model import FusionFluxModelArtifact
from config import ORIGINAL_ROW_INDEX_COLUMN, RAW_CSV_ROW_NUMBER_COLUMN


class NegativePredictingModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert list(frame.columns) == [
            "fuel_density_m3",
            "temperature_keV",
            "confinement_time_s",
            "triple_product",
        ]
        return np.array([-5.0], dtype=float)


class FeatureEchoModel:
    def __init__(self, feature_name: str) -> None:
        self.feature_name = feature_name

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.feature_name].to_numpy(dtype=float)


def _build_feature_echo_artifact(
    *,
    feature_name: str,
    training_run_id: str,
    model_name: str,
    feature_columns: list[str] | None = None,
) -> FusionFluxModelArtifact:
    resolved_feature_columns = feature_columns or [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    return FusionFluxModelArtifact(
        FeatureEchoModel(feature_name),
        schema_version=inference.ARTIFACT_SCHEMA_VERSION,
        training_run_id=training_run_id,
        feature_columns=resolved_feature_columns,
        model_name=model_name,
        preprocessing_contract=features.build_preprocessing_contract(),
    )


def _build_temperature_echo_artifact(
    *,
    training_run_id: str,
    feature_columns: list[str] | None = None,
) -> FusionFluxModelArtifact:
    return _build_feature_echo_artifact(
        feature_name="temperature_keV",
        training_run_id=training_run_id,
        model_name="temperature_echo",
        feature_columns=feature_columns,
    )


def _write_prediction_artifact_run(
    processed_dir: Path,
    *,
    training_run_id: str,
    artifact: FusionFluxModelArtifact,
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    run_dir = processed_dir / train_model.TRAINING_RUNS_DIRNAME / training_run_id
    model_path = run_dir / "models" / train_model.TRAINING_MODEL_FILENAME
    metadata_path = run_dir / train_model.TRAINING_METADATA_FILENAME
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return model_path, metadata_path


def _bump_version_component(version: str, *, component_index: int) -> str:
    numeric_parts = re.findall(r"\d+", version)
    if len(numeric_parts) <= component_index:
        raise AssertionError(f"Version {version!r} does not contain component {component_index}.")
    updated_parts = [int(part) for part in numeric_parts[: max(component_index + 1, 3)]]
    updated_parts[component_index] += 1
    for reset_index in range(component_index + 1, len(updated_parts)):
        updated_parts[reset_index] = 0
    return ".".join(str(part) for part in updated_parts[:3])


@pytest.fixture
def isolated_project_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw_dir)

    return {
        "raw": raw_dir,
        "processed": processed_dir,
    }


def _write_dataset(tmp_path: Path, frame: pd.DataFrame, name: str = "dataset.csv") -> Path:
    dataset_path = tmp_path / name
    frame.to_csv(dataset_path, index=False)
    return dataset_path


def _build_negative_artifact() -> FusionFluxModelArtifact:
    return FusionFluxModelArtifact(
        NegativePredictingModel(),
        schema_version=inference.ARTIFACT_SCHEMA_VERSION,
        training_run_id="negative_model_run",
        feature_columns=("fuel_density_m3", "temperature_keV", "confinement_time_s", "triple_product"),
        model_name="negative_dummy",
        preprocessing_contract=features.build_preprocessing_contract(),
    )


def _build_artifact_metadata(
    *,
    model_path: Path,
    training_run_id: str = "negative_model_run",
    feature_columns: list[str] | None = None,
    best_model_name: str = "negative_dummy",
    preprocessing_contract: dict[str, object] | None = None,
    assume_temperature_unit: str | None = None,
    shot_prediction_cutoff_rows: int = features.DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS,
) -> dict[str, object]:
    resolved_feature_columns = feature_columns or [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    contract = preprocessing_contract or features.build_preprocessing_contract()
    return {
        "schema_version": inference.ARTIFACT_SCHEMA_VERSION,
        "training_run_id": training_run_id,
        "feature_columns": resolved_feature_columns,
        "best_model_name": best_model_name,
        "preprocessing": contract,
        "runtime_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "scikit_learn": train_model.sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "saved_model": {
            "path": str(model_path),
            "artifact_type": "FusionFluxModelArtifact",
            "model_name": best_model_name,
            "fit_scope": "full_prepared_dataset",
            "row_count": 1,
            "training_run_id": training_run_id,
        },
        "dataset_preparation": {
            "assume_temperature_unit": assume_temperature_unit,
            "shot_prediction_cutoff_rows": shot_prediction_cutoff_rows,
            "row_identity_columns": [ORIGINAL_ROW_INDEX_COLUMN, RAW_CSV_ROW_NUMBER_COLUMN],
        },
    }


def _load_training_metadata_record(metadata_path: Path) -> inference.TrainingArtifactMetadata:
    return inference._parse_training_artifact_metadata(
        inference._resolve_training_metadata_paths(
            json.loads(metadata_path.read_text()),
            metadata_path=metadata_path,
        ),
        metadata_path=metadata_path,
    )


def _build_grouped_time_series_frame(*, shot_count: int = 15, rows_per_shot: int = 4) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for shot_id in range(1, shot_count + 1):
        base_density = 1.0e20 + shot_id * 8.0e17
        base_temperature = 7.5 + shot_id * 0.35
        base_tau = 0.7 + shot_id * 0.04
        base_yield = 80.0 + shot_id * 18.0
        for offset in range(rows_per_shot):
            rows.append(
                {
                    "shot_id": shot_id,
                    "time_s": float(offset),
                    "fuel_density_m3": base_density + offset * 2.0e17,
                    "temperature_keV": base_temperature + offset * 1.25,
                    "confinement_time_s": base_tau + offset * 0.08,
                    "neutron_yield": base_yield + offset * 7.0,
                }
            )
    return pd.DataFrame(rows)


def test_create_synthetic_dataset_handles_non_multiple_row_counts(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    dataset_path = features.create_synthetic_dataset(tmp_path / "synthetic.csv", n_rows=10, random_state=7)
    dataset = pd.read_csv(dataset_path)

    assert len(dataset) == 10
    assert dataset["shot_id"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_prepare_dataset_normalizes_aliases_and_aggregates_shots(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "experiment_id": [101, 101, 101, 202, 202, 202],
            "time_ms": [0, 50, 100, 0, 50, 100],
            "density_m3": [1.00e20, 1.10e20, 1.20e20, 0.90e20, 0.95e20, 1.00e20],
            "temperature_eV": [10000, 12000, 14000, 15000, 17000, 19000],
            "tau_E": [1.0, 1.2, 1.4, 0.8, 1.0, 1.2],
            "yield": [100.0, 150.0, 200.0, 50.0, 60.0, 70.0],
            "fuel_mix_purity": [0.95, 0.95, 0.95, 0.92, 0.92, 0.92],
            "energy_input": [30.0, 32.0, 34.0, 20.0, 22.0, 24.0],
            "pressure": [1.1e5, 1.2e5, 1.3e5, 0.9e5, 1.0e5, 1.1e5],
            "plasma_current_MA": [10.0, 10.5, 11.0, 8.0, 8.2, 8.4],
            "magnetic_field_T": [5.0, 5.1, 5.2, 4.3, 4.4, 4.5],
            "major_radius_m": [3.0, 3.0, 3.0, 2.7, 2.7, 2.7],
            "minor_radius_m": [1.0, 1.0, 1.0, 0.85, 0.85, 0.85],
            "elongation": [1.8, 1.8, 1.8, 1.7, 1.7, 1.7],
            "power_input_MW": [25.0, 25.5, 26.0, 18.0, 18.5, 19.0],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "aliased.csv")

    prepared = features.prepare_dataset(dataset_path)
    aggregated = prepared.dataframe.sort_values("shot_id").reset_index(drop=True)

    assert len(aggregated) == 2
    assert prepared.column_mapping["experiment_id"] == "shot_id"
    assert prepared.column_mapping["density_m3"] == "fuel_density_m3"
    assert prepared.column_mapping["yield"] == "neutron_yield"
    assert aggregated.loc[0, "temperature_keV"] == pytest.approx(11.0)
    assert aggregated.loc[1, "temperature_keV"] == pytest.approx(16.0)
    assert aggregated.loc[0, "neutron_yield"] == pytest.approx(150.0)
    assert aggregated.loc[1, "neutron_yield"] == pytest.approx(60.0)
    assert "tau_E_ipb98_s" in aggregated.columns
    assert aggregated["tau_E_ipb98_s"].notna().all()
    assert prepared.processed_path.exists()


def test_prepare_dataset_writes_processed_csv_atomically(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20, 1.2e20],
            "temperature_keV": [10.0, 11.0, 12.0],
            "confinement_time_s": [1.0, 1.1, 1.2],
            "neutron_yield": [100.0, 110.0, 120.0],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "atomic_prepare.csv")
    output_path = tmp_path / "prepared_atomic.csv"
    output_path.write_text("stable-existing-output")

    def fail_replace(_src: Path, _dst: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(storage.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        features.prepare_dataset(dataset_path, processed_output_path=output_path)

    assert output_path.read_text() == "stable-existing-output"
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []


def test_prepare_dataset_rejects_invalid_optional_physics_inputs(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20],
            "temperature_keV": [12.0, 13.0],
            "confinement_time_s": [1.0, 1.1],
            "neutron_yield": [100.0, 110.0],
            "pressure_Pa": [-1.0, 1.2e5],
            "Ip_MA": [10.0, 0.0],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "invalid_optional.csv")

    with pytest.raises(ValueError, match="pressure_Pa.*Ip_MA|Ip_MA.*pressure_Pa"):
        features.prepare_dataset(dataset_path)


def test_prepare_dataset_rejects_optional_columns_that_are_present_but_non_numeric(
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
    dataset_path = _write_dataset(tmp_path, frame, "invalid_optional_strings.csv")

    with pytest.raises(ValueError, match="pressure_Pa.*numeric when provided"):
        features.prepare_dataset(dataset_path)


def test_prepare_dataset_rejects_inconsistent_ne_20(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20],
            "temperature_keV": [12.0, 13.0],
            "confinement_time_s": [1.0, 1.1],
            "neutron_yield": [100.0, 110.0],
            "ne_20": [1.0, 1.5],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "inconsistent_ne20.csv")

    with pytest.raises(ValueError, match="ne_20.*fuel_density_m3 / 1e20"):
        features.prepare_dataset(dataset_path)


def test_prepare_dataset_derives_missing_ne_20_before_ipb98(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.2e20, 1.4e20],
            "temperature_keV": [12.0, 13.0, 14.0],
            "confinement_time_s": [1.0, 1.1, 1.2],
            "neutron_yield": [100.0, 110.0, 120.0],
            "Ip_MA": [10.0, 10.5, 11.0],
            "Bt_T": [5.0, 5.1, 5.2],
            "R_m": [3.0, 3.0, 3.0],
            "a_m": [1.0, 1.0, 1.0],
            "kappa": [1.8, 1.8, 1.8],
            "Pin_MW": [25.0, 25.5, 26.0],
            "M_amu": [2.5, 2.5, 2.5],
            "ne_20": [np.nan, 1.2, ""],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "derived_ne20.csv")

    prepared = features.prepare_dataset(dataset_path)

    assert prepared.dataframe["ne_20"].tolist() == pytest.approx([1.0, 1.2, 1.4])
    assert prepared.dataframe["tau_E_ipb98_s"].notna().all()


def test_prepare_dataset_rejects_conflicting_duplicate_alias_columns(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20],
            "temperature_keV": [12.0, 13.0],
            "confinement_time_s": [1.0, 1.1],
            "neutron_yield": [100.0, 110.0],
            "pressure_Pa": [1.0e5, 1.2e5],
            "pressure": [1.0e5, 9.9e5],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "conflicting_aliases.csv")

    with pytest.raises(ValueError, match="Conflicting source columns for pressure_Pa"):
        features.prepare_dataset(dataset_path)


def test_prepare_dataset_rejects_conflicting_temperature_sources(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20],
            "temperature_keV": [12.0, 13.0],
            "temperature_eV": [12000.0, 15000.0],
            "confinement_time_s": [1.0, 1.1],
            "neutron_yield": [100.0, 110.0],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "conflicting_temperature.csv")

    with pytest.raises(ValueError, match="Conflicting source columns for temperature_keV"):
        features.prepare_dataset(dataset_path)


def test_prepare_dataset_rejects_bare_temperature_without_units(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20],
            "temperature": [12.0, 13.0],
            "confinement_time_s": [1.0, 1.1],
            "neutron_yield": [100.0, 110.0],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "bare_temperature.csv")

    with pytest.raises(ValueError, match="temperature_unit.*assume_temperature_unit"):
        features.prepare_dataset(dataset_path)


def test_prepare_dataset_allows_explicit_temperature_unit_assumption(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20],
            "temperature": [12000.0, 13000.0],
            "confinement_time_s": [1.0, 1.1],
            "neutron_yield": [100.0, 110.0],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "assumed_temperature.csv")

    prepared = features.prepare_dataset(dataset_path, assume_temperature_unit="eV")

    assert prepared.dataframe["temperature_keV"].tolist() == pytest.approx([12.0, 13.0])


def test_aggregate_time_resolved_shots_uses_fixed_cutoff_rows() -> None:
    frame = pd.DataFrame(
        {
            "shot_id": [10, 10, 20, 20, 20, 20],
            "time_s": [0.0, 1.0, 0.0, 1.0, 2.0, 3.0],
            "fuel_density_m3": [1.0e20, 2.0e20, 1.0e20, 2.0e20, 3.0e20, 4.0e20],
            "temperature_keV": [10.0, 30.0, 10.0, 20.0, 30.0, 40.0],
            "confinement_time_s": [1.0, 3.0, 1.0, 2.0, 3.0, 4.0],
            "neutron_yield": [5.0, 10.0, 5.0, 10.0, 15.0, 20.0],
        }
    )

    aggregated = features.aggregate_time_resolved_shots(frame).sort_values("shot_id").reset_index(drop=True)

    assert aggregated.loc[0, "temperature_keV"] == pytest.approx(20.0)
    assert aggregated.loc[0, "neutron_yield"] == pytest.approx(10.0)
    assert aggregated.loc[0, "time_s"] == pytest.approx(1.0)
    assert aggregated.loc[1, "temperature_keV"] == pytest.approx(15.0)
    assert aggregated.loc[1, "neutron_yield"] == pytest.approx(10.0)
    assert aggregated.loc[1, "time_s"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("column", "values", "expected_message"),
    [
        ("time_s", [0.0, "bad"], "time_s.*numeric"),
        ("time_ms", [0, ""], "time_ms.*present"),
    ],
)
def test_prepare_dataset_rejects_invalid_timestamps_for_shot_aggregation(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    column: str,
    values: list[object],
    expected_message: str,
) -> None:
    frame = pd.DataFrame(
        {
            "shot_id": [10, 10],
            column: values,
            "fuel_density_m3": [1.0e20, 1.1e20],
            "temperature_keV": [10.0, 11.0],
            "confinement_time_s": [1.0, 1.1],
            "neutron_yield": [5.0, 6.0],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, f"invalid_{column}.csv")

    with pytest.raises(ValueError, match=expected_message):
        features.prepare_dataset(dataset_path)


def test_prepare_dataset_rejects_missing_shot_ids_before_group_split_logic(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = _build_grouped_time_series_frame()
    frame.loc[4, "shot_id"] = np.nan
    dataset_path = _write_dataset(tmp_path, frame, "missing_shot_id.csv")

    with pytest.raises(ValueError, match=r"shot_id rows \[4\].*present and non-empty"):
        features.prepare_dataset(dataset_path)


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


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"confinement_time_s": -1.0}, "confinement_time_s"),
        ({"pressure_pa": -1.0}, "pressure_Pa"),
        ({"ne_20": 1.25}, "ne_20 must match fuel_density_m3 / 1e20"),
    ],
)
def test_predict_single_case_rejects_invalid_inputs_before_loading_model(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    kwargs: dict[str, float],
    expected_message: str,
) -> None:
    params = {
        "density_m3": 1.0e20,
        "temperature": 12.0,
        "confinement_time_s": 1.0,
        "temp_unit": "keV",
        "fuel_purity": 0.95,
        "energy_input_mj": 20.0,
        "pressure_pa": 1.0e5,
        "ip_ma": 10.0,
        "bt_t": 5.0,
        "r_m": 3.0,
        "a_m": 1.0,
        "kappa": 1.8,
        "ne_20": 1.0,
        "m_amu": 2.5,
        "pin_mw": 20.0,
        "model_path": tmp_path / "missing.joblib",
        "metadata_path": tmp_path / "missing.json",
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=expected_message):
        train_model.predict_single_case(**params)


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_validate_physics_value_rejects_boolean_inputs(value: object) -> None:
    with pytest.raises(ValueError, match="fuel_density_m3.*boolean"):
        validation.validate_physics_value(value, "fuel_density_m3")


def test_validate_physics_dataframe_rejects_boolean_inputs() -> None:
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20],
            "temperature_keV": [12.0, 13.0],
            "confinement_time_s": [1.0, 1.1],
            "neutron_yield": [100.0, 110.0],
            "pressure_Pa": [True, 1.2e5],
        }
    )

    with pytest.raises(ValueError, match="pressure_Pa.*boolean"):
        validation.validate_physics_dataframe(
            frame,
            required_fields=("fuel_density_m3", "temperature_keV", "confinement_time_s", "neutron_yield"),
            optional_fields=("pressure_Pa",),
        )


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
    assert metadata["preprocessing"]["source_sha256"] == features.build_preprocessing_contract()["source_sha256"]
    assert (
        metadata["preprocessing"]["source_fingerprint_method"]
        == features.PREPROCESSING_LOGIC_FINGERPRINT_METHOD
    )
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
    assert metadata["model_selection"]["primary_metric"] == "cv_rmse_mean"
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
        def fit(self, _X: pd.DataFrame, _y: pd.Series) -> NoOpModel:
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


def test_predict_batch_parser_help_mentions_csv_scoring() -> None:
    parser = train_model.build_parser()
    predict_batch_subparser = next(
        action.choices["predict-batch"]
        for action in parser._actions
        if hasattr(action, "choices") and action.choices is not None and "predict-batch" in action.choices
    )
    help_text = predict_batch_subparser.format_help()

    assert "--input-csv" in help_text
    assert "--output-path" in help_text
    assert "--assume-temperature-unit" in help_text


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


def test_train_and_predict_end_to_end(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    dataset_path = features.create_synthetic_dataset(tmp_path / "synthetic_train.csv", n_rows=60, random_state=11)
    artifacts = train_model.train_models(dataset_path)
    metrics = pd.read_csv(artifacts["metrics_path"])
    source_row = pd.read_csv(dataset_path).iloc[0]

    assert {"cv_rmse_mean", "holdout_rmse", "holdout_r2"}.issubset(metrics.columns)
    assert Path(artifacts["model_path"]).exists()
    assert Path(artifacts["metadata_path"]).exists()

    prediction = train_model.predict_single_case(
        density_m3=float(source_row["fuel_density_m3"]),
        temperature=float(source_row["temperature_keV"]),
        confinement_time_s=float(source_row["confinement_time_s"]),
        temp_unit="keV",
        fuel_purity=float(source_row["fuel_purity"]),
        energy_input_mj=float(source_row["energy_input_MJ"]),
        pressure_pa=float(source_row["pressure_Pa"]),
        ip_ma=float(source_row["Ip_MA"]),
        bt_t=float(source_row["Bt_T"]),
        r_m=float(source_row["R_m"]),
        a_m=float(source_row["a_m"]),
        kappa=float(source_row["kappa"]),
        ne_20=None,
        m_amu=float(source_row["M_amu"]),
        pin_mw=float(source_row["Pin_MW"]),
        model_path=artifacts["model_path"],
        metadata_path=artifacts["metadata_path"],
    )

    assert np.isfinite(prediction["predicted_neutron_yield"])
    assert prediction["predicted_neutron_yield"] > 0
    assert np.isfinite(prediction["lawson_ratio"])
    assert prediction["status"] in {"IGNITION REACHED", "SUB-CRITICAL"}
    assert prediction["clipped_negative_prediction"] is False
    assert prediction["prediction_warnings"] == []


def test_predict_batch_scores_rows_and_emits_artifact_metadata(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    dataset_path = features.create_synthetic_dataset(tmp_path / "synthetic_train.csv", n_rows=60, random_state=41)
    artifacts = train_model.train_models(dataset_path)
    source_rows = pd.read_csv(dataset_path).iloc[:3].copy()
    batch_input = pd.DataFrame(
        {
            "density_m3": source_rows["fuel_density_m3"].to_numpy(),
            "temperature": source_rows["temperature_keV"].to_numpy() * 1e3,
            "temperature_unit": ["eV"] * len(source_rows),
            "tau_E": source_rows["confinement_time_s"].to_numpy(),
            "fuel_mix_purity": source_rows["fuel_purity"].to_numpy(),
            "energy_input": source_rows["energy_input_MJ"].to_numpy(),
        }
    )
    output_path = tmp_path / "batch_predictions.csv"

    result = inference.predict_batch(
        batch_input,
        output_path=output_path,
        model_path=artifacts["model_path"],
        metadata_path=artifacts["metadata_path"],
    )

    assert result.output_path == output_path.resolve()
    assert result.column_mapping["density_m3"] == "fuel_density_m3"
    assert output_path.exists()
    assert len(result.predictions) == 3
    assert ORIGINAL_ROW_INDEX_COLUMN in result.predictions.columns
    assert RAW_CSV_ROW_NUMBER_COLUMN in result.predictions.columns
    assert result.predictions["predicted_neutron_yield"].gt(0).all()
    assert result.predictions["artifact_training_run_id"].eq(result.training_run_id).all()
    assert result.predictions["artifact_model_name"].eq(result.model_name).all()
    assert result.predictions["artifact_schema_version"].eq(result.schema_version).all()


def test_predict_batch_writes_output_atomically(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "temperature_echo.joblib"
    metadata_path = tmp_path / "temperature_echo_metadata.json"
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    artifact = FusionFluxModelArtifact(
        FeatureEchoModel("temperature_keV"),
        schema_version=inference.ARTIFACT_SCHEMA_VERSION,
        training_run_id="atomic_batch_output_run",
        feature_columns=feature_columns,
        model_name="temperature_echo",
        preprocessing_contract=features.build_preprocessing_contract(),
    )
    joblib.dump(artifact, model_path)
    metadata_path.write_text(
        json.dumps(
            _build_artifact_metadata(
                model_path=model_path,
                training_run_id="atomic_batch_output_run",
                feature_columns=feature_columns,
                best_model_name="temperature_echo",
            ),
            indent=2,
        )
    )
    batch_input = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20],
            "temperature_keV": [12.0],
            "confinement_time_s": [1.0],
        }
    )
    output_path = tmp_path / "atomic_batch_predictions.csv"
    output_path.write_text("existing-predictions")

    def fail_replace(_src: Path, _dst: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(storage.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        inference.predict_batch(
            batch_input,
            output_path=output_path,
            model_path=model_path,
            metadata_path=metadata_path,
        )

    assert output_path.read_text() == "existing-predictions"
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []


def test_predict_batch_aggregates_time_resolved_shots_using_saved_cutoff_and_runtime_reuse(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "temperature_echo.joblib"
    metadata_path = tmp_path / "temperature_echo_metadata.json"
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    artifact = FusionFluxModelArtifact(
        FeatureEchoModel("temperature_keV"),
        schema_version=inference.ARTIFACT_SCHEMA_VERSION,
        training_run_id="time_resolved_run",
        feature_columns=feature_columns,
        model_name="temperature_echo",
        preprocessing_contract=features.build_preprocessing_contract(),
    )
    joblib.dump(artifact, model_path)
    metadata_path.write_text(
        json.dumps(
            _build_artifact_metadata(
                model_path=model_path,
                training_run_id="time_resolved_run",
                feature_columns=feature_columns,
                best_model_name="temperature_echo",
                shot_prediction_cutoff_rows=3,
            ),
            indent=2,
        )
    )

    runtime = inference.load_prediction_runtime(model_path=model_path, metadata_path=metadata_path)
    batch_input = pd.DataFrame(
        {
            "pulse_id": [11, 11, 11, 11, 22, 22, 22, 22],
            "time_s": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
            "density_m3": [1.0e20, 1.1e20, 1.2e20, 1.3e20, 1.5e20, 1.6e20, 1.7e20, 1.8e20],
            "temperature_eV": [10000.0, 20000.0, 30000.0, 100000.0, 40000.0, 50000.0, 60000.0, 70000.0],
            "tau_E": [1.0, 2.0, 3.0, 10.0, 1.5, 2.5, 3.5, 4.5],
        }
    )

    monkeypatch.setattr(
        inference,
        "_load_prediction_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prediction runtime should reuse the loaded artifact")),
    )

    result = inference.predict_batch(batch_input, runtime=runtime)
    single_prediction = inference.predict_single_case(
        density_m3=1.0e20,
        temperature=12.0,
        confinement_time_s=1.0,
        temp_unit="keV",
        fuel_purity=None,
        energy_input_mj=None,
        pressure_pa=None,
        ip_ma=None,
        bt_t=None,
        r_m=None,
        a_m=None,
        kappa=None,
        ne_20=None,
        m_amu=None,
        pin_mw=None,
        runtime=runtime,
    )

    assert runtime.shot_prediction_cutoff_rows == 3
    assert result.column_mapping["pulse_id"] == "shot_id"
    assert result.column_mapping["density_m3"] == "fuel_density_m3"
    assert len(result.predictions) == 2
    assert result.predictions["shot_id"].tolist() == [11, 22]
    assert result.predictions["temperature_keV"].tolist() == pytest.approx([20.0, 50.0])
    assert result.predictions["time_s"].tolist() == pytest.approx([2.0, 2.0])
    assert result.predictions["predicted_neutron_yield"].tolist() == pytest.approx([20.0, 50.0])
    assert single_prediction["predicted_neutron_yield"] == pytest.approx(12.0)


def test_predict_batch_preserves_duplicate_input_rows(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "temperature_echo.joblib"
    metadata_path = tmp_path / "temperature_echo_metadata.json"
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    artifact = FusionFluxModelArtifact(
        FeatureEchoModel("temperature_keV"),
        schema_version=inference.ARTIFACT_SCHEMA_VERSION,
        training_run_id="duplicate_batch_rows_run",
        feature_columns=feature_columns,
        model_name="temperature_echo",
        preprocessing_contract=features.build_preprocessing_contract(),
    )
    joblib.dump(artifact, model_path)
    metadata_path.write_text(
        json.dumps(
            _build_artifact_metadata(
                model_path=model_path,
                training_run_id="duplicate_batch_rows_run",
                feature_columns=feature_columns,
                best_model_name="temperature_echo",
            ),
            indent=2,
        )
    )
    batch_input = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.0e20],
            "temperature_keV": [12.0, 12.0],
            "confinement_time_s": [1.0, 1.0],
        }
    )

    result = inference.predict_batch(
        batch_input,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    assert len(result.predictions) == 2
    assert result.predictions["predicted_neutron_yield"].tolist() == pytest.approx([12.0, 12.0])
    assert result.predictions[ORIGINAL_ROW_INDEX_COLUMN].tolist() == [0, 1]
    assert result.predictions[RAW_CSV_ROW_NUMBER_COLUMN].tolist() == [2, 3]


def test_predict_batch_streams_row_wise_csv_inputs_and_preserves_global_row_identity(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "temperature_echo.joblib"
    metadata_path = tmp_path / "temperature_echo_metadata.json"
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    artifact = FusionFluxModelArtifact(
        FeatureEchoModel("temperature_keV"),
        schema_version=inference.ARTIFACT_SCHEMA_VERSION,
        training_run_id="streamed_batch_rows_run",
        feature_columns=feature_columns,
        model_name="temperature_echo",
        preprocessing_contract=features.build_preprocessing_contract(),
    )
    joblib.dump(artifact, model_path)
    metadata_path.write_text(
        json.dumps(
            _build_artifact_metadata(
                model_path=model_path,
                training_run_id="streamed_batch_rows_run",
                feature_columns=feature_columns,
                best_model_name="temperature_echo",
            ),
            indent=2,
        )
    )
    batch_input = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20, 1.2e20, 1.3e20, 1.4e20],
            "temperature_keV": [10.0, 11.0, 12.0, 13.0, 14.0],
            "confinement_time_s": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )
    input_path = _write_dataset(tmp_path, batch_input, "streamed_batch_rows.csv")
    output_path = tmp_path / "streamed_batch_predictions.csv"
    monkeypatch.setattr(inference, "BATCH_PREDICTION_CSV_CHUNK_ROWS", 2)

    result = inference.predict_batch(
        input_path,
        output_path=output_path,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    written_output = pd.read_csv(output_path)

    assert result.predictions["predicted_neutron_yield"].tolist() == pytest.approx([10.0, 11.0, 12.0, 13.0, 14.0])
    assert result.predictions[ORIGINAL_ROW_INDEX_COLUMN].tolist() == [0, 1, 2, 3, 4]
    assert result.predictions[RAW_CSV_ROW_NUMBER_COLUMN].tolist() == [2, 3, 4, 5, 6]
    assert written_output[ORIGINAL_ROW_INDEX_COLUMN].tolist() == [0, 1, 2, 3, 4]
    assert written_output[RAW_CSV_ROW_NUMBER_COLUMN].tolist() == [2, 3, 4, 5, 6]


def test_predict_batch_can_skip_returning_predictions_for_low_memory_file_output(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "temperature_echo.joblib"
    metadata_path = tmp_path / "temperature_echo_metadata.json"
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    artifact = FusionFluxModelArtifact(
        FeatureEchoModel("temperature_keV"),
        schema_version=inference.ARTIFACT_SCHEMA_VERSION,
        training_run_id="stream_to_disk_only_run",
        feature_columns=feature_columns,
        model_name="temperature_echo",
        preprocessing_contract=features.build_preprocessing_contract(),
    )
    joblib.dump(artifact, model_path)
    metadata_path.write_text(
        json.dumps(
            _build_artifact_metadata(
                model_path=model_path,
                training_run_id="stream_to_disk_only_run",
                feature_columns=feature_columns,
                best_model_name="temperature_echo",
            ),
            indent=2,
        )
    )
    batch_input = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20, 1.2e20],
            "temperature_keV": [10.0, 11.0, 12.0],
            "confinement_time_s": [1.0, 1.1, 1.2],
        }
    )
    input_path = _write_dataset(tmp_path, batch_input, "disk_only_batch.csv")
    output_path = tmp_path / "disk_only_batch_predictions.csv"

    result = inference.predict_batch(
        input_path,
        output_path=output_path,
        model_path=model_path,
        metadata_path=metadata_path,
        return_predictions=False,
    )

    assert result.predictions is None
    assert result.row_count == 3
    assert pd.read_csv(output_path)["predicted_neutron_yield"].tolist() == pytest.approx([10.0, 11.0, 12.0])


@pytest.mark.parametrize("identity_column", [ORIGINAL_ROW_INDEX_COLUMN, RAW_CSV_ROW_NUMBER_COLUMN])
def test_predict_batch_rejects_single_row_identity_column(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    identity_column: str,
) -> None:
    model_path = tmp_path / "negative.joblib"
    metadata_path = tmp_path / "negative_metadata.json"
    joblib.dump(_build_negative_artifact(), model_path)
    metadata_path.write_text(json.dumps(_build_artifact_metadata(model_path=model_path), indent=2))
    batch_input = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20],
            "temperature_keV": [12.0],
            "confinement_time_s": [1.0],
            identity_column: [0],
        }
    )

    with pytest.raises(ValueError, match="both row identity columns together or omit both"):
        train_model.predict_batch(
            batch_input,
            model_path=model_path,
            metadata_path=metadata_path,
        )


def test_predict_single_case_uses_latest_manifest_and_pipeline_imputation(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    dataset_path = features.create_synthetic_dataset(tmp_path / "synthetic_train.csv", n_rows=60, random_state=19)
    train_model.train_models(dataset_path)
    source_row = pd.read_csv(dataset_path).iloc[0]

    prediction = train_model.predict_single_case(
        density_m3=float(source_row["fuel_density_m3"]),
        temperature=float(source_row["temperature_keV"]),
        confinement_time_s=float(source_row["confinement_time_s"]),
        temp_unit="keV",
        fuel_purity=None,
        energy_input_mj=None,
        pressure_pa=None,
        ip_ma=None,
        bt_t=None,
        r_m=None,
        a_m=None,
        kappa=None,
        ne_20=None,
        m_amu=None,
        pin_mw=None,
    )

    assert np.isfinite(prediction["predicted_neutron_yield"])
    assert prediction["predicted_neutron_yield"] > 0


def test_predict_single_case_rejects_stale_preprocessing_contract(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    dataset_path = features.create_synthetic_dataset(tmp_path / "synthetic_train.csv", n_rows=60, random_state=29)
    artifacts = train_model.train_models(dataset_path)
    metadata_path = Path(artifacts["metadata_path"])
    stale_metadata_path = metadata_path.parent / "stale_preprocessing_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["preprocessing"]["target_column"] = "tampered_target"
    stale_metadata_path.write_text(json.dumps(metadata, indent=2))
    source_row = pd.read_csv(dataset_path).iloc[0]

    with pytest.raises(ValueError, match="preprocessing contract"):
        train_model.predict_single_case(
            density_m3=float(source_row["fuel_density_m3"]),
            temperature=float(source_row["temperature_keV"]),
            confinement_time_s=float(source_row["confinement_time_s"]),
            temp_unit="keV",
            fuel_purity=None,
            energy_input_mj=None,
            pressure_pa=None,
            ip_ma=None,
            bt_t=None,
            r_m=None,
            a_m=None,
            kappa=None,
            ne_20=None,
            m_amu=None,
            pin_mw=None,
            model_path=artifacts["model_path"],
            metadata_path=stale_metadata_path,
        )


def test_predict_single_case_rejects_version_skewed_artifacts_before_deserialization(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "negative.joblib"
    metadata_path = tmp_path / "negative_metadata.json"
    joblib.dump(_build_negative_artifact(), model_path)
    metadata = _build_artifact_metadata(model_path=model_path)
    metadata["runtime_versions"]["scikit_learn"] = "0.0.test"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    def fail_load(_path: Path) -> None:
        raise AssertionError("joblib.load should not be called when version preflight fails")

    monkeypatch.setattr(train_model.joblib, "load", fail_load)

    with pytest.raises(ValueError, match="runtime version mismatch for scikit_learn"):
        train_model.predict_single_case(
            density_m3=1.0e20,
            temperature=12.0,
            confinement_time_s=1.0,
            temp_unit="keV",
            fuel_purity=None,
            energy_input_mj=None,
            pressure_pa=None,
            ip_ma=None,
            bt_t=None,
            r_m=None,
            a_m=None,
            kappa=None,
            ne_20=None,
            m_amu=None,
            pin_mw=None,
            model_path=model_path,
            metadata_path=metadata_path,
        )


def test_predict_single_case_explicit_artifact_paths_remain_strict_even_with_usable_default_artifact(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    default_run_id = "default_runtime_exact"
    default_artifact = _build_temperature_echo_artifact(
        training_run_id=default_run_id,
        feature_columns=feature_columns,
    )
    default_metadata = _build_artifact_metadata(
        model_path=isolated_project_dirs["processed"] / train_model.TRAINING_RUNS_DIRNAME / default_run_id / "models" / train_model.TRAINING_MODEL_FILENAME,
        training_run_id=default_run_id,
        feature_columns=feature_columns,
        best_model_name="temperature_echo",
    )
    default_model_path, default_metadata_path = _write_prediction_artifact_run(
        isolated_project_dirs["processed"],
        training_run_id=default_run_id,
        artifact=default_artifact,
        metadata=default_metadata,
    )
    train_model._write_latest_training_run_manifest(
        run_id=default_run_id,
        model_path=default_model_path,
        metadata_path=default_metadata_path,
    )

    bad_model_path = tmp_path / "negative.joblib"
    bad_metadata_path = tmp_path / "negative_metadata.json"
    joblib.dump(_build_negative_artifact(), bad_model_path)
    bad_metadata = _build_artifact_metadata(model_path=bad_model_path)
    bad_metadata["runtime_versions"]["scikit_learn"] = "0.0.test"
    bad_metadata_path.write_text(json.dumps(bad_metadata, indent=2))

    with pytest.raises(ValueError, match="runtime version mismatch for scikit_learn"):
        train_model.predict_single_case(
            density_m3=1.0e20,
            temperature=12.0,
            confinement_time_s=1.0,
            temp_unit="keV",
            fuel_purity=None,
            energy_input_mj=None,
            pressure_pa=None,
            ip_ma=None,
            bt_t=None,
            r_m=None,
            a_m=None,
            kappa=None,
            ne_20=None,
            m_amu=None,
            pin_mw=None,
            model_path=bad_model_path,
            metadata_path=bad_metadata_path,
        )


@pytest.mark.parametrize("runtime_field", ["python", "pandas"])
def test_predict_single_case_rejects_python_and_pandas_version_skew_before_deserialization(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_field: str,
) -> None:
    model_path = tmp_path / "negative.joblib"
    metadata_path = tmp_path / "negative_metadata.json"
    joblib.dump(_build_negative_artifact(), model_path)
    metadata = _build_artifact_metadata(model_path=model_path)
    metadata["runtime_versions"][runtime_field] = "0.0.test"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    def fail_load(_path: Path) -> None:
        raise AssertionError("joblib.load should not be called when version preflight fails")

    monkeypatch.setattr(train_model.joblib, "load", fail_load)

    with pytest.raises(ValueError, match=fr"runtime version mismatch for {runtime_field}"):
        train_model.predict_single_case(
            density_m3=1.0e20,
            temperature=12.0,
            confinement_time_s=1.0,
            temp_unit="keV",
            fuel_purity=None,
            energy_input_mj=None,
            pressure_pa=None,
            ip_ma=None,
            bt_t=None,
            r_m=None,
            a_m=None,
            kappa=None,
            ne_20=None,
            m_amu=None,
            pin_mw=None,
            model_path=model_path,
            metadata_path=metadata_path,
        )


def test_predict_single_case_rejects_corrupted_metadata_before_deserialization(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "negative.joblib"
    metadata_path = tmp_path / "corrupted_metadata.json"
    joblib.dump(_build_negative_artifact(), model_path)
    metadata_path.write_text("{not-json")

    def fail_load(_path: Path) -> None:
        raise AssertionError("joblib.load should not be called when metadata JSON is corrupted")

    monkeypatch.setattr(train_model.joblib, "load", fail_load)

    with pytest.raises(ValueError, match="Training metadata .* is not valid JSON"):
        train_model.predict_single_case(
            density_m3=1.0e20,
            temperature=12.0,
            confinement_time_s=1.0,
            temp_unit="keV",
            fuel_purity=None,
            energy_input_mj=None,
            pressure_pa=None,
            ip_ma=None,
            bt_t=None,
            r_m=None,
            a_m=None,
            kappa=None,
            ne_20=None,
            m_amu=None,
            pin_mw=None,
            model_path=model_path,
            metadata_path=metadata_path,
        )


def test_predict_single_case_rejects_corrupted_latest_manifest_before_deserialization(
    isolated_project_dirs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = isolated_project_dirs["processed"] / train_model.LATEST_TRAINING_RUN_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{bad-manifest")

    def fail_load(_path: Path) -> None:
        raise AssertionError("joblib.load should not be called when manifest JSON is corrupted")

    monkeypatch.setattr(train_model.joblib, "load", fail_load)

    with pytest.raises(ValueError, match="No usable training artifacts were found.*Artifact manifest .* is not valid JSON"):
        train_model.predict_single_case(
            density_m3=1.0e20,
            temperature=12.0,
            confinement_time_s=1.0,
            temp_unit="keV",
            fuel_purity=None,
            energy_input_mj=None,
            pressure_pa=None,
            ip_ma=None,
            bt_t=None,
            r_m=None,
            a_m=None,
            kappa=None,
            ne_20=None,
            m_amu=None,
            pin_mw=None,
        )


def test_predict_single_case_falls_back_to_older_default_artifact_and_tolerates_minor_version_drift(
    isolated_project_dirs: dict[str, Path],
) -> None:
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    latest_run_id = "train_20260329T120000Z_latestbad"
    fallback_run_id = "train_20260329T110000Z_fallback"

    latest_artifact = _build_temperature_echo_artifact(
        training_run_id=latest_run_id,
        feature_columns=feature_columns,
    )
    latest_metadata = _build_artifact_metadata(
        model_path=isolated_project_dirs["processed"] / train_model.TRAINING_RUNS_DIRNAME / latest_run_id / "models" / train_model.TRAINING_MODEL_FILENAME,
        training_run_id=latest_run_id,
        feature_columns=feature_columns,
        best_model_name="temperature_echo",
    )
    latest_metadata["runtime_versions"]["python"] = "0.0.test"
    latest_model_path, latest_metadata_path = _write_prediction_artifact_run(
        isolated_project_dirs["processed"],
        training_run_id=latest_run_id,
        artifact=latest_artifact,
        metadata=latest_metadata,
    )

    fallback_artifact = _build_temperature_echo_artifact(
        training_run_id=fallback_run_id,
        feature_columns=feature_columns,
    )
    fallback_metadata = _build_artifact_metadata(
        model_path=isolated_project_dirs["processed"] / train_model.TRAINING_RUNS_DIRNAME / fallback_run_id / "models" / train_model.TRAINING_MODEL_FILENAME,
        training_run_id=fallback_run_id,
        feature_columns=feature_columns,
        best_model_name="temperature_echo",
    )
    fallback_metadata["runtime_versions"]["scikit_learn"] = _bump_version_component(
        str(fallback_metadata["runtime_versions"]["scikit_learn"]),
        component_index=1,
    )
    fallback_model_path, fallback_metadata_path = _write_prediction_artifact_run(
        isolated_project_dirs["processed"],
        training_run_id=fallback_run_id,
        artifact=fallback_artifact,
        metadata=fallback_metadata,
    )

    train_model._write_latest_training_run_manifest(
        run_id=latest_run_id,
        model_path=latest_model_path,
        metadata_path=latest_metadata_path,
    )

    prediction = train_model.predict_single_case(
        density_m3=1.0e20,
        temperature=12.0,
        confinement_time_s=1.0,
        temp_unit="keV",
        fuel_purity=None,
        energy_input_mj=None,
        pressure_pa=None,
        ip_ma=None,
        bt_t=None,
        r_m=None,
        a_m=None,
        kappa=None,
        ne_20=None,
        m_amu=None,
        pin_mw=None,
    )
    runtime = inference.load_prediction_runtime()

    assert fallback_model_path.exists()
    assert fallback_metadata_path.exists()
    assert prediction["predicted_neutron_yield"] == pytest.approx(12.0)
    assert prediction["model_name"] == "temperature_echo"
    assert runtime.metadata.training_run_id == fallback_run_id
    assert any("skipped 1 unusable candidate" in warning for warning in prediction["prediction_warnings"])
    assert any("scikit_learn minor version drift" in warning for warning in prediction["prediction_warnings"])


def test_predict_single_case_prefers_exact_runtime_match_over_newer_compatible_default_artifact(
    isolated_project_dirs: dict[str, Path],
) -> None:
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    exact_run_id = "exact_runtime_match"
    newer_drift_run_id = "newer_minor_drift"

    exact_artifact = _build_feature_echo_artifact(
        feature_name="confinement_time_s",
        training_run_id=exact_run_id,
        model_name="tau_echo",
        feature_columns=feature_columns,
    )
    exact_metadata = _build_artifact_metadata(
        model_path=isolated_project_dirs["processed"] / train_model.TRAINING_RUNS_DIRNAME / exact_run_id / "models" / train_model.TRAINING_MODEL_FILENAME,
        training_run_id=exact_run_id,
        feature_columns=feature_columns,
        best_model_name="tau_echo",
    )
    exact_metadata["created_at_utc"] = "2026-03-29T11:00:00Z"
    _write_prediction_artifact_run(
        isolated_project_dirs["processed"],
        training_run_id=exact_run_id,
        artifact=exact_artifact,
        metadata=exact_metadata,
    )

    newer_drift_artifact = _build_temperature_echo_artifact(
        training_run_id=newer_drift_run_id,
        feature_columns=feature_columns,
    )
    newer_drift_metadata = _build_artifact_metadata(
        model_path=isolated_project_dirs["processed"] / train_model.TRAINING_RUNS_DIRNAME / newer_drift_run_id / "models" / train_model.TRAINING_MODEL_FILENAME,
        training_run_id=newer_drift_run_id,
        feature_columns=feature_columns,
        best_model_name="temperature_echo",
    )
    newer_drift_metadata["created_at_utc"] = "2026-03-29T12:00:00Z"
    newer_drift_metadata["runtime_versions"]["scikit_learn"] = _bump_version_component(
        str(newer_drift_metadata["runtime_versions"]["scikit_learn"]),
        component_index=1,
    )
    _write_prediction_artifact_run(
        isolated_project_dirs["processed"],
        training_run_id=newer_drift_run_id,
        artifact=newer_drift_artifact,
        metadata=newer_drift_metadata,
    )

    prediction = train_model.predict_single_case(
        density_m3=1.0e20,
        temperature=12.0,
        confinement_time_s=1.0,
        temp_unit="keV",
        fuel_purity=None,
        energy_input_mj=None,
        pressure_pa=None,
        ip_ma=None,
        bt_t=None,
        r_m=None,
        a_m=None,
        kappa=None,
        ne_20=None,
        m_amu=None,
        pin_mw=None,
    )

    assert prediction["predicted_neutron_yield"] == pytest.approx(1.0)
    assert prediction["model_name"] == "tau_echo"
    assert any("selection mode 'best_compatibility'" in warning for warning in prediction["prediction_warnings"])
    assert any("exact_runtime_match" in warning and "newer_minor_drift" in warning for warning in prediction["prediction_warnings"])


def test_predict_single_case_default_selection_uses_metadata_created_at_not_run_directory_name(
    isolated_project_dirs: dict[str, Path],
) -> None:
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    older_run_id = "zzzz_run_name_sorts_late"
    newer_run_id = "aaaa_run_name_sorts_early"

    older_artifact = _build_feature_echo_artifact(
        feature_name="confinement_time_s",
        training_run_id=older_run_id,
        model_name="tau_echo",
        feature_columns=feature_columns,
    )
    older_metadata = _build_artifact_metadata(
        model_path=isolated_project_dirs["processed"] / train_model.TRAINING_RUNS_DIRNAME / older_run_id / "models" / train_model.TRAINING_MODEL_FILENAME,
        training_run_id=older_run_id,
        feature_columns=feature_columns,
        best_model_name="tau_echo",
    )
    older_metadata["created_at_utc"] = "2026-03-29T08:00:00Z"
    _write_prediction_artifact_run(
        isolated_project_dirs["processed"],
        training_run_id=older_run_id,
        artifact=older_artifact,
        metadata=older_metadata,
    )

    newer_artifact = _build_temperature_echo_artifact(
        training_run_id=newer_run_id,
        feature_columns=feature_columns,
    )
    newer_metadata = _build_artifact_metadata(
        model_path=isolated_project_dirs["processed"] / train_model.TRAINING_RUNS_DIRNAME / newer_run_id / "models" / train_model.TRAINING_MODEL_FILENAME,
        training_run_id=newer_run_id,
        feature_columns=feature_columns,
        best_model_name="temperature_echo",
    )
    newer_metadata["created_at_utc"] = "2026-03-29T13:00:00Z"
    _write_prediction_artifact_run(
        isolated_project_dirs["processed"],
        training_run_id=newer_run_id,
        artifact=newer_artifact,
        metadata=newer_metadata,
    )

    prediction = train_model.predict_single_case(
        density_m3=1.0e20,
        temperature=12.0,
        confinement_time_s=1.0,
        temp_unit="keV",
        fuel_purity=None,
        energy_input_mj=None,
        pressure_pa=None,
        ip_ma=None,
        bt_t=None,
        r_m=None,
        a_m=None,
        kappa=None,
        ne_20=None,
        m_amu=None,
        pin_mw=None,
    )

    assert prediction["predicted_neutron_yield"] == pytest.approx(12.0)
    assert prediction["model_name"] == "temperature_echo"


def test_prediction_artifact_listing_and_run_id_resolution_are_publicly_supported(
    isolated_project_dirs: dict[str, Path],
) -> None:
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    older_run_id = "listed_older_run"
    newer_run_id = "listed_newer_run"
    older_model_path, older_metadata_path = _write_prediction_artifact_run(
        isolated_project_dirs["processed"],
        training_run_id=older_run_id,
        artifact=_build_feature_echo_artifact(
            feature_name="confinement_time_s",
            training_run_id=older_run_id,
            model_name="tau_echo",
            feature_columns=feature_columns,
        ),
        metadata={
            **_build_artifact_metadata(
                model_path=isolated_project_dirs["processed"]
                / train_model.TRAINING_RUNS_DIRNAME
                / older_run_id
                / "models"
                / train_model.TRAINING_MODEL_FILENAME,
                training_run_id=older_run_id,
                feature_columns=feature_columns,
                best_model_name="tau_echo",
            ),
            "created_at_utc": "2026-03-29T09:00:00Z",
        },
    )
    _write_prediction_artifact_run(
        isolated_project_dirs["processed"],
        training_run_id=newer_run_id,
        artifact=_build_temperature_echo_artifact(
            training_run_id=newer_run_id,
            feature_columns=feature_columns,
        ),
        metadata={
            **_build_artifact_metadata(
                model_path=isolated_project_dirs["processed"]
                / train_model.TRAINING_RUNS_DIRNAME
                / newer_run_id
                / "models"
                / train_model.TRAINING_MODEL_FILENAME,
                training_run_id=newer_run_id,
                feature_columns=feature_columns,
                best_model_name="temperature_echo",
            ),
            "created_at_utc": "2026-03-29T12:00:00Z",
        },
    )

    listed_runs = inference.list_prediction_artifacts()
    resolved_model_path, resolved_metadata_path = inference.resolve_prediction_artifact_paths(
        training_run_id=older_run_id,
    )
    runtime = inference.load_prediction_runtime(training_run_id=older_run_id)

    assert [candidate.training_run_id for candidate in listed_runs] == [newer_run_id, older_run_id]
    assert resolved_model_path == older_model_path.resolve()
    assert resolved_metadata_path == older_metadata_path.resolve()
    assert runtime.metadata.training_run_id == older_run_id
    assert runtime.metadata.best_model_name == "tau_echo"


def test_default_artifact_selection_mode_can_prefer_newest_compatible_run(
    isolated_project_dirs: dict[str, Path],
) -> None:
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    current_versions = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "scikit_learn": train_model.sklearn.__version__,
        "joblib": joblib.__version__,
    }
    older_run_id = "exact_runtime_run"
    newer_run_id = "newer_minor_drift_run"
    _write_prediction_artifact_run(
        isolated_project_dirs["processed"],
        training_run_id=older_run_id,
        artifact=_build_feature_echo_artifact(
            feature_name="confinement_time_s",
            training_run_id=older_run_id,
            model_name="tau_echo",
            feature_columns=feature_columns,
        ),
        metadata={
            **_build_artifact_metadata(
                model_path=isolated_project_dirs["processed"]
                / train_model.TRAINING_RUNS_DIRNAME
                / older_run_id
                / "models"
                / train_model.TRAINING_MODEL_FILENAME,
                training_run_id=older_run_id,
                feature_columns=feature_columns,
                best_model_name="tau_echo",
            ),
            "created_at_utc": "2026-03-29T09:00:00Z",
            "runtime_versions": current_versions,
        },
    )
    _write_prediction_artifact_run(
        isolated_project_dirs["processed"],
        training_run_id=newer_run_id,
        artifact=_build_temperature_echo_artifact(
            training_run_id=newer_run_id,
            feature_columns=feature_columns,
        ),
        metadata={
            **_build_artifact_metadata(
                model_path=isolated_project_dirs["processed"]
                / train_model.TRAINING_RUNS_DIRNAME
                / newer_run_id
                / "models"
                / train_model.TRAINING_MODEL_FILENAME,
                training_run_id=newer_run_id,
                feature_columns=feature_columns,
                best_model_name="temperature_echo",
            ),
            "created_at_utc": "2026-03-29T12:00:00Z",
            "runtime_versions": {
                **current_versions,
                "joblib": _bump_version_component(current_versions["joblib"], component_index=1),
            },
        },
    )

    default_runtime = inference.load_prediction_runtime()
    newest_runtime = inference.load_prediction_runtime(
        default_artifact_selection=inference.DEFAULT_ARTIFACT_SELECTION_NEWEST_COMPATIBLE
    )

    assert default_runtime.metadata.training_run_id == older_run_id
    assert newest_runtime.metadata.training_run_id == newer_run_id
    assert any("best_compatibility" in warning for warning in default_runtime.load_warnings)
    assert any("newest_compatible" in warning for warning in newest_runtime.load_warnings)


def test_predict_single_case_defaults_missing_dataset_preparation_metadata(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "negative.joblib"
    metadata_path = tmp_path / "legacy_negative_metadata.json"
    metadata = _build_artifact_metadata(model_path=model_path)
    metadata.pop("dataset_preparation")
    joblib.dump(_build_negative_artifact(), model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    with pytest.warns(RuntimeWarning, match="clipped to 0.0"):
        prediction = train_model.predict_single_case(
            density_m3=1.0e20,
            temperature=12.0,
            confinement_time_s=1.0,
            temp_unit="keV",
            fuel_purity=None,
            energy_input_mj=None,
            pressure_pa=None,
            ip_ma=None,
            bt_t=None,
            r_m=None,
            a_m=None,
            kappa=None,
            ne_20=None,
            m_amu=None,
            pin_mw=None,
            model_path=model_path,
            metadata_path=metadata_path,
        )

    assert prediction["predicted_neutron_yield"] == 0.0


def test_predict_single_case_rejects_mismatched_model_and_metadata(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    dataset_path = features.create_synthetic_dataset(tmp_path / "synthetic_train.csv", n_rows=60, random_state=23)
    artifacts = train_model.train_models(dataset_path)
    metadata_path = Path(artifacts["metadata_path"])
    mismatched_metadata_path = metadata_path.parent / "tampered_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["training_run_id"] = "different_run"
    mismatched_metadata_path.write_text(json.dumps(metadata, indent=2))
    source_row = pd.read_csv(dataset_path).iloc[0]

    with pytest.raises(ValueError, match="incompatible with metadata"):
        train_model.predict_single_case(
            density_m3=float(source_row["fuel_density_m3"]),
            temperature=float(source_row["temperature_keV"]),
            confinement_time_s=float(source_row["confinement_time_s"]),
            temp_unit="keV",
            fuel_purity=None,
            energy_input_mj=None,
            pressure_pa=None,
            ip_ma=None,
            bt_t=None,
            r_m=None,
            a_m=None,
            kappa=None,
            ne_20=None,
            m_amu=None,
            pin_mw=None,
            model_path=artifacts["model_path"],
            metadata_path=mismatched_metadata_path,
        )


def test_predict_single_case_accepts_legacy_preprocessing_fingerprint_artifacts(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "legacy_temperature_echo.joblib"
    metadata_path = tmp_path / "legacy_temperature_echo_metadata.json"
    feature_columns = [
        "fuel_density_m3",
        "temperature_keV",
        "confinement_time_s",
        "triple_product",
    ]
    legacy_contract = features.build_preprocessing_contract(
        fingerprint_method=features.LEGACY_PREPROCESSING_LOGIC_FINGERPRINT_METHOD,
    )
    artifact = FusionFluxModelArtifact(
        FeatureEchoModel("temperature_keV"),
        schema_version=inference.ARTIFACT_SCHEMA_VERSION,
        training_run_id="legacy_preprocessing_contract_run",
        feature_columns=feature_columns,
        model_name="temperature_echo",
        preprocessing_contract=legacy_contract,
    )
    joblib.dump(artifact, model_path)
    metadata_path.write_text(
        json.dumps(
            _build_artifact_metadata(
                model_path=model_path,
                training_run_id="legacy_preprocessing_contract_run",
                feature_columns=feature_columns,
                best_model_name="temperature_echo",
                preprocessing_contract=legacy_contract,
            ),
            indent=2,
        )
    )

    prediction = inference.predict_single_case(
        density_m3=1.0e20,
        temperature=12.0,
        confinement_time_s=1.0,
        temp_unit="keV",
        fuel_purity=None,
        energy_input_mj=None,
        pressure_pa=None,
        ip_ma=None,
        bt_t=None,
        r_m=None,
        a_m=None,
        kappa=None,
        ne_20=None,
        m_amu=None,
        pin_mw=None,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    assert prediction["predicted_neutron_yield"] == pytest.approx(12.0)
    assert prediction["prediction_warnings"] == [] or any(
        "legacy bytecode-based preprocessing fingerprints" in warning
        for warning in prediction["prediction_warnings"]
    )


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


def test_grouped_time_series_training_and_batch_inference_share_prepared_representation(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    frame = _build_grouped_time_series_frame()
    dataset_path = _write_dataset(tmp_path, frame, "grouped_time_series.csv")
    artifacts = train_model.train_models(
        dataset_path,
        shot_prediction_cutoff_rows=3,
        generate_reports=False,
    )
    runtime = train_model.load_prediction_runtime(
        model_path=artifacts["model_path"],
        metadata_path=artifacts["metadata_path"],
    )
    training_representation = features.prepare_dataset(
        dataset_path,
        processed_output_path=tmp_path / "prepared_grouped_time_series.csv",
        shot_prediction_cutoff_rows=3,
    ).dataframe

    batch_result = train_model.predict_batch(pd.read_csv(dataset_path), runtime=runtime)
    saved_model = joblib.load(artifacts["model_path"])
    expected_predictions = saved_model.predict(
        features.align_to_feature_schema(training_representation, runtime.metadata.feature_columns)
    )

    assert len(batch_result.predictions) == frame["shot_id"].nunique()
    assert batch_result.predictions["shot_id"].tolist() == training_representation["shot_id"].tolist()
    assert batch_result.predictions["time_s"].tolist() == pytest.approx(training_representation["time_s"].tolist())
    assert batch_result.predictions["temperature_keV"].tolist() == pytest.approx(
        training_representation["temperature_keV"].tolist()
    )
    assert batch_result.predictions["predicted_neutron_yield"].tolist() == pytest.approx(
        expected_predictions.tolist()
    )


def test_predict_single_case_clips_negative_predictions(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "negative.joblib"
    metadata_path = tmp_path / "negative_metadata.json"
    joblib.dump(_build_negative_artifact(), model_path)
    metadata_path.write_text(json.dumps(_build_artifact_metadata(model_path=model_path), indent=2))

    with pytest.warns(RuntimeWarning, match="clipped to 0.0"):
        prediction = train_model.predict_single_case(
            density_m3=1.0e20,
            temperature=12.0,
            confinement_time_s=1.0,
            temp_unit="keV",
            fuel_purity=None,
            energy_input_mj=None,
            pressure_pa=None,
            ip_ma=None,
            bt_t=None,
            r_m=None,
            a_m=None,
            kappa=None,
            ne_20=None,
            m_amu=None,
            pin_mw=None,
            model_path=model_path,
            metadata_path=metadata_path,
        )

    assert prediction["predicted_neutron_yield"] == 0.0
    assert prediction["clipped_negative_prediction"] is True
    assert prediction["prediction_warnings"] == [
        "Model predicted a negative neutron yield; output was clipped to 0.0."
    ]


def test_raw_loaded_artifact_predict_clips_and_warns() -> None:
    artifact = _build_negative_artifact()
    frame = pd.DataFrame(
        [
            {
                "fuel_density_m3": 1.0e20,
                "temperature_keV": 12.0,
                "confinement_time_s": 1.0,
                "triple_product": 1.2e21,
            }
        ]
    )

    with pytest.warns(RuntimeWarning, match="clipped to 0.0"):
        predictions = artifact.predict(frame)

    assert predictions.tolist() == [0.0]
    assert artifact.last_prediction_info.clipped_negative_prediction is True
    assert artifact.last_prediction_info.prediction_warnings == (
        "Model predicted a negative neutron yield; output was clipped to 0.0.",
    )


def test_raw_loaded_artifact_predict_enforces_runtime_preprocessing_contract() -> None:
    artifact = _build_negative_artifact()
    artifact.fusionflux_preprocessing_contract["target_column"] = "tampered_target"
    frame = pd.DataFrame(
        [
            {
                "fuel_density_m3": 1.0e20,
                "temperature_keV": 12.0,
                "confinement_time_s": 1.0,
                "triple_product": 1.2e21,
            }
        ]
    )

    with pytest.raises(ValueError, match="different preprocessing contract"):
        artifact.predict(frame)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_sha256", "0" * 64),
        ("lawson_dt_ignition", features.LAWSON_DT_IGNITION * 1.01),
        ("supported_temperature_units", ["keV", "eV"]),
    ],
)
def test_raw_loaded_artifact_predict_detects_runtime_drift_for_stronger_contract_fields(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    artifact = _build_negative_artifact()
    frame = pd.DataFrame(
        [
            {
                "fuel_density_m3": 1.0e20,
                "temperature_keV": 12.0,
                "confinement_time_s": 1.0,
                "triple_product": 1.2e21,
            }
        ]
    )
    drifted_contract = json.loads(json.dumps(features.build_preprocessing_contract()))
    drifted_contract[field] = replacement
    monkeypatch.setattr(features, "build_preprocessing_contract", lambda: drifted_contract)

    with pytest.raises(ValueError, match="different preprocessing contract"):
        artifact.predict(frame)


def test_raw_loaded_artifact_predict_rechecks_runtime_contract_after_cache_warming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build_negative_artifact()
    frame = pd.DataFrame(
        [
            {
                "fuel_density_m3": 1.0e20,
                "temperature_keV": 12.0,
                "confinement_time_s": 1.0,
                "triple_product": 1.2e21,
            }
        ]
    )

    artifact.predict(frame)
    monkeypatch.setattr(features, "LAWSON_DT_IGNITION", features.LAWSON_DT_IGNITION * 1.01)

    with pytest.raises(ValueError, match="different preprocessing contract"):
        artifact.predict(frame)


def test_build_preprocessing_contract_does_not_require_source_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_no_source(*_args: object, **_kwargs: object) -> str:
        raise OSError("no source")

    monkeypatch.setattr(inspect, "getsource", raise_no_source)

    contract = features.build_preprocessing_contract()

    assert contract["sha256"]
    assert contract["source_sha256"]
    assert contract["source_fingerprint_method"] == features.PREPROCESSING_LOGIC_FINGERPRINT_METHOD


def test_committed_artifact_manifest_supports_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    manifest_path = processed_dir / train_model.LATEST_TRAINING_RUN_FILENAME

    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert not Path(manifest["model_path"]).is_absolute()
    assert not Path(manifest["metadata_path"]).is_absolute()

    source_model_path = (manifest_path.parent / manifest["model_path"]).resolve()
    source_metadata_path = (manifest_path.parent / manifest["metadata_path"]).resolve()
    assert source_model_path.exists()
    assert source_metadata_path.exists()

    relocated_root = tmp_path / "relocated_clone"
    relocated_processed_dir = relocated_root / "data" / "processed"
    relocated_processed_dir.mkdir(parents=True)
    shutil.copy2(manifest_path, relocated_processed_dir / train_model.LATEST_TRAINING_RUN_FILENAME)
    shutil.copytree(
        source_metadata_path.parent,
        relocated_processed_dir / "runs" / source_metadata_path.parent.name,
    )

    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", relocated_processed_dir)
    monkeypatch.setattr(config, "DATA_RAW_DIR", relocated_root / "data" / "raw")

    prediction = train_model.predict_single_case(
        density_m3=1.0e20,
        temperature=12.0,
        confinement_time_s=1.0,
        temp_unit="keV",
        fuel_purity=None,
        energy_input_mj=None,
        pressure_pa=None,
        ip_ma=None,
        bt_t=None,
        r_m=None,
        a_m=None,
        kappa=None,
        ne_20=None,
        m_amu=None,
        pin_mw=None,
    )
    resolved_model_path, resolved_metadata_path = inference._resolve_prediction_artifact_paths(None, None)
    metadata = _load_training_metadata_record(resolved_metadata_path)
    model = joblib.load(resolved_model_path)

    inference._ensure_artifact_compatibility(
        model,
        metadata,
        model_path=resolved_model_path,
        metadata_path=resolved_metadata_path,
    )
    assert np.isfinite(prediction["predicted_neutron_yield"])
