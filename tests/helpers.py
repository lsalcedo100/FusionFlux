from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

import train_model
from config import ORIGINAL_ROW_INDEX_COLUMN, RAW_CSV_ROW_NUMBER_COLUMN
from neutron_yield import features, inference
from neutron_yield.artifact_model import FusionFluxModelArtifact


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
) -> dict[str, Any]:
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
            "scikit_learn": sklearn.__version__,
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
