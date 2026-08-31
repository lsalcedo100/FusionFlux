"""Artifact schema, metadata parsing, and run-manifest persistence.

Defines the versioned prediction-artifact contract: the dataclasses describing a
saved training run, the strict JSON parsers/validators the inference loader relies
on, and the writers ``training`` uses to persist a run the loader can later
discover. Deliberately free of model-loading and prediction logic so both the
training and inference sides can depend on it without an import cycle.
"""

from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Optional, Union, cast

import joblib
import pandas as pd
import sklearn

import config
from storage import write_json_strict

from .artifact_model import FusionFluxModelArtifact
from .features import DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS

ARTIFACT_SCHEMA_VERSION = 3
TRAINING_METADATA_FILENAME = "training_metadata.json"
TRAINING_MODEL_FILENAME = "best_model.joblib"
LATEST_TRAINING_RUN_FILENAME = "latest_training_run.json"
ARTIFACT_RUNTIME_VERSION_FIELDS = ("python", "pandas", "scikit_learn", "joblib")
ARTIFACT_RUNTIME_PATCH_COMPATIBILITY_COMPONENTS = {
    "python": 2,
    "pandas": 2,
    "scikit_learn": 2,
    "joblib": 2,
}
ARTIFACT_RUNTIME_MINOR_COMPATIBILITY_COMPONENTS = {
    "python": 2,
    "pandas": 1,
    "scikit_learn": 1,
    "joblib": 1,
}
RUNTIME_COMPATIBILITY_EXACT = 0
RUNTIME_COMPATIBILITY_PATCH = 1
RUNTIME_COMPATIBILITY_MINOR = 2
DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY = "best_compatibility"
DEFAULT_ARTIFACT_SELECTION_NEWEST_COMPATIBLE = "newest_compatible"
SUPPORTED_DEFAULT_ARTIFACT_SELECTION_MODES = (
    DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
    DEFAULT_ARTIFACT_SELECTION_NEWEST_COMPATIBLE,
)
METADATA_RELATIVE_PATH_FIELDS = (
    ("raw_dataset_path",),
    ("processed_dataset_path",),
    ("artifact_run_directory",),
    ("dataset_source", "requested_dataset_path"),
    ("dataset_source", "resolved_dataset_path"),
    ("holdout_evaluation", "metrics_artifact_path"),
    ("holdout_evaluation", "prediction_artifact_path"),
    ("holdout_evaluation", "mismatch_artifact_path"),
    ("holdout_evaluation", "residual_plot_path"),
    ("saved_model", "path"),
    ("artifacts", "metrics_path"),
    ("artifacts", "prediction_path"),
    ("artifacts", "mismatch_path"),
    ("artifacts", "feature_importance_path"),
    ("artifacts", "residual_plot_path"),
    ("artifacts", "importance_plot_path"),
)


@dataclass(frozen=True)
class PredictionArtifactManifest:
    schema_version: int
    training_run_id: str
    model_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class TrainingArtifactMetadata:
    payload: dict[str, object]
    schema_version: int
    training_run_id: str
    feature_columns: list[str]
    best_model_name: str
    preprocessing: dict[str, object]
    runtime_versions: dict[str, str]
    assume_temperature_unit: str | None
    shot_prediction_cutoff_rows: int


@dataclass(frozen=True)
class LoadedPredictionArtifact:
    model: FusionFluxModelArtifact
    metadata: TrainingArtifactMetadata
    model_path: Path
    metadata_path: Path
    load_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionArtifactCandidate:
    model_path: Path
    metadata_path: Path
    training_run_id: str
    created_at_utc: datetime | None
    runtime_compatibility_rank: int


@dataclass(frozen=True)
class ResolvedPredictionArtifactSelection:
    model_path: Path
    metadata_path: Path
    training_run_id: str | None
    resolution_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BatchPredictionResult:
    predictions: pd.DataFrame | None
    output_path: Path | None
    row_count: int
    model_name: str
    training_run_id: str
    schema_version: int
    model_path: Path
    metadata_path: Path
    clipped_negative_prediction_count: int
    prediction_warnings: list[str]
    column_mapping: dict[str, str]


@dataclass(frozen=True)
class PredictionRuntime:
    loaded_artifact: LoadedPredictionArtifact
    default_assume_temperature_unit: str | None
    shot_prediction_cutoff_rows: int
    default_artifact_selection: str

    @property
    def model(self) -> FusionFluxModelArtifact:
        return self.loaded_artifact.model

    @property
    def metadata(self) -> TrainingArtifactMetadata:
        return self.loaded_artifact.metadata

    @property
    def model_path(self) -> Path:
        return self.loaded_artifact.model_path

    @property
    def metadata_path(self) -> Path:
        return self.loaded_artifact.metadata_path

    @property
    def load_warnings(self) -> tuple[str, ...]:
        return self.loaded_artifact.load_warnings


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _current_runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _extract_numeric_version_components(version: str, *, component_count: int) -> tuple[int, ...] | None:
    version_parts = re.findall(r"\d+", version)
    if len(version_parts) < component_count:
        return None
    return tuple(int(part) for part in version_parts[:component_count])


def _parse_artifact_created_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _artifact_candidate_recency_key(candidate: PredictionArtifactCandidate) -> float:
    if candidate.created_at_utc is not None:
        return candidate.created_at_utc.timestamp()
    model_mtime = candidate.model_path.stat().st_mtime if candidate.model_path.exists() else float("-inf")
    metadata_mtime = candidate.metadata_path.stat().st_mtime if candidate.metadata_path.exists() else float("-inf")
    return max(model_mtime, metadata_mtime)


def _describe_runtime_compatibility_rank(rank: int) -> str:
    if rank == RUNTIME_COMPATIBILITY_EXACT:
        return "exact"
    if rank == RUNTIME_COMPATIBILITY_PATCH:
        return "patch-drift"
    if rank == RUNTIME_COMPATIBILITY_MINOR:
        return "minor-drift"
    return f"rank-{rank}"


def _validate_default_artifact_selection_mode(selection_mode: str) -> str:
    if selection_mode not in SUPPORTED_DEFAULT_ARTIFACT_SELECTION_MODES:
        raise ValueError(
            "default_artifact_selection must be one of "
            f"{list(SUPPORTED_DEFAULT_ARTIFACT_SELECTION_MODES)}, got {selection_mode!r}."
        )
    return selection_mode


def _read_json_object(path: Path, *, object_name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except JSONDecodeError as exc:
        raise ValueError(f"{object_name} at {path} is not valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{object_name} at {path} must contain a JSON object.")
    return payload


def _require_string(payload: dict[str, object], key: str, *, path: Path, object_name: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{object_name} at {path} is missing a valid '{key}'.")
    return value


def _require_string_list(payload: dict[str, object], key: str, *, path: Path, object_name: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{object_name} at {path} is missing a valid '{key}' list.")
    return [str(item) for item in value]


def _require_string_mapping(payload: dict[str, object], key: str, *, path: Path, object_name: str) -> dict[str, str]:
    value = payload.get(key)
    if not isinstance(value, dict) or not all(isinstance(map_key, str) for map_key in value):
        raise ValueError(f"{object_name} at {path} is missing a valid '{key}' object.")
    normalized_mapping: dict[str, str] = {}
    for map_key, map_value in value.items():
        if not isinstance(map_value, str) or not map_value:
            raise ValueError(
                f"{object_name} at {path} must record non-empty string values in '{key}.{map_key}'."
            )
        normalized_mapping[str(map_key)] = map_value
    return normalized_mapping


def _require_object_mapping(payload: dict[str, object], key: str, *, path: Path, object_name: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{object_name} at {path} is missing a valid '{key}' object.")
    return value


def _get_nested_value(payload: dict[str, object], keys: tuple[str, ...]) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _set_nested_value(payload: dict[str, object], keys: tuple[str, ...], value: object) -> None:
    current = payload
    for key in keys[:-1]:
        current = cast(dict[str, object], current[key])
    current[keys[-1]] = value


def _relative_storage_path(path_value: str | Path, *, base_dir: Path) -> str:
    path = Path(path_value).expanduser().resolve()
    try:
        relative_path = os.path.relpath(path, start=base_dir.resolve())
    except ValueError:
        return str(path)
    return Path(relative_path).as_posix()


def _resolve_stored_path(path_value: str | Path | None, *, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _parse_prediction_manifest(payload: dict[str, object], *, manifest_path: Path) -> PredictionArtifactManifest:
    schema_version = payload.get("schema_version")
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Artifact manifest at {manifest_path} has schema {schema_version!r}; "
            f"expected version {ARTIFACT_SCHEMA_VERSION}."
        )

    training_run_id = _require_string(
        payload,
        "training_run_id",
        path=manifest_path,
        object_name="Artifact manifest",
    )
    model_path = _resolve_stored_path(
        _require_string(payload, "model_path", path=manifest_path, object_name="Artifact manifest"),
        base_dir=manifest_path.parent,
    )
    metadata_path = _resolve_stored_path(
        _require_string(payload, "metadata_path", path=manifest_path, object_name="Artifact manifest"),
        base_dir=manifest_path.parent,
    )
    assert model_path is not None
    assert metadata_path is not None
    return PredictionArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        training_run_id=training_run_id,
        model_path=model_path,
        metadata_path=metadata_path,
    )


def _parse_training_artifact_metadata(
    payload: dict[str, object],
    *,
    metadata_path: Path,
) -> TrainingArtifactMetadata:
    schema_version = payload.get("schema_version")
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Training metadata at {metadata_path} has schema {schema_version!r}; "
            f"expected version {ARTIFACT_SCHEMA_VERSION}."
        )

    training_run_id = _require_string(
        payload,
        "training_run_id",
        path=metadata_path,
        object_name="Training metadata",
    )
    feature_columns = _require_string_list(
        payload,
        "feature_columns",
        path=metadata_path,
        object_name="Training metadata",
    )
    best_model_name = _require_string(
        payload,
        "best_model_name",
        path=metadata_path,
        object_name="Training metadata",
    )
    preprocessing = _require_object_mapping(
        payload,
        "preprocessing",
        path=metadata_path,
        object_name="Training metadata",
    )
    runtime_versions = _require_string_mapping(
        payload,
        "runtime_versions",
        path=metadata_path,
        object_name="Training metadata",
    )
    dataset_preparation = payload.get("dataset_preparation")
    if dataset_preparation is None:
        dataset_preparation_mapping: dict[str, object] = {}
    elif not isinstance(dataset_preparation, dict):
        raise ValueError(f"Training metadata at {metadata_path} is missing a valid 'dataset_preparation' object.")
    else:
        dataset_preparation_mapping = dataset_preparation
    assume_temperature_unit = dataset_preparation_mapping.get("assume_temperature_unit")
    if assume_temperature_unit is not None and assume_temperature_unit not in {"keV", "eV", "K"}:
        raise ValueError(
            f"Training metadata at {metadata_path} contains an unsupported "
            f"dataset_preparation.assume_temperature_unit value."
        )
    shot_prediction_cutoff_rows = dataset_preparation_mapping.get("shot_prediction_cutoff_rows")
    if shot_prediction_cutoff_rows is None:
        normalized_shot_prediction_cutoff_rows = DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS
    elif isinstance(shot_prediction_cutoff_rows, bool) or not isinstance(shot_prediction_cutoff_rows, int):
        raise ValueError(
            f"Training metadata at {metadata_path} must record dataset_preparation.shot_prediction_cutoff_rows "
            "as a positive integer."
        )
    elif shot_prediction_cutoff_rows <= 0:
        raise ValueError(
            f"Training metadata at {metadata_path} must record dataset_preparation.shot_prediction_cutoff_rows "
            "as a positive integer."
        )
    else:
        normalized_shot_prediction_cutoff_rows = shot_prediction_cutoff_rows
    return TrainingArtifactMetadata(
        payload=payload,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        training_run_id=training_run_id,
        feature_columns=feature_columns,
        best_model_name=best_model_name,
        preprocessing=preprocessing,
        runtime_versions=runtime_versions,
        assume_temperature_unit=cast(Optional[str], assume_temperature_unit),
        shot_prediction_cutoff_rows=normalized_shot_prediction_cutoff_rows,
    )


def _validate_saved_model_metadata_paths(
    metadata: dict[str, object],
    *,
    metadata_path: Path,
    model_path: Path,
) -> None:
    saved_model = metadata.get("saved_model")
    if not isinstance(saved_model, dict):
        raise ValueError(f"Training metadata at {metadata_path} is missing a valid 'saved_model' object.")
    artifact_type = saved_model.get("artifact_type")
    if artifact_type != "FusionFluxModelArtifact":
        raise ValueError(
            f"Training metadata at {metadata_path} must record saved_model.artifact_type as "
            "'FusionFluxModelArtifact'."
        )
    recorded_model_path = _resolve_stored_path(
        cast(Union[str, Path, None], saved_model.get("path")),
        base_dir=metadata_path.parent,
    )
    if recorded_model_path is None:
        raise ValueError(f"Training metadata at {metadata_path} is missing a valid saved_model.path.")
    if recorded_model_path != model_path:
        raise ValueError(
            f"Training metadata at {metadata_path} points to {recorded_model_path}, "
            f"but inference requested {model_path}. Use matching model and metadata files."
        )


def _validate_runtime_versions_for_loading(
    metadata: TrainingArtifactMetadata,
    *,
    metadata_path: Path,
    allow_compatible_drift: bool = False,
) -> tuple[int, list[str]]:
    current_versions = _current_runtime_versions()
    warnings: list[str] = []
    max_runtime_compatibility_rank = RUNTIME_COMPATIBILITY_EXACT
    for library_name in ARTIFACT_RUNTIME_VERSION_FIELDS:
        current_version = current_versions[library_name]
        recorded_version = metadata.runtime_versions.get(library_name)
        if recorded_version is None:
            raise ValueError(
                f"Training metadata at {metadata_path} is missing runtime_versions.{library_name}; "
                "cannot safely load the saved model."
            )
        if recorded_version == current_version:
            continue
        if not allow_compatible_drift:
            raise ValueError(
                f"Saved model runtime version mismatch for {library_name}: "
                f"artifact was trained with {recorded_version}, current runtime is {current_version}. "
                "Retrain the model in this environment or load it with matching library versions."
            )
        patch_component_count = ARTIFACT_RUNTIME_PATCH_COMPATIBILITY_COMPONENTS[library_name]
        patch_recorded_key = _extract_numeric_version_components(recorded_version, component_count=patch_component_count)
        patch_current_key = _extract_numeric_version_components(current_version, component_count=patch_component_count)
        if patch_recorded_key is not None and patch_current_key is not None and patch_recorded_key == patch_current_key:
            max_runtime_compatibility_rank = max(max_runtime_compatibility_rank, RUNTIME_COMPATIBILITY_PATCH)
            warnings.append(
                f"Loaded default artifact despite {library_name} patch version drift "
                f"({recorded_version} -> {current_version}); compatibility is not guaranteed."
            )
            continue
        minor_component_count = ARTIFACT_RUNTIME_MINOR_COMPATIBILITY_COMPONENTS[library_name]
        minor_recorded_key = _extract_numeric_version_components(recorded_version, component_count=minor_component_count)
        minor_current_key = _extract_numeric_version_components(current_version, component_count=minor_component_count)
        if minor_recorded_key is None or minor_current_key is None or minor_recorded_key != minor_current_key:
            raise ValueError(
                f"Saved model runtime version mismatch for {library_name}: "
                f"artifact was trained with {recorded_version}, current runtime is {current_version}. "
                "Retrain the model in this environment or load it with matching library versions."
            )
        max_runtime_compatibility_rank = max(max_runtime_compatibility_rank, RUNTIME_COMPATIBILITY_MINOR)
        warnings.append(
            f"Loaded default artifact despite {library_name} minor version drift "
            f"({recorded_version} -> {current_version}); compatibility is not guaranteed."
        )
    return max_runtime_compatibility_rank, warnings


# --- Artifact writers (used by training.py to persist runs the loader reads) ---


def _serialize_training_metadata_paths(metadata: dict[str, object], *, metadata_path: Path) -> dict[str, object]:
    serialized_metadata = cast(dict[str, object], json.loads(json.dumps(metadata)))
    run_dir = metadata_path.parent.resolve()
    for field in METADATA_RELATIVE_PATH_FIELDS:
        value = _get_nested_value(serialized_metadata, field)
        if value is None:
            continue
        _set_nested_value(
            serialized_metadata,
            field,
            _relative_storage_path(cast(Union[str, Path], value), base_dir=run_dir),
        )
    return serialized_metadata


def _resolve_training_metadata_paths(metadata: dict[str, object], *, metadata_path: Path) -> dict[str, object]:
    resolved_metadata = cast(dict[str, object], json.loads(json.dumps(metadata)))
    run_dir = metadata_path.parent.resolve()
    for field in METADATA_RELATIVE_PATH_FIELDS:
        value = _get_nested_value(resolved_metadata, field)
        if value is None:
            continue
        resolved_path = _resolve_stored_path(cast(Union[str, Path], value), base_dir=run_dir)
        _set_nested_value(resolved_metadata, field, str(resolved_path))
    return resolved_metadata


def _write_latest_training_run_manifest(
    *,
    run_id: str,
    model_path: Path,
    metadata_path: Path,
) -> Path:
    manifest_path = config.get_data_processed_dir() / LATEST_TRAINING_RUN_FILENAME
    write_json_strict(
        manifest_path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "training_run_id": run_id,
            "updated_at_utc": _timestamp_utc(),
            "model_path": _relative_storage_path(model_path, base_dir=manifest_path.parent),
            "metadata_path": _relative_storage_path(metadata_path, base_dir=manifest_path.parent),
        },
    )
    return manifest_path
