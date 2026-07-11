from __future__ import annotations

import json
import os
import platform
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Optional, Union, cast

import joblib
import numpy as np
import pandas as pd
import sklearn

import config
from artifact_model import FusionFluxModelArtifact
from config import (
    GROUP_COLUMN,
    ORIGINAL_ROW_INDEX_COLUMN,
    RAW_CSV_ROW_NUMBER_COLUMN,
    TARGET_LOG_COLUMN,
)
from features import (
    DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS,
    OPTIONAL_PHYSICS_COLUMNS,
    add_source_identity_columns,
    align_to_feature_schema,
    describe_preprocessing_contract_differences,
    ensure_project_directories,
    prepare_model_frame,
    preprocessing_contract_matches,
    resolve_column_mapping,
)
from lawson import calculate_lawson_status, to_kev
from storage import atomic_output_path, ensure_parent_directory, write_json_strict
from validation import validate_physics_inputs

ARTIFACT_SCHEMA_VERSION = 3
TRAINING_METADATA_FILENAME = "training_metadata.json"
TRAINING_MODEL_FILENAME = "best_model.joblib"
LATEST_TRAINING_RUN_FILENAME = "latest_training_run.json"
BATCH_PREDICTION_CSV_CHUNK_ROWS = 20000
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
    is_manifest_default: bool = False


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


def _write_json_strict(path: Path, payload: dict[str, object]) -> None:
    write_json_strict(path, payload)


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


def _ensure_parent_directory(path: Path) -> None:
    ensure_parent_directory(path)


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
    _write_json_strict(
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


def _discover_prediction_artifact_candidate_paths() -> tuple[list[tuple[Path, Path, bool]], list[str]]:
    manifest_path = config.get_data_processed_dir() / LATEST_TRAINING_RUN_FILENAME
    candidate_paths: list[tuple[Path, Path, bool]] = []
    discovery_failures: list[str] = []
    seen_candidates: set[tuple[Path, Path]] = set()

    if manifest_path.exists():
        try:
            manifest = _parse_prediction_manifest(
                _read_json_object(manifest_path, object_name="Artifact manifest"),
                manifest_path=manifest_path,
            )
        except Exception as exc:
            discovery_failures.append(f"Latest artifact manifest could not be used: {exc}")
        else:
            manifest_candidate = (manifest.model_path.resolve(), manifest.metadata_path.resolve())
            candidate_paths.append((manifest_candidate[0], manifest_candidate[1], True))
            seen_candidates.add(manifest_candidate)

    runs_dir = config.get_data_processed_dir() / "runs"
    if runs_dir.exists():
        for run_dir in sorted(
            (path for path in runs_dir.iterdir() if path.is_dir() and not path.name.startswith(".")),
            key=lambda path: path.name,
            reverse=True,
        ):
            candidate = (
                (run_dir / "models" / TRAINING_MODEL_FILENAME).resolve(),
                (run_dir / TRAINING_METADATA_FILENAME).resolve(),
            )
            if candidate in seen_candidates:
                continue
            if not candidate[0].exists() or not candidate[1].exists():
                continue
            candidate_paths.append((candidate[0], candidate[1], False))
            seen_candidates.add(candidate)

    return candidate_paths, discovery_failures


def _inspect_available_prediction_artifact_candidates() -> tuple[list[PredictionArtifactCandidate], list[str]]:
    candidate_paths, discovery_failures = _discover_prediction_artifact_candidate_paths()
    attempted_failures = list(discovery_failures)
    inspected_candidates: list[PredictionArtifactCandidate] = []
    for resolved_model_path, resolved_metadata_path, is_manifest_default in candidate_paths:
        try:
            inspected_candidates.append(
                _inspect_prediction_artifact_candidate(
                    model_path=resolved_model_path,
                    metadata_path=resolved_metadata_path,
                    is_manifest_default=is_manifest_default,
                )
            )
        except Exception as exc:
            attempted_failures.append(
                f"Skipped default artifact candidate {resolved_metadata_path.parent}: {exc}"
            )
    inspected_candidates.sort(
        key=lambda candidate: (
            candidate.runtime_compatibility_rank,
            -_artifact_candidate_recency_key(candidate),
        )
    )
    return inspected_candidates, attempted_failures


def _build_default_artifact_selection_warnings(
    candidate: PredictionArtifactCandidate,
    inspected_candidates: list[PredictionArtifactCandidate],
    attempted_failures: list[str],
    *,
    selection_mode: str,
) -> tuple[str, ...]:
    more_recent_lower_compatibility_candidates = [
        other
        for other in inspected_candidates
        if _artifact_candidate_recency_key(other) > _artifact_candidate_recency_key(candidate)
        and other.runtime_compatibility_rank > candidate.runtime_compatibility_rank
    ]
    older_higher_compatibility_candidates = [
        other
        for other in inspected_candidates
        if _artifact_candidate_recency_key(other) < _artifact_candidate_recency_key(candidate)
        and other.runtime_compatibility_rank < candidate.runtime_compatibility_rank
    ]
    selection_warnings: list[str] = []
    if attempted_failures:
        selection_warnings.append(
            "Default artifact loading skipped "
            f"{len(attempted_failures)} unusable candidate(s) before selecting training run "
            f"{candidate.training_run_id}."
        )
    if (
        selection_mode == DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY
        and more_recent_lower_compatibility_candidates
    ):
        preferred_over_candidate = max(
            more_recent_lower_compatibility_candidates,
            key=_artifact_candidate_recency_key,
        )
        selection_warnings.append(
            "Default artifact loading used selection mode "
            f"{selection_mode!r} and preferred "
            f"training run {candidate.training_run_id} over newer run "
            f"{preferred_over_candidate.training_run_id} because it matched the current runtime more closely "
            f"({_describe_runtime_compatibility_rank(candidate.runtime_compatibility_rank)} vs "
            f"{_describe_runtime_compatibility_rank(preferred_over_candidate.runtime_compatibility_rank)})."
        )
    if (
        selection_mode == DEFAULT_ARTIFACT_SELECTION_NEWEST_COMPATIBLE
        and older_higher_compatibility_candidates
        and candidate.runtime_compatibility_rank != RUNTIME_COMPATIBILITY_EXACT
    ):
        more_compatible_candidate = min(
            older_higher_compatibility_candidates,
            key=lambda other: other.runtime_compatibility_rank,
        )
        selection_warnings.append(
            "Default artifact loading used selection mode "
            f"{selection_mode!r} and preferred newer run {candidate.training_run_id} over older run "
            f"{more_compatible_candidate.training_run_id} despite looser runtime matching "
            f"({_describe_runtime_compatibility_rank(candidate.runtime_compatibility_rank)} vs "
            f"{_describe_runtime_compatibility_rank(more_compatible_candidate.runtime_compatibility_rank)})."
        )
    return tuple(selection_warnings)


def list_prediction_artifacts() -> list[PredictionArtifactCandidate]:
    ensure_project_directories()
    inspected_candidates, _ = _inspect_available_prediction_artifact_candidates()
    return sorted(
        inspected_candidates,
        key=_artifact_candidate_recency_key,
        reverse=True,
    )


def _resolve_prediction_artifact_selection(
    model_path: str | Path | None,
    metadata_path: str | Path | None,
    *,
    training_run_id: str | None = None,
    default_artifact_selection: str = DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
) -> ResolvedPredictionArtifactSelection:
    selection_mode = _validate_default_artifact_selection_mode(default_artifact_selection)
    if training_run_id is not None and (model_path is not None or metadata_path is not None):
        raise ValueError(
            "Pass either a training_run_id or explicit model/metadata paths, not both."
        )
    if model_path is not None or metadata_path is not None:
        resolved_model_path = _resolve_stored_path(model_path, base_dir=Path.cwd()) if model_path is not None else None
        resolved_metadata_path = (
            _resolve_stored_path(metadata_path, base_dir=Path.cwd()) if metadata_path is not None else None
        )
        if resolved_metadata_path is None:
            assert resolved_model_path is not None
            resolved_metadata_path = resolved_model_path.parent.parent / TRAINING_METADATA_FILENAME
        if resolved_model_path is None:
            assert resolved_metadata_path is not None
            resolved_model_path = resolved_metadata_path.parent / "models" / TRAINING_MODEL_FILENAME
        return ResolvedPredictionArtifactSelection(
            model_path=resolved_model_path.expanduser().resolve(),
            metadata_path=resolved_metadata_path.expanduser().resolve(),
            training_run_id=training_run_id,
        )

    inspected_candidates, attempted_failures = _inspect_available_prediction_artifact_candidates()
    if not inspected_candidates:
        if attempted_failures:
            failure_excerpt = " ".join(attempted_failures[:3])
            raise ValueError(
                "No usable training artifacts were found for default prediction. "
                f"{failure_excerpt}"
            )
        raise FileNotFoundError(
            "No training artifacts were found. Train a model first or pass both --model-path and --metadata-path."
        )

    if training_run_id is not None:
        for candidate in inspected_candidates:
            if candidate.training_run_id == training_run_id:
                return ResolvedPredictionArtifactSelection(
                    model_path=candidate.model_path,
                    metadata_path=candidate.metadata_path,
                    training_run_id=candidate.training_run_id,
                )
        raise FileNotFoundError(
            f"No training artifact run was found for training_run_id={training_run_id!r}."
        )

    if selection_mode == DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY:
        selected_candidate = min(
            inspected_candidates,
            key=lambda candidate: (
                candidate.runtime_compatibility_rank,
                -_artifact_candidate_recency_key(candidate),
            ),
        )
    else:
        selected_candidate = max(
            inspected_candidates,
            key=lambda candidate: (
                _artifact_candidate_recency_key(candidate),
                -candidate.runtime_compatibility_rank,
            ),
        )
    return ResolvedPredictionArtifactSelection(
        model_path=selected_candidate.model_path,
        metadata_path=selected_candidate.metadata_path,
        training_run_id=selected_candidate.training_run_id,
        resolution_warnings=_build_default_artifact_selection_warnings(
            selected_candidate,
            inspected_candidates,
            attempted_failures,
            selection_mode=selection_mode,
        ),
    )


def _resolve_prediction_artifact_paths(
    model_path: str | Path | None,
    metadata_path: str | Path | None,
    *,
    training_run_id: str | None = None,
    default_artifact_selection: str = DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
) -> tuple[Path, Path]:
    selection = _resolve_prediction_artifact_selection(
        model_path,
        metadata_path,
        training_run_id=training_run_id,
        default_artifact_selection=default_artifact_selection,
    )
    return selection.model_path, selection.metadata_path


def resolve_prediction_artifact_paths(
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    *,
    training_run_id: str | None = None,
    default_artifact_selection: str = DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
) -> tuple[Path, Path]:
    return _resolve_prediction_artifact_paths(
        model_path,
        metadata_path,
        training_run_id=training_run_id,
        default_artifact_selection=default_artifact_selection,
    )


def _ensure_artifact_compatibility(
    model: FusionFluxModelArtifact,
    metadata: TrainingArtifactMetadata,
    *,
    model_path: Path,
    metadata_path: Path,
) -> None:
    model_schema_version = getattr(model, "fusionflux_schema_version", None)
    model_training_run_id = getattr(model, "fusionflux_training_run_id", None)
    model_feature_columns = list(getattr(model, "fusionflux_feature_columns", ()))
    model_name = getattr(model, "fusionflux_model_name", None)
    model_preprocessing_contract = getattr(model, "fusionflux_preprocessing_contract", None)
    if (
        model_schema_version != ARTIFACT_SCHEMA_VERSION
        or model_training_run_id != metadata.training_run_id
        or model_feature_columns != metadata.feature_columns
        or model_name != metadata.best_model_name
    ):
        raise ValueError(
            f"Model artifact {model_path} is incompatible with metadata {metadata_path}. "
            "Use model and metadata files from the same training run."
        )

    if not isinstance(model_preprocessing_contract, dict):
        raise ValueError(
            f"Model artifact {model_path} is missing its embedded preprocessing contract metadata."
        )
    if not preprocessing_contract_matches(model_preprocessing_contract, metadata.preprocessing):
        differing_fields = describe_preprocessing_contract_differences(
            model_preprocessing_contract,
            metadata.preprocessing,
        )
        difference_suffix = f" Mismatched fields: {', '.join(differing_fields)}." if differing_fields else ""
        raise ValueError(
            f"Model artifact {model_path} is incompatible with metadata {metadata_path}. "
            "Embedded preprocessing contract does not match the metadata contract."
            f"{difference_suffix}"
        )


def _inspect_prediction_artifact_candidate(
    *,
    model_path: Path,
    metadata_path: Path,
    is_manifest_default: bool = False,
) -> PredictionArtifactCandidate:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    raw_metadata = _read_json_object(metadata_path, object_name="Training metadata")
    resolved_metadata = _resolve_training_metadata_paths(raw_metadata, metadata_path=metadata_path)
    metadata_record = _parse_training_artifact_metadata(resolved_metadata, metadata_path=metadata_path)
    _validate_saved_model_metadata_paths(
        resolved_metadata,
        metadata_path=metadata_path,
        model_path=model_path,
    )
    runtime_compatibility_rank, _ = _validate_runtime_versions_for_loading(
        metadata_record,
        metadata_path=metadata_path,
        allow_compatible_drift=True,
    )
    return PredictionArtifactCandidate(
        model_path=model_path,
        metadata_path=metadata_path,
        training_run_id=metadata_record.training_run_id,
        created_at_utc=_parse_artifact_created_at(metadata_record.payload.get("created_at_utc")),
        runtime_compatibility_rank=runtime_compatibility_rank,
        is_manifest_default=is_manifest_default,
    )


def _load_prediction_artifact(
    model_path: str | Path | None,
    metadata_path: str | Path | None,
    *,
    training_run_id: str | None = None,
    default_artifact_selection: str = DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
) -> LoadedPredictionArtifact:
    selection_mode = _validate_default_artifact_selection_mode(default_artifact_selection)

    def _load_prediction_artifact_from_resolved_paths(
        resolved_model_path: Path,
        resolved_metadata_path: Path,
        *,
        allow_compatible_runtime_drift: bool,
    ) -> LoadedPredictionArtifact:
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Model not found: {resolved_model_path}")
        if not resolved_metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {resolved_metadata_path}")

        raw_metadata = _read_json_object(resolved_metadata_path, object_name="Training metadata")
        metadata = _resolve_training_metadata_paths(
            raw_metadata,
            metadata_path=resolved_metadata_path,
        )
        metadata_record = _parse_training_artifact_metadata(metadata, metadata_path=resolved_metadata_path)
        _validate_saved_model_metadata_paths(
            metadata,
            metadata_path=resolved_metadata_path,
            model_path=resolved_model_path,
        )
        _, runtime_warnings = _validate_runtime_versions_for_loading(
            metadata_record,
            metadata_path=resolved_metadata_path,
            allow_compatible_drift=allow_compatible_runtime_drift,
        )
        load_warnings = tuple(runtime_warnings)

        try:
            loaded_model = joblib.load(resolved_model_path)
        except Exception as exc:
            raise ValueError(
                f"Failed to deserialize model artifact at {resolved_model_path}. "
                "The file may be corrupted or incompatible with this runtime."
            ) from exc

        model = cast(FusionFluxModelArtifact, loaded_model)
        _ensure_artifact_compatibility(
            model,
            metadata_record,
            model_path=resolved_model_path,
            metadata_path=resolved_metadata_path,
        )
        preprocessing_warnings = model.validate_runtime_preprocessing()
        return LoadedPredictionArtifact(
            model=model,
            metadata=metadata_record,
            model_path=resolved_model_path,
            metadata_path=resolved_metadata_path,
            load_warnings=(*load_warnings, *preprocessing_warnings),
        )

    ensure_project_directories()
    if model_path is not None or metadata_path is not None or training_run_id is not None:
        # Explicit artifact paths stay strict so callers never silently fall back
        # to a different training run than the one they requested.
        resolved_model_path, resolved_metadata_path = _resolve_prediction_artifact_paths(
            model_path,
            metadata_path,
            training_run_id=training_run_id,
            default_artifact_selection=selection_mode,
        )
        return _load_prediction_artifact_from_resolved_paths(
            resolved_model_path,
            resolved_metadata_path,
            allow_compatible_runtime_drift=False,
        )

    inspected_candidates, attempted_failures = _inspect_available_prediction_artifact_candidates()
    if not inspected_candidates:
        if attempted_failures:
            failure_excerpt = " ".join(attempted_failures[:3])
            raise ValueError(
                "No usable training artifacts were found for default prediction. "
                f"{failure_excerpt}"
            )
        raise FileNotFoundError(
            "No training artifacts were found. Train a model first or pass both --model-path and --metadata-path."
        )

    if selection_mode == DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY:
        candidate_order = sorted(
            inspected_candidates,
            key=lambda candidate: (
                candidate.runtime_compatibility_rank,
                -_artifact_candidate_recency_key(candidate),
            ),
        )
    else:
        candidate_order = sorted(
            inspected_candidates,
            key=lambda candidate: (
                -_artifact_candidate_recency_key(candidate),
                candidate.runtime_compatibility_rank,
            ),
        )

    for candidate in candidate_order:
        try:
            loaded_artifact = _load_prediction_artifact_from_resolved_paths(
                candidate.model_path,
                candidate.metadata_path,
                allow_compatible_runtime_drift=True,
            )
        except Exception as exc:
            attempted_failures.append(
                f"Skipped default artifact candidate {candidate.metadata_path.parent}: {exc}"
            )
            continue
        selection_warnings = _build_default_artifact_selection_warnings(
            candidate,
            inspected_candidates,
            attempted_failures,
            selection_mode=selection_mode,
        )
        if selection_warnings:
            return LoadedPredictionArtifact(
                model=loaded_artifact.model,
                metadata=loaded_artifact.metadata,
                model_path=loaded_artifact.model_path,
                metadata_path=loaded_artifact.metadata_path,
                load_warnings=(
                    *selection_warnings,
                    *loaded_artifact.load_warnings,
                ),
            )
        return loaded_artifact

    failure_excerpt = " ".join(attempted_failures[:3])
    raise ValueError(
        "No usable training artifacts were found for default prediction. "
        f"{failure_excerpt}"
    )


def load_prediction_runtime(
    *,
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    training_run_id: str | None = None,
    default_artifact_selection: str = DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
) -> PredictionRuntime:
    loaded_artifact = _load_prediction_artifact(
        model_path,
        metadata_path,
        training_run_id=training_run_id,
        default_artifact_selection=default_artifact_selection,
    )
    return PredictionRuntime(
        loaded_artifact=loaded_artifact,
        default_assume_temperature_unit=loaded_artifact.metadata.assume_temperature_unit,
        shot_prediction_cutoff_rows=loaded_artifact.metadata.shot_prediction_cutoff_rows,
        default_artifact_selection=_validate_default_artifact_selection_mode(default_artifact_selection),
    )


def _resolve_prediction_runtime(
    runtime: PredictionRuntime | None,
    *,
    model_path: str | Path | None,
    metadata_path: str | Path | None,
    training_run_id: str | None,
    default_artifact_selection: str,
) -> PredictionRuntime:
    if runtime is not None:
        if model_path is not None or metadata_path is not None or training_run_id is not None:
            raise ValueError(
                "Pass either a reusable prediction runtime or an artifact selector, not both."
            )
        return runtime
    return load_prediction_runtime(
        model_path=model_path,
        metadata_path=metadata_path,
        training_run_id=training_run_id,
        default_artifact_selection=default_artifact_selection,
    )


def _ensure_prediction_row_identity_columns(
    frame: pd.DataFrame,
    *,
    start_index: int = 0,
) -> pd.DataFrame:
    if ORIGINAL_ROW_INDEX_COLUMN in frame.columns and RAW_CSV_ROW_NUMBER_COLUMN in frame.columns:
        return frame.copy()
    if ORIGINAL_ROW_INDEX_COLUMN in frame.columns or RAW_CSV_ROW_NUMBER_COLUMN in frame.columns:
        raise ValueError(
            "Batch prediction input must provide both row identity columns together or omit both."
        )
    return add_source_identity_columns(frame, start_index=start_index)


def _prepare_batch_inference_frame(
    frame: pd.DataFrame,
    *,
    runtime: PredictionRuntime,
    assume_temperature_unit: str | None,
    start_index: int = 0,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if frame.empty:
        raise ValueError("Batch prediction input must contain at least one row.")

    resolved_assume_temperature_unit = (
        assume_temperature_unit
        if assume_temperature_unit is not None
        else runtime.default_assume_temperature_unit
    )
    prepared_frame = _ensure_prediction_row_identity_columns(
        frame,
        start_index=start_index,
    )
    prepared_model_frame = prepare_model_frame(
        prepared_frame,
        assume_temperature_unit=resolved_assume_temperature_unit,
        shot_prediction_cutoff_rows=runtime.shot_prediction_cutoff_rows,
        require_target=False,
        deduplicate_rows=False,
    )
    return prepared_model_frame.dataframe, prepared_model_frame.column_mapping


def _default_batch_prediction_output_path(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    return input_path.with_name(f"{input_path.stem}_predictions.csv")


def _canonicalized_batch_input_columns(input_path: Path) -> set[str]:
    header_frame = pd.read_csv(input_path, nrows=0)
    rename_map = resolve_column_mapping(header_frame)
    return (set(header_frame.columns) - set(rename_map.keys())) | set(rename_map.values())


def _can_stream_batch_prediction_csv(input_path: Path) -> bool:
    canonical_columns = _canonicalized_batch_input_columns(input_path)
    return not (
        GROUP_COLUMN in canonical_columns
        and ("time_s" in canonical_columns or "time_ms" in canonical_columns)
    )


def _build_prediction_output_frame(
    prepared_frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    prediction_runtime: PredictionRuntime,
) -> pd.DataFrame:
    prediction_frame = prepared_frame.copy()
    if TARGET_LOG_COLUMN in prediction_frame.columns:
        prediction_frame = prediction_frame.drop(columns=[TARGET_LOG_COLUMN])
    prediction_frame["predicted_neutron_yield"] = predictions
    prediction_frame["artifact_training_run_id"] = prediction_runtime.metadata.training_run_id
    prediction_frame["artifact_model_name"] = prediction_runtime.metadata.best_model_name
    prediction_frame["artifact_schema_version"] = prediction_runtime.metadata.schema_version
    prediction_frame["artifact_model_path"] = str(prediction_runtime.model_path)
    prediction_frame["artifact_metadata_path"] = str(prediction_runtime.metadata_path)
    created_at_utc = prediction_runtime.metadata.payload.get("created_at_utc")
    prediction_frame["artifact_created_at_utc"] = created_at_utc if isinstance(created_at_utc, str) else np.nan
    return prediction_frame


def _predict_prepared_batch_frame(
    prepared_frame: pd.DataFrame,
    *,
    prediction_runtime: PredictionRuntime,
) -> tuple[pd.DataFrame, int, list[str]]:
    predictions, prediction_info = prediction_runtime.model.predict_with_info(
        align_to_feature_schema(prepared_frame, prediction_runtime.metadata.feature_columns)
    )
    return (
        _build_prediction_output_frame(
            prepared_frame,
            predictions,
            prediction_runtime=prediction_runtime,
        ),
        prediction_info.clipped_count,
        list(prediction_info.prediction_warnings),
    )


def _append_prediction_warning(prediction_warnings: list[str], warning: str) -> None:
    if warning not in prediction_warnings:
        prediction_warnings.append(warning)


def _read_batch_prediction_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


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
    training_run_id: str | None = None,
    runtime: PredictionRuntime | None = None,
    default_artifact_selection: str = DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
) -> dict:
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
    prediction_runtime = _resolve_prediction_runtime(
        runtime,
        model_path=model_path,
        metadata_path=metadata_path,
        training_run_id=training_run_id,
        default_artifact_selection=default_artifact_selection,
    )

    row = {
        "fuel_density_m3": validated_inputs["fuel_density_m3"],
        "temperature_keV": validated_inputs["temperature_keV"],
        "confinement_time_s": validated_inputs["confinement_time_s"],
    }
    for column in OPTIONAL_PHYSICS_COLUMNS:
        value = validated_inputs[column]
        row[column] = np.nan if value is None else value

    inference_df = prepare_model_frame(
        pd.DataFrame([row]),
        assume_temperature_unit=prediction_runtime.default_assume_temperature_unit,
        shot_prediction_cutoff_rows=prediction_runtime.shot_prediction_cutoff_rows,
        require_target=False,
    ).dataframe

    feature_columns = prediction_runtime.metadata.feature_columns
    predicted_yield, prediction_info = prediction_runtime.model.predict_with_info(
        align_to_feature_schema(inference_df, feature_columns)
    )
    prediction_warnings = list(prediction_runtime.load_warnings)
    for warning in prediction_info.prediction_warnings:
        _append_prediction_warning(prediction_warnings, warning)
    lawson_result = calculate_lawson_status(
        density_m3=density_m3,
        temperature=temperature,
        confinement_time_s=confinement_time_s,
        temp_unit=temp_unit,
    )

    return {
        "predicted_neutron_yield": float(predicted_yield[0]),
        "triple_product": lawson_result.triple_product,
        "lawson_ratio": lawson_result.lawson_ratio,
        "status": lawson_result.status,
        "model_name": prediction_runtime.metadata.best_model_name,
        "clipped_negative_prediction": prediction_info.clipped_negative_prediction,
        "prediction_warnings": prediction_warnings,
    }


def predict_batch(
    input_data: str | Path | pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    assume_temperature_unit: str | None = None,
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    training_run_id: str | None = None,
    runtime: PredictionRuntime | None = None,
    default_artifact_selection: str = DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
    return_predictions: bool = True,
) -> BatchPredictionResult:
    prediction_runtime = _resolve_prediction_runtime(
        runtime,
        model_path=model_path,
        metadata_path=metadata_path,
        training_run_id=training_run_id,
        default_artifact_selection=default_artifact_selection,
    )

    resolved_output_path: Path | None = None
    if output_path is not None:
        resolved_output_path = Path(output_path).expanduser().resolve()
    if not return_predictions and resolved_output_path is None:
        raise ValueError("predict_batch(return_predictions=False) requires an output_path.")

    prediction_frame: pd.DataFrame | None = None
    column_mapping: dict[str, str] | None = None
    clipped_negative_prediction_count = 0
    chunk_prediction_warnings: list[str] = []
    row_count = 0
    if isinstance(input_data, pd.DataFrame):
        prepared_frame, column_mapping = _prepare_batch_inference_frame(
            input_data.copy(),
            runtime=prediction_runtime,
            assume_temperature_unit=assume_temperature_unit,
        )
        materialized_prediction_frame, clipped_negative_prediction_count, chunk_prediction_warnings = _predict_prepared_batch_frame(
            prepared_frame,
            prediction_runtime=prediction_runtime,
        )
        row_count = int(len(materialized_prediction_frame))
        if resolved_output_path is not None:
            with atomic_output_path(resolved_output_path) as temp_output_path:
                materialized_prediction_frame.to_csv(temp_output_path, index=False)
        if return_predictions:
            prediction_frame = materialized_prediction_frame
    else:
        input_path = Path(input_data).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Batch prediction input not found: {input_path}")
        if _can_stream_batch_prediction_csv(input_path):
            column_mapping = None
            row_offset = 0

            def process_streamed_chunks(*, sink_path: Path) -> int:
                nonlocal column_mapping, row_offset, clipped_negative_prediction_count
                write_header = True
                streamed_row_count = 0
                for raw_chunk in pd.read_csv(input_path, chunksize=BATCH_PREDICTION_CSV_CHUNK_ROWS):
                    prepared_frame, chunk_column_mapping = _prepare_batch_inference_frame(
                        raw_chunk,
                        runtime=prediction_runtime,
                        assume_temperature_unit=assume_temperature_unit,
                        start_index=row_offset,
                    )
                    row_offset += len(raw_chunk)
                    if column_mapping is None:
                        column_mapping = chunk_column_mapping
                    elif chunk_column_mapping != column_mapping:
                        raise ValueError(
                            "Batch prediction input changed its column mapping across streamed chunks."
                        )
                    chunk_prediction_frame, chunk_clipped_count, chunk_warnings = _predict_prepared_batch_frame(
                        prepared_frame,
                        prediction_runtime=prediction_runtime,
                    )
                    clipped_negative_prediction_count += chunk_clipped_count
                    for warning in chunk_warnings:
                        _append_prediction_warning(chunk_prediction_warnings, warning)
                    streamed_row_count += int(len(chunk_prediction_frame))
                    chunk_prediction_frame.to_csv(
                        sink_path,
                        mode="w" if write_header else "a",
                        header=write_header,
                        index=False,
                    )
                    write_header = False
                return streamed_row_count

            if resolved_output_path is not None:
                with atomic_output_path(resolved_output_path) as temp_output_path:
                    row_count = process_streamed_chunks(sink_path=temp_output_path)
                    if column_mapping is None:
                        raise ValueError("Batch prediction input must contain at least one row.")
                    if return_predictions:
                        prediction_frame = _read_batch_prediction_frame(temp_output_path)
            else:
                with tempfile.TemporaryDirectory(prefix="fusionflux_batch_predictions_") as temp_dir:
                    temp_prediction_path = Path(temp_dir) / "batch_predictions.csv"
                    row_count = process_streamed_chunks(sink_path=temp_prediction_path)
                    if column_mapping is None:
                        raise ValueError("Batch prediction input must contain at least one row.")
                    if return_predictions:
                        prediction_frame = _read_batch_prediction_frame(temp_prediction_path)
            assert column_mapping is not None
        else:
            raw_frame = pd.read_csv(input_path)
            prepared_frame, column_mapping = _prepare_batch_inference_frame(
                raw_frame,
                runtime=prediction_runtime,
                assume_temperature_unit=assume_temperature_unit,
            )
            materialized_prediction_frame, clipped_negative_prediction_count, chunk_prediction_warnings = _predict_prepared_batch_frame(
                prepared_frame,
                prediction_runtime=prediction_runtime,
            )
            row_count = int(len(materialized_prediction_frame))
            if resolved_output_path is not None:
                with atomic_output_path(resolved_output_path) as temp_output_path:
                    materialized_prediction_frame.to_csv(temp_output_path, index=False)
            if return_predictions:
                prediction_frame = materialized_prediction_frame

    prediction_warnings = list(prediction_runtime.load_warnings)
    for warning in chunk_prediction_warnings:
        _append_prediction_warning(prediction_warnings, warning)

    return BatchPredictionResult(
        predictions=prediction_frame,
        output_path=resolved_output_path,
        row_count=row_count,
        model_name=prediction_runtime.metadata.best_model_name,
        training_run_id=prediction_runtime.metadata.training_run_id,
        schema_version=prediction_runtime.metadata.schema_version,
        model_path=prediction_runtime.model_path,
        metadata_path=prediction_runtime.metadata_path,
        clipped_negative_prediction_count=clipped_negative_prediction_count,
        prediction_warnings=prediction_warnings,
        column_mapping=column_mapping,
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "BATCH_PREDICTION_CSV_CHUNK_ROWS",
    "DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY",
    "DEFAULT_ARTIFACT_SELECTION_NEWEST_COMPATIBLE",
    "LATEST_TRAINING_RUN_FILENAME",
    "TRAINING_METADATA_FILENAME",
    "TRAINING_MODEL_FILENAME",
    "BatchPredictionResult",
    "LoadedPredictionArtifact",
    "PredictionArtifactCandidate",
    "PredictionArtifactManifest",
    "PredictionRuntime",
    "ResolvedPredictionArtifactSelection",
    "SUPPORTED_DEFAULT_ARTIFACT_SELECTION_MODES",
    "TrainingArtifactMetadata",
    "list_prediction_artifacts",
    "load_prediction_runtime",
    "predict_batch",
    "predict_single_case",
    "resolve_prediction_artifact_paths",
]
