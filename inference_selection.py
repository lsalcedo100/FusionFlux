"""Prediction artifact discovery, default selection, and loading.

Turns the on-disk collection of training runs into a single loadable
``PredictionRuntime``: it enumerates candidate artifacts, ranks them by runtime
compatibility and recency under the configured selection mode, and deserializes
the first candidate that passes every compatibility check. Depends on
``inference_artifacts`` for the schema and parsers.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import joblib

import config
from artifact_model import FusionFluxModelArtifact
from features import (
    describe_preprocessing_contract_differences,
    ensure_project_directories,
    preprocessing_contract_matches,
)
from inference_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
    DEFAULT_ARTIFACT_SELECTION_NEWEST_COMPATIBLE,
    LATEST_TRAINING_RUN_FILENAME,
    RUNTIME_COMPATIBILITY_EXACT,
    TRAINING_METADATA_FILENAME,
    TRAINING_MODEL_FILENAME,
    LoadedPredictionArtifact,
    PredictionArtifactCandidate,
    PredictionRuntime,
    ResolvedPredictionArtifactSelection,
    TrainingArtifactMetadata,
    _artifact_candidate_recency_key,
    _describe_runtime_compatibility_rank,
    _parse_artifact_created_at,
    _parse_prediction_manifest,
    _parse_training_artifact_metadata,
    _read_json_object,
    _resolve_stored_path,
    _resolve_training_metadata_paths,
    _validate_default_artifact_selection_mode,
    _validate_runtime_versions_for_loading,
    _validate_saved_model_metadata_paths,
)


def _discover_prediction_artifact_candidate_paths() -> tuple[list[tuple[Path, Path]], list[str]]:
    manifest_path = config.get_data_processed_dir() / LATEST_TRAINING_RUN_FILENAME
    candidate_paths: list[tuple[Path, Path]] = []
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
            candidate_paths.append(manifest_candidate)
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
            candidate_paths.append(candidate)
            seen_candidates.add(candidate)

    return candidate_paths, discovery_failures


def _inspect_available_prediction_artifact_candidates() -> tuple[list[PredictionArtifactCandidate], list[str]]:
    candidate_paths, discovery_failures = _discover_prediction_artifact_candidate_paths()
    attempted_failures = list(discovery_failures)
    inspected_candidates: list[PredictionArtifactCandidate] = []
    for resolved_model_path, resolved_metadata_path in candidate_paths:
        try:
            inspected_candidates.append(
                _inspect_prediction_artifact_candidate(
                    model_path=resolved_model_path,
                    metadata_path=resolved_metadata_path,
                )
            )
        except Exception as exc:
            attempted_failures.append(
                f"Skipped default artifact candidate {resolved_metadata_path.parent}: {exc}"
            )
    # Callers impose their own ordering (list_prediction_artifacts sorts by
    # recency, _select_loadable_default_artifact sorts via
    # _sorted_default_artifact_candidates, and the training_run_id lookup is
    # order-independent), so sorting here would just be redundant work.
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

    if training_run_id is not None:
        inspected_candidates, attempted_failures = _inspect_available_prediction_artifact_candidates()
        if not inspected_candidates:
            raise _no_usable_default_artifact_error(attempted_failures)
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

    # Default selection must agree with the loader: return the highest-priority
    # candidate that actually deserializes and passes compatibility checks, not
    # merely the one whose metadata ranks best. Otherwise resolve_prediction_
    # artifact_paths() could hand back an artifact that load_prediction_runtime()
    # would then skip.
    loaded_artifact = _select_loadable_default_artifact(selection_mode=selection_mode)
    return ResolvedPredictionArtifactSelection(
        model_path=loaded_artifact.model_path,
        metadata_path=loaded_artifact.metadata_path,
        training_run_id=loaded_artifact.metadata.training_run_id,
        resolution_warnings=loaded_artifact.load_warnings,
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
    )


def _no_usable_default_artifact_error(attempted_failures: list[str]) -> Exception:
    if attempted_failures:
        failure_excerpt = " ".join(attempted_failures[:3])
        return ValueError(
            "No usable training artifacts were found for default prediction. "
            f"{failure_excerpt}"
        )
    return FileNotFoundError(
        "No training artifacts were found. Train a model first or pass both --model-path and --metadata-path."
    )


def _sorted_default_artifact_candidates(
    inspected_candidates: list[PredictionArtifactCandidate],
    *,
    selection_mode: str,
) -> list[PredictionArtifactCandidate]:
    if selection_mode == DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY:
        return sorted(
            inspected_candidates,
            key=lambda candidate: (
                candidate.runtime_compatibility_rank,
                -_artifact_candidate_recency_key(candidate),
            ),
        )
    return sorted(
        inspected_candidates,
        key=lambda candidate: (
            -_artifact_candidate_recency_key(candidate),
            candidate.runtime_compatibility_rank,
        ),
    )


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


def _select_loadable_default_artifact(*, selection_mode: str) -> LoadedPredictionArtifact:
    """Pick the highest-priority default artifact that actually loads.

    Candidates are ranked by ``selection_mode`` and attempted in order; any that
    fail to deserialize or fail a compatibility check are skipped. This is the
    single source of truth for default selection so ``resolve_prediction_artifact_paths``
    and ``load_prediction_runtime`` can never disagree on which artifact is default.
    """
    inspected_candidates, attempted_failures = _inspect_available_prediction_artifact_candidates()
    if not inspected_candidates:
        raise _no_usable_default_artifact_error(attempted_failures)

    for candidate in _sorted_default_artifact_candidates(inspected_candidates, selection_mode=selection_mode):
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
                load_warnings=(*selection_warnings, *loaded_artifact.load_warnings),
            )
        return loaded_artifact

    raise _no_usable_default_artifact_error(attempted_failures)


def _load_prediction_artifact(
    model_path: str | Path | None,
    metadata_path: str | Path | None,
    *,
    training_run_id: str | None = None,
    default_artifact_selection: str = DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
) -> LoadedPredictionArtifact:
    selection_mode = _validate_default_artifact_selection_mode(default_artifact_selection)
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

    return _select_loadable_default_artifact(selection_mode=selection_mode)


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
