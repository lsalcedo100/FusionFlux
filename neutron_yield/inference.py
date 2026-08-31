"""Single-case and batch prediction entry points for FusionFlux.

Artifact schema/parsing lives in ``inference_artifacts`` and artifact
discovery/loading in ``inference_selection``; this module drives the actual
prediction flow and re-exports the public inference API (plus the artifact
writers ``training`` depends on) so ``import inference`` stays the single stable
entry point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import GROUP_COLUMN, ORIGINAL_ROW_INDEX_COLUMN, RAW_CSV_ROW_NUMBER_COLUMN, TARGET_LOG_COLUMN
from lawson import calculate_lawson_status, to_kev
from storage import atomic_output_path
from validation import validate_physics_inputs

from .features import (
    OPTIONAL_PHYSICS_COLUMNS,
    add_source_identity_columns,
    align_to_feature_schema,
    prepare_model_frame,
    read_dataset_csv,
    resolve_column_mapping,
)
from .inference_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
    DEFAULT_ARTIFACT_SELECTION_NEWEST_COMPATIBLE,
    LATEST_TRAINING_RUN_FILENAME,
    SUPPORTED_DEFAULT_ARTIFACT_SELECTION_MODES,
    TRAINING_METADATA_FILENAME,
    TRAINING_MODEL_FILENAME,
    BatchPredictionResult,
    LoadedPredictionArtifact,
    PredictionArtifactCandidate,
    PredictionArtifactManifest,
    PredictionRuntime,
    ResolvedPredictionArtifactSelection,
    TrainingArtifactMetadata,
)
from .inference_artifacts import _current_runtime_versions as _current_runtime_versions
from .inference_artifacts import _parse_training_artifact_metadata as _parse_training_artifact_metadata
from .inference_artifacts import _resolve_training_metadata_paths as _resolve_training_metadata_paths
from .inference_artifacts import _serialize_training_metadata_paths as _serialize_training_metadata_paths
from .inference_artifacts import _write_latest_training_run_manifest as _write_latest_training_run_manifest
from .inference_selection import _ensure_artifact_compatibility as _ensure_artifact_compatibility
from .inference_selection import _load_prediction_artifact as _load_prediction_artifact
from .inference_selection import _resolve_prediction_artifact_paths as _resolve_prediction_artifact_paths
from .inference_selection import (
    _resolve_prediction_runtime,
    list_prediction_artifacts,
    load_prediction_runtime,
    resolve_prediction_artifact_paths,
)

BATCH_PREDICTION_CSV_CHUNK_ROWS = 20000


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
    try:
        header_frame = pd.read_csv(input_path, nrows=0)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Batch prediction input is empty or has no header row: {input_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Batch prediction input could not be parsed as CSV: {input_path} ({exc}).") from exc
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
    # The runtime already validated the preprocessing contract when the artifact
    # was loaded, so skip the identical per-chunk revalidation on every batch.
    predictions, prediction_info = prediction_runtime.model.predict_with_info(
        align_to_feature_schema(prepared_frame, prediction_runtime.metadata.feature_columns),
        revalidate_runtime=False,
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
    # Contract already validated at load time; avoid redundant revalidation here.
    predicted_yield, prediction_info = prediction_runtime.model.predict_with_info(
        align_to_feature_schema(inference_df, feature_columns),
        revalidate_runtime=False,
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
            row_offset = 0
            # Only retained when the caller wants the predictions returned in
            # memory; when return_predictions is False this stays empty so the
            # streaming path keeps its constant-memory behavior.
            collected_prediction_frames: list[pd.DataFrame] = []

            def process_streamed_chunks(*, sink_path: Path | None) -> int:
                nonlocal column_mapping, row_offset, clipped_negative_prediction_count
                write_header = True
                streamed_row_count = 0
                # Use the reader as a context manager so the underlying file
                # handle is closed even if a chunk raises mid-stream (e.g. the
                # column-mapping-change guard below).
                with pd.read_csv(input_path, chunksize=BATCH_PREDICTION_CSV_CHUNK_ROWS) as chunk_reader:
                    for raw_chunk in chunk_reader:
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
                        if return_predictions:
                            collected_prediction_frames.append(chunk_prediction_frame)
                        # When there is no real output target (caller only wants the
                        # predictions returned in memory) skip the CSV write entirely
                        # instead of streaming every chunk to a throwaway temp file.
                        if sink_path is not None:
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
            else:
                row_count = process_streamed_chunks(sink_path=None)
                if column_mapping is None:
                    raise ValueError("Batch prediction input must contain at least one row.")
            if return_predictions:
                # Reuse the in-memory per-chunk frames instead of round-tripping
                # the streamed CSV back through pd.read_csv, which would re-infer
                # dtypes and diverge from the non-streaming return frame.
                prediction_frame = pd.concat(collected_prediction_frames, ignore_index=True)
            assert column_mapping is not None
        else:
            raw_frame = read_dataset_csv(input_path, context="Batch prediction input")
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

    assert column_mapping is not None
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
