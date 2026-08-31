"""Per-run artifact path layout and atomic publish/cleanup for training runs.

A training run is written under a hidden ``.staging`` directory and only renamed
into place once every artifact is on disk, so a crash mid-run never leaves a
partially-written run for the inference loader to discover.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from uuid import uuid4

import config

from .inference import TRAINING_METADATA_FILENAME, TRAINING_MODEL_FILENAME

TRAINING_RUNS_DIRNAME = "runs"


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
