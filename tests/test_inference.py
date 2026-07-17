from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from helpers import (
    FeatureEchoModel,
    _build_artifact_metadata,
    _build_feature_echo_artifact,
    _build_grouped_time_series_frame,
    _build_negative_artifact,
    _build_temperature_echo_artifact,
    _bump_version_component,
    _load_training_metadata_record,
    _write_dataset,
    _write_prediction_artifact_run,
)

import config
import features
import inference
import storage
import train_model
from artifact_model import FusionFluxModelArtifact
from config import ORIGINAL_ROW_INDEX_COLUMN, RAW_CSV_ROW_NUMBER_COLUMN


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"confinement_time_s": -1.0}, "confinement_time_s"),
        ({"pressure_pa": -1.0}, "pressure_Pa"),
        ({"ne_20": 12.0}, "ne_20 must match fuel_density_m3 / 1e20"),
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
    assert result.predictions is not None
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
    assert result.predictions is not None
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

    assert result.predictions is not None
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

    assert result.predictions is not None
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
        "scikit_learn": sklearn.__version__,
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
        ("contract_version", 999),
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


def test_committed_artifact_manifest_supports_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    manifest_path = processed_dir / train_model.LATEST_TRAINING_RUN_FILENAME

    # data/processed/ is gitignored, so a fresh clone (e.g. CI before any training
    # run) has no committed artifact to relocate. Skip rather than fail there; the
    # test still validates relocation whenever a local artifact is present.
    if not manifest_path.exists():
        pytest.skip("No local training artifact present; run training first to exercise relocation.")

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


def test_predict_cli_dispatch_scores_single_case(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = features.create_synthetic_dataset(tmp_path / "synthetic_train.csv", n_rows=60, random_state=7)
    train_model.train_models(dataset_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "predict",
            "--density-m3",
            "1e20",
            "--temperature",
            "12",
            "--temp-unit",
            "keV",
            "--confinement-time-s",
            "1",
        ],
    )

    train_model.main()

    payload = json.loads(capsys.readouterr().out)
    assert np.isfinite(payload["predicted_neutron_yield"])
    assert payload["predicted_neutron_yield"] >= 0.0


def test_predict_batch_cli_dispatch_writes_scored_csv(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = features.create_synthetic_dataset(tmp_path / "synthetic_train.csv", n_rows=60, random_state=9)
    train_model.train_models(dataset_path)

    input_frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.4e20, 1.8e20],
            "temperature_keV": [10.0, 14.0, 18.0],
            "confinement_time_s": [0.9, 1.3, 1.7],
        }
    )
    input_csv = tmp_path / "batch_input.csv"
    input_frame.to_csv(input_csv, index=False)
    output_csv = tmp_path / "batch_output.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "predict-batch",
            "--input-csv",
            str(input_csv),
            "--output-path",
            str(output_csv),
        ],
    )

    train_model.main()

    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["output_path"]) == output_csv.resolve()
    assert payload["row_count"] == 3
    scored = pd.read_csv(output_csv)
    assert len(scored) == 3
    assert "predicted_neutron_yield" in scored.columns
    assert (scored["predicted_neutron_yield"] >= 0.0).all()
