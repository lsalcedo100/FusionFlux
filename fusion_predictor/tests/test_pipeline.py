from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

import features
import train_model


@pytest.fixture
def isolated_project_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    plots_dir = processed_dir / "plots"
    models_dir = processed_dir / "models"

    for module in (features, train_model):
        monkeypatch.setattr(module, "DATA_PROCESSED_DIR", processed_dir)
        monkeypatch.setattr(module, "PLOTS_DIR", plots_dir)
        monkeypatch.setattr(module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(features, "DATA_RAW_DIR", raw_dir)

    return {
        "raw": raw_dir,
        "processed": processed_dir,
        "plots": plots_dir,
        "models": models_dir,
    }


def _write_dataset(tmp_path: Path, frame: pd.DataFrame, name: str = "dataset.csv") -> Path:
    dataset_path = tmp_path / name
    frame.to_csv(dataset_path, index=False)
    return dataset_path


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
    assert aggregated.loc[0, "temperature_keV"] == pytest.approx(12.0)
    assert aggregated.loc[1, "temperature_keV"] == pytest.approx(17.0)
    assert aggregated.loc[0, "neutron_yield"] == pytest.approx(200.0)
    assert aggregated.loc[1, "neutron_yield"] == pytest.approx(70.0)
    assert "tau_E_ipb98_s" in aggregated.columns
    assert aggregated["tau_E_ipb98_s"].notna().all()
    assert prepared.processed_path.exists()


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


def test_train_models_requires_explicit_dataset_source(
    isolated_project_dirs: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="--dataset-path.*--allow-synthetic"):
        train_model.train_models()


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
    assert metadata["saved_model"]["fit_scope"] == "full_prepared_dataset"


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

    monkeypatch.setattr(train_model, "build_model_registry", build_dummy_registry)
    monkeypatch.setattr(
        train_model,
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


def test_train_parser_help_mentions_explicit_synthetic_flag() -> None:
    parser = train_model.build_parser()
    train_subparser = next(
        action.choices["train"]
        for action in parser._actions
        if hasattr(action, "choices") and action.choices is not None and "train" in action.choices
    )
    help_text = train_subparser.format_help()

    assert "--allow-synthetic" in help_text
    assert "Generate and train on synthetic demo data" in help_text
    assert "--dataset-path" in help_text


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
