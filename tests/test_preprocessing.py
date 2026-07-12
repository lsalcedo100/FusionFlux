from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from helpers import (
    _build_grouped_time_series_frame,
    _write_dataset,
)

import features
import storage
import validation


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
            # Second row is a gross unit mistake (~11x the expected ne_20),
            # which the loosened consistency check must still reject.
            "ne_20": [1.0, 12.0],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "inconsistent_ne20.csv")

    with pytest.raises(ValueError, match="ne_20.*fuel_density_m3 / 1e20"):
        features.prepare_dataset(dataset_path)


def test_prepare_dataset_accepts_physically_divergent_ne_20(
    isolated_project_dirs: dict[str, Path],
    tmp_path: Path,
) -> None:
    # Electron density can differ from fuel-ion density by an order-unity factor
    # (impurities / Z_eff / isotope mix); such rows must be accepted rather than
    # rejected as "inconsistent".
    frame = pd.DataFrame(
        {
            "fuel_density_m3": [1.0e20, 1.1e20],
            "temperature_keV": [12.0, 13.0],
            "confinement_time_s": [1.0, 1.1],
            "neutron_yield": [100.0, 110.0],
            "ne_20": [1.3, 1.4],
        }
    )
    dataset_path = _write_dataset(tmp_path, frame, "divergent_ne20.csv")

    prepared = features.prepare_dataset(dataset_path)

    assert len(prepared.dataframe) == 2


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


def test_build_preprocessing_contract_is_structural_and_source_independent() -> None:
    # The contract no longer inspects function source or bytecode: it is a purely
    # structural, versioned description, so it never touches the filesystem or the
    # inspect module and never carries source/logic fingerprint fields.
    contract = features.build_preprocessing_contract()

    assert contract["sha256"]
    assert contract["contract_version"] == features.PREPROCESSING_CONTRACT_VERSION
    assert "source_sha256" not in contract
    assert "logic_fingerprints" not in contract
    assert "source_fingerprint_method" not in contract
