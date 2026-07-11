from __future__ import annotations

import ast
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import CodeType
from typing import Callable, Mapping, Optional, Union, cast
from uuid import uuid4
import tokenize

import numpy as np
import pandas as pd

import config
from config import (
    BASE_FEATURE_COLUMNS,
    COLUMN_ALIASES,
    ENGINEERED_FEATURE_COLUMNS,
    GROUP_COLUMN,
    LAWSON_DT_IGNITION,
    LEAKAGE_COLUMNS,
    NE_20_ABSOLUTE_TOLERANCE,
    NE_20_REFERENCE_DENSITY_M3,
    NE_20_RELATIVE_TOLERANCE,
    ORIGINAL_ROW_INDEX_COLUMN,
    RANDOM_STATE,
    RAW_CSV_ROW_NUMBER_COLUMN,
    SYNTHETIC_DATASET_ROWS,
    TARGET_COLUMN,
    TARGET_LOG_COLUMN,
)
from lawson import to_kev
from storage import write_dataframe_csv_atomic
from validation import is_boolean_like, validate_physics_dataframe

REQUIRED_PHYSICS_COLUMNS = ("fuel_density_m3", "temperature_keV", "confinement_time_s", TARGET_COLUMN)
OPTIONAL_PHYSICS_COLUMNS = (
    "fuel_purity",
    "energy_input_MJ",
    "pressure_Pa",
    "Ip_MA",
    "Bt_T",
    "R_m",
    "a_m",
    "kappa",
    "ne_20",
    "M_amu",
    "Pin_MW",
)
ALIAS_RELATIVE_TOLERANCE = 1e-6
ALIAS_ABSOLUTE_TOLERANCE = 1e-12
SUPPORTED_TEMPERATURE_UNITS = ("keV", "eV", "K")
PREPROCESSING_CONTRACT_VERSION = 2
ROW_IDENTITY_COLUMNS = (ORIGINAL_ROW_INDEX_COLUMN, RAW_CSV_ROW_NUMBER_COLUMN)
DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS = 2
PREPROCESSING_LOGIC_FINGERPRINT_METHOD = "python_source_tokens_v1"
LEGACY_PREPROCESSING_LOGIC_FINGERPRINT_METHOD = "python_code_object_v1"


@dataclass
class PreparedDataset:
    """Prepared training data plus candidate features before train-split schema selection."""

    raw_path: Path
    processed_path: Path
    dataframe: pd.DataFrame
    audit_summary: dict[str, object]
    column_mapping: dict[str, str]
    candidate_feature_columns: list[str]
    dataset_source_kind: str
    synthetic_data_used: bool
    requested_dataset_path: Path | None
    synthetic_random_state: int | None
    synthetic_row_count: int | None

    @property
    def feature_columns(self) -> list[str]:
        return self.candidate_feature_columns


@dataclass(frozen=True)
class PreparedModelFrame:
    dataframe: pd.DataFrame
    column_mapping: dict[str, str]


@dataclass(frozen=True)
class PreprocessingContractCompatibilityReport:
    compatible: bool
    differing_fields: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def ensure_project_directories() -> None:
    config.get_data_raw_dir().mkdir(parents=True, exist_ok=True)
    config.get_data_processed_dir().mkdir(parents=True, exist_ok=True)


def create_synthetic_dataset(
    output_path: Optional[Path] = None,
    n_rows: int = SYNTHETIC_DATASET_ROWS,
    random_state: int = RANDOM_STATE,
) -> Path:
    ensure_project_directories()
    output_path = output_path or (config.get_data_raw_dir() / "synthetic_nuclear_fusion_experiment.csv")
    output_path = output_path.expanduser().resolve()
    if n_rows <= 0:
        raise ValueError("n_rows must be a positive integer.")

    rng = np.random.default_rng(random_state)
    shot_count = int(np.ceil(n_rows / 6))
    shot_ids = np.repeat(np.arange(shot_count), 6)[:n_rows]
    fuel_density_m3 = 10 ** rng.uniform(19.3, 20.5, n_rows)
    temperature_keV = rng.uniform(4.0, 28.0, n_rows)
    confinement_time_s = rng.uniform(0.15, 5.5, n_rows)
    fuel_purity = rng.uniform(0.72, 0.995, n_rows)
    energy_input_MJ = rng.uniform(12.0, 220.0, n_rows)
    pressure_Pa = 10 ** rng.uniform(4.8, 6.6, n_rows)

    ip_ma = rng.uniform(4.5, 18.0, n_rows)
    bt_t = rng.uniform(2.2, 7.2, n_rows)
    r_m = rng.uniform(1.8, 6.5, n_rows)
    a_m = r_m * rng.uniform(0.24, 0.38, n_rows)
    kappa = rng.uniform(1.45, 2.2, n_rows)
    pin_mw = rng.uniform(6.0, 130.0, n_rows)
    m_amu = rng.uniform(2.35, 2.65, n_rows)

    ne_20 = fuel_density_m3 / NE_20_REFERENCE_DENSITY_M3
    epsilon = a_m / r_m
    tau_e_ipb98_s = (
        0.0562
        * np.power(ip_ma, 0.93)
        * np.power(bt_t, 0.15)
        * np.power(r_m, 1.97)
        * np.power(epsilon, 0.58)
        * np.power(kappa, 0.78)
        * np.power(ne_20, 0.41)
        * np.power(m_amu, 0.19)
        * np.power(pin_mw, -0.69)
    )

    triple_product = fuel_density_m3 * temperature_keV * confinement_time_s
    lawson_ratio = triple_product / LAWSON_DT_IGNITION

    log_yield_signal = (
        12.5
        + 1.6 * np.log1p(lawson_ratio * 10.0)
        + 0.35 * np.log1p(energy_input_MJ)
        + 0.25 * np.log1p(pressure_Pa / 1e5)
        + 0.40 * np.log1p(tau_e_ipb98_s * 10.0)
        + 2.5 * (fuel_purity - 0.75)
        + rng.normal(0.0, 0.55, n_rows)
    )
    neutron_yield = np.expm1(np.clip(log_yield_signal, a_min=4.0, a_max=None))
    power_output_mw = np.maximum(0.05, neutron_yield * 2.5e-7 + rng.normal(0.0, 0.2, n_rows))

    df = pd.DataFrame(
        {
            "shot_id": shot_ids.astype(int),
            "fuel_density_m3": fuel_density_m3,
            "temperature_keV": temperature_keV,
            "confinement_time_s": confinement_time_s,
            "fuel_purity": fuel_purity,
            "energy_input_MJ": energy_input_MJ,
            "pressure_Pa": pressure_Pa,
            "Ip_MA": ip_ma,
            "Bt_T": bt_t,
            "R_m": r_m,
            "a_m": a_m,
            "kappa": kappa,
            "ne_20": ne_20,
            "M_amu": m_amu,
            "Pin_MW": pin_mw,
            "tau_E_ipb98_s": tau_e_ipb98_s,
            "power_output_MW": power_output_mw,
            "neutron_yield": neutron_yield,
        }
    )
    write_dataframe_csv_atomic(output_path, df, index=False)
    return output_path


def resolve_training_dataset_path(
    dataset_path: Optional[Union[str, Path]] = None,
    *,
    allow_synthetic: bool = False,
    synthetic_output_path: Path | None = None,
    synthetic_n_rows: int = SYNTHETIC_DATASET_ROWS,
    synthetic_random_state: int = RANDOM_STATE,
) -> tuple[Path, str, Path | None]:
    ensure_project_directories()
    if dataset_path is not None:
        resolved = Path(dataset_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset not found: {resolved}")
        return resolved, "provided_path", resolved

    if allow_synthetic:
        synthetic_path = create_synthetic_dataset(
            output_path=synthetic_output_path,
            n_rows=synthetic_n_rows,
            random_state=synthetic_random_state,
        )
        return synthetic_path, "synthetic_generated", None

    raise ValueError(
        "Training dataset is required. Pass --dataset-path /path/to/dataset.csv to train on a real dataset, "
        "or rerun with --allow-synthetic to generate synthetic demo data explicitly."
    )


def audit_dataframe(df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "duplicates": int(df.duplicated().sum()),
        "null_counts": {column: int(value) for column, value in df.isna().sum().items()},
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }
    if TARGET_COLUMN in df.columns:
        target = _numeric_series(df, TARGET_COLUMN)
        summary["target_distribution"] = {
            "min": float(target.min(skipna=True)),
            "median": float(target.median(skipna=True)),
            "mean": float(target.mean(skipna=True)),
            "max": float(target.max(skipna=True)),
        }
    return summary


def resolve_column_mapping(df: pd.DataFrame) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical_name
    return rename_map


def _is_missing_tabular_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip() == ""
    return bool(pd.isna(value))


def _format_rows(rows: list[object]) -> str:
    displayed_rows = ", ".join(str(row) for row in rows[:5])
    if len(rows) > 5:
        displayed_rows = f"{displayed_rows}, ..."
    return displayed_rows


def _find_conflicting_rows(left: pd.Series, right: pd.Series) -> list[object]:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    conflicting_rows: list[object] = []
    for index, left_value, right_value, left_number, right_number in zip(
        left.index,
        left,
        right,
        left_numeric,
        right_numeric,
    ):
        if _is_missing_tabular_value(left_value) or _is_missing_tabular_value(right_value):
            continue
        if pd.notna(left_number) and pd.notna(right_number):
            if not np.isclose(
                float(left_number),
                float(right_number),
                rtol=ALIAS_RELATIVE_TOLERANCE,
                atol=ALIAS_ABSOLUTE_TOLERANCE,
            ):
                conflicting_rows.append(index)
            continue
        if str(left_value).strip() != str(right_value).strip():
            conflicting_rows.append(index)
    return conflicting_rows


def _fill_missing_values(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    missing_mask = primary.map(_is_missing_tabular_value)
    if not missing_mask.any():
        return primary
    combined = primary.copy()
    combined.loc[missing_mask] = fallback.loc[missing_mask]
    return combined


def _raise_column_conflict(canonical_name: str, left_column: str, right_column: str, rows: list[object]) -> None:
    raise ValueError(
        f"Conflicting source columns for {canonical_name}: "
        f"{left_column} and {right_column} disagree at rows [{_format_rows(rows)}]."
    )


def canonicalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical_df = df.copy()
    for canonical_name, aliases in COLUMN_ALIASES.items():
        present_aliases = [alias for alias in aliases if alias in canonical_df.columns]
        if not present_aliases:
            continue

        combined = _series(canonical_df, present_aliases[0]).copy()
        for alias in present_aliases[1:]:
            alias_series = _series(canonical_df, alias)
            conflicting_rows = _find_conflicting_rows(combined, alias_series)
            if conflicting_rows:
                _raise_column_conflict(canonical_name, present_aliases[0], alias, conflicting_rows)
            combined = _fill_missing_values(combined, alias_series)

        canonical_df[canonical_name] = combined
        redundant_columns = [alias for alias in present_aliases if alias != canonical_name]
        if redundant_columns:
            canonical_df = canonical_df.drop(columns=redundant_columns)
    return canonical_df


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, df[column])


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(_series(df, column), errors="coerce"))


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_candidates = set(BASE_FEATURE_COLUMNS) | set(ENGINEERED_FEATURE_COLUMNS) | {
        TARGET_COLUMN,
        "time_s",
        "time_ms",
        "power_output_MW",
        "tau_E_ipb98_s",
    }
    for column in numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _coerce_numeric_source_series(series: pd.Series) -> tuple[pd.Series, list[object]]:
    numeric_series = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    missing_mask = series.map(_is_missing_tabular_value)
    boolean_mask = series.map(is_boolean_like) & ~missing_mask
    invalid_rows = series.index[((numeric_series.isna() | boolean_mask) & ~missing_mask).to_numpy(dtype=bool)].tolist()
    return numeric_series, invalid_rows


def _validated_time_series_for_aggregation(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    raw_series = _series(df, column)
    numeric_series, invalid_rows = _coerce_numeric_source_series(raw_series)
    missing_mask = raw_series.map(_is_missing_tabular_value)
    missing_rows = raw_series.index[missing_mask.to_numpy(dtype=bool)].tolist()
    if missing_rows:
        raise ValueError(
            f"{column} rows [{_format_rows(missing_rows)}] must be present for time-resolved shot aggregation."
        )
    if invalid_rows:
        raise ValueError(
            f"{column} rows [{_format_rows(invalid_rows)}] must be numeric for time-resolved shot aggregation."
        )
    return numeric_series


def _validate_aggregation_timestamps(df: pd.DataFrame) -> None:
    if GROUP_COLUMN not in df.columns:
        return
    if df[GROUP_COLUMN].nunique() == len(df):
        return
    if "time_s" in df.columns:
        _validated_time_series_for_aggregation(df, "time_s")
    elif "time_ms" in df.columns:
        _validated_time_series_for_aggregation(df, "time_ms")


def _is_hashable_tabular_value(value: object) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def validate_group_identifier_column(df: pd.DataFrame) -> None:
    if GROUP_COLUMN not in df.columns:
        return

    group_series = _series(df, GROUP_COLUMN)
    missing_mask = group_series.map(_is_missing_tabular_value)
    invalid_mask = group_series.map(
        lambda value: (
            not _is_missing_tabular_value(value)
            and (is_boolean_like(value) or not _is_hashable_tabular_value(value))
        )
    )

    missing_rows = group_series.index[missing_mask.to_numpy(dtype=bool)].tolist()
    invalid_rows = group_series.index[invalid_mask.to_numpy(dtype=bool)].tolist()
    if missing_rows:
        raise ValueError(
            f"{GROUP_COLUMN} rows [{_format_rows(missing_rows)}] must be present and non-empty "
            "before grouped shot aggregation or cross-validation."
        )
    if invalid_rows:
        raise ValueError(
            f"{GROUP_COLUMN} rows [{_format_rows(invalid_rows)}] must be hashable, non-boolean identifiers "
            "before grouped shot aggregation or cross-validation."
        )


def _temperature_series_with_units(temperature: pd.Series, units: pd.Series) -> tuple[pd.Series, list[object]]:
    converted_values: list[float] = []
    invalid_rows: list[object] = []
    for index, value, unit in zip(temperature.index, temperature, units):
        value_missing = _is_missing_tabular_value(value)
        unit_missing = _is_missing_tabular_value(unit)
        if value_missing and unit_missing:
            converted_values.append(np.nan)
            continue
        if value_missing or unit_missing:
            invalid_rows.append(index)
            converted_values.append(np.nan)
            continue
        try:
            converted_values.append(float(to_kev(value, str(unit))))
        except ValueError:
            invalid_rows.append(index)
            converted_values.append(np.nan)
    return pd.Series(converted_values, index=temperature.index, dtype=float), invalid_rows


def _convert_temperature_series(
    temperature: pd.Series,
    unit: str,
) -> tuple[pd.Series, list[object]]:
    converted_values: list[float] = []
    invalid_rows: list[object] = []
    for index, value in zip(temperature.index, temperature):
        if _is_missing_tabular_value(value):
            converted_values.append(np.nan)
            continue
        try:
            converted_values.append(float(to_kev(value, unit)))
        except ValueError:
            invalid_rows.append(index)
            converted_values.append(np.nan)
    return pd.Series(converted_values, index=temperature.index, dtype=float), invalid_rows


def standardize_temperature_column(
    df: pd.DataFrame,
    *,
    assume_temperature_unit: str | None = None,
) -> pd.DataFrame:
    if assume_temperature_unit is not None and assume_temperature_unit not in SUPPORTED_TEMPERATURE_UNITS:
        raise ValueError(
            "assume_temperature_unit must be one of "
            f"{', '.join(SUPPORTED_TEMPERATURE_UNITS)} when provided."
        )

    temperature_candidates: dict[str, pd.Series] = {}

    if "temperature_keV" in df.columns:
        temperature_kev, invalid_rows = _coerce_numeric_source_series(_series(df, "temperature_keV"))
        if invalid_rows:
            raise ValueError(
                f"temperature_keV rows [{_format_rows(invalid_rows)}] must be numeric when provided."
            )
        temperature_candidates["temperature_keV"] = temperature_kev

    if "temperature_eV" in df.columns:
        temperature_ev, invalid_rows = _coerce_numeric_source_series(_series(df, "temperature_eV"))
        if invalid_rows:
            raise ValueError(
                f"temperature_eV rows [{_format_rows(invalid_rows)}] must be numeric when provided."
            )
        temperature_candidates["temperature_eV"] = temperature_ev / 1e3

    if "temperature_K" in df.columns:
        temperature_k, invalid_rows = _coerce_numeric_source_series(_series(df, "temperature_K"))
        if invalid_rows:
            raise ValueError(
                f"temperature_K rows [{_format_rows(invalid_rows)}] must be numeric when provided."
            )
        temperature_candidates["temperature_K"] = temperature_k.apply(
            lambda value: to_kev(float(value), "K") if pd.notna(value) else np.nan
        )

    if "temperature" in df.columns and "temperature_unit" in df.columns:
        temperature_with_units, invalid_rows = _temperature_series_with_units(
            _series(df, "temperature"),
            _series(df, "temperature_unit"),
        )
        if invalid_rows:
            raise ValueError(
                "temperature/temperature_unit rows "
                f"[{_format_rows(invalid_rows)}] must provide a numeric temperature with unit keV, eV, or K."
        )
        temperature_candidates["temperature_with_unit"] = temperature_with_units
    elif "temperature" in df.columns:
        if assume_temperature_unit is None:
            raise ValueError(
                "A generic 'temperature' column requires a companion 'temperature_unit' column "
                "or an explicit assume_temperature_unit value."
            )
        generic_temperature, invalid_rows = _convert_temperature_series(
            _series(df, "temperature"),
            assume_temperature_unit,
        )
        if invalid_rows:
            raise ValueError(
                "temperature rows "
                f"[{_format_rows(invalid_rows)}] must be numeric and compatible with "
                f"{assume_temperature_unit} when provided."
            )
        temperature_candidates["temperature"] = generic_temperature

    if not temperature_candidates:
        raise ValueError("No temperature column could be mapped to a canonical temperature_keV field.")

    candidate_items = list(temperature_candidates.items())
    resolved_name, resolved_temperature = candidate_items[0]
    for candidate_name, candidate_series in candidate_items[1:]:
        conflicting_rows = _find_conflicting_rows(resolved_temperature, candidate_series)
        if conflicting_rows:
            _raise_column_conflict("temperature_keV", resolved_name, candidate_name, conflicting_rows)
        resolved_temperature = _fill_missing_values(resolved_temperature, candidate_series)

    df["temperature_keV"] = resolved_temperature
    return df


def aggregate_time_resolved_shots(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_time_resolved_shots_at_cutoff(
        df,
        prediction_cutoff_rows=DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS,
    )


def aggregate_time_resolved_shots_at_cutoff(
    df: pd.DataFrame,
    *,
    prediction_cutoff_rows: int,
) -> pd.DataFrame:
    if prediction_cutoff_rows <= 0:
        raise ValueError("prediction_cutoff_rows must be a positive integer.")

    if GROUP_COLUMN not in df.columns:
        return df

    if df[GROUP_COLUMN].nunique() == len(df):
        return df

    normalized_df = df.copy()
    if "time_s" in normalized_df.columns:
        normalized_df["time_s"] = _validated_time_series_for_aggregation(normalized_df, "time_s")
    elif "time_ms" in normalized_df.columns:
        normalized_df["time_ms"] = _validated_time_series_for_aggregation(normalized_df, "time_ms")
        normalized_df["time_s"] = cast(pd.Series, normalized_df["time_ms"] / 1e3)
    else:
        return df

    aggregated_rows: list[dict[str, object]] = []
    for shot_id, group in normalized_df.sort_values([GROUP_COLUMN, "time_s"]).groupby(GROUP_COLUMN):
        # Use a fixed observation-count cutoff so each shot example only depends on
        # measurements available by that explicit prediction point.
        cutoff_row_count = min(len(group), prediction_cutoff_rows)
        cutoff_window = group.iloc[:cutoff_row_count]
        cutoff_row = cutoff_window.iloc[-1]
        record: dict[str, object] = {GROUP_COLUMN: shot_id}

        for identity_column in ROW_IDENTITY_COLUMNS:
            if identity_column in group.columns:
                identity_value = cutoff_row[identity_column]
                record[identity_column] = (
                    int(identity_value) if pd.notna(identity_value) else np.nan
                )

        for column in group.columns:
            if column == GROUP_COLUMN or column in ROW_IDENTITY_COLUMNS:
                continue
            group_column = _series(group, column)
            cutoff_column = _series(cutoff_window, column)
            if pd.api.types.is_numeric_dtype(group_column):
                if column in {TARGET_COLUMN, "time_s", "time_ms"}:
                    # Predict the yield observed at the explicit cutoff row rather than
                    # a full-shot future summary.
                    cutoff_value = cutoff_row[column]
                    record[column] = float(cutoff_value) if pd.notna(cutoff_value) else np.nan
                else:
                    record[column] = float(cutoff_column.median())
            else:
                mode = cutoff_column.mode(dropna=True)
                record[column] = mode.iloc[0] if not mode.empty else cutoff_column.iloc[0]
        aggregated_rows.append(record)

    return pd.DataFrame(aggregated_rows)


def _deduplicate_dataframe_rows(df: pd.DataFrame) -> pd.DataFrame:
    dedupe_columns = [column for column in df.columns if column not in ROW_IDENTITY_COLUMNS]
    return df.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _deduplicate_dataframe_rows(df)
    validate_physics_dataframe(
        df,
        required_fields=REQUIRED_PHYSICS_COLUMNS,
        optional_fields=OPTIONAL_PHYSICS_COLUMNS,
    )
    return df


def prepare_model_frame(
    frame: pd.DataFrame,
    *,
    assume_temperature_unit: str | None = None,
    shot_prediction_cutoff_rows: int = DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS,
    require_target: bool = False,
    deduplicate_rows: bool = True,
) -> PreparedModelFrame:
    column_mapping = resolve_column_mapping(frame)
    prepared_frame = canonicalize_dataframe_columns(frame)
    prepared_frame = standardize_temperature_column(
        prepared_frame,
        assume_temperature_unit=assume_temperature_unit,
    )
    required_fields = (
        REQUIRED_PHYSICS_COLUMNS
        if require_target
        else ("fuel_density_m3", "temperature_keV", "confinement_time_s")
    )
    validate_physics_dataframe(
        prepared_frame,
        required_fields=required_fields,
        optional_fields=OPTIONAL_PHYSICS_COLUMNS,
    )
    validate_group_identifier_column(prepared_frame)
    _validate_aggregation_timestamps(prepared_frame)
    prepared_frame = _coerce_numeric_columns(prepared_frame)
    prepared_frame = aggregate_time_resolved_shots_at_cutoff(
        prepared_frame,
        prediction_cutoff_rows=shot_prediction_cutoff_rows,
    )
    if deduplicate_rows:
        prepared_frame = _deduplicate_dataframe_rows(prepared_frame)
    prepared_frame = engineer_features(prepared_frame)
    return PreparedModelFrame(
        dataframe=prepared_frame,
        column_mapping=column_mapping,
    )


def add_ipb98_proxy(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Ip_MA", "Bt_T", "R_m", "a_m", "kappa", "Pin_MW"}
    if not required_columns.issubset(df.columns):
        return df

    ip_ma = _numeric_series(df, "Ip_MA")
    bt_t = _numeric_series(df, "Bt_T")
    r_m = _numeric_series(df, "R_m")
    a_m = _numeric_series(df, "a_m")
    kappa = _numeric_series(df, "kappa")
    pin_mw = _numeric_series(df, "Pin_MW")
    ne_20 = (
        _numeric_series(df, "ne_20")
        if "ne_20" in df.columns
        else _numeric_series(df, "fuel_density_m3") / NE_20_REFERENCE_DENSITY_M3
    )
    ion_mass_amu = (
        _numeric_series(df, "M_amu")
        if "M_amu" in df.columns
        else pd.Series(2.5, index=df.index, dtype=float)
    )
    epsilon = a_m / r_m
    valid_mask = cast(
        pd.Series,
        (ip_ma > 0)
        & (bt_t > 0)
        & (r_m > 0)
        & (a_m > 0)
        & (kappa > 0)
        & (ne_20 > 0)
        & (ion_mass_amu > 0)
        & (pin_mw > 0)
        & (epsilon > 0),
    )
    valid_mask_array = cast(np.ndarray, valid_mask.to_numpy(dtype=bool))
    ip_ma_array = cast(np.ndarray, ip_ma.to_numpy(dtype=float))
    bt_t_array = cast(np.ndarray, bt_t.to_numpy(dtype=float))
    r_m_array = cast(np.ndarray, r_m.to_numpy(dtype=float))
    epsilon_array = cast(np.ndarray, epsilon.to_numpy(dtype=float))
    kappa_array = cast(np.ndarray, kappa.to_numpy(dtype=float))
    ne_20_array = cast(np.ndarray, ne_20.to_numpy(dtype=float))
    ion_mass_amu_array = cast(np.ndarray, ion_mass_amu.to_numpy(dtype=float))
    pin_mw_array = cast(np.ndarray, pin_mw.to_numpy(dtype=float))

    tau_e_ipb98 = np.full(len(df), np.nan, dtype=float)
    tau_e_ipb98[valid_mask_array] = (
        0.0562
        * np.power(ip_ma_array[valid_mask_array], 0.93)
        * np.power(bt_t_array[valid_mask_array], 0.15)
        * np.power(r_m_array[valid_mask_array], 1.97)
        * np.power(epsilon_array[valid_mask_array], 0.58)
        * np.power(kappa_array[valid_mask_array], 0.78)
        * np.power(ne_20_array[valid_mask_array], 0.41)
        * np.power(ion_mass_amu_array[valid_mask_array], 0.19)
        * np.power(pin_mw_array[valid_mask_array], -0.69)
    )
    df["tau_E_ipb98_s"] = tau_e_ipb98
    return df


def _derive_ne_20_series(df: pd.DataFrame) -> pd.Series:
    derived_ne_20 = _numeric_series(df, "fuel_density_m3") / NE_20_REFERENCE_DENSITY_M3
    if "ne_20" not in df.columns:
        return derived_ne_20
    provided_ne_20 = _numeric_series(df, "ne_20")
    return cast(pd.Series, provided_ne_20.fillna(derived_ne_20))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    validate_physics_dataframe(
        df,
        required_fields=("fuel_density_m3", "temperature_keV", "confinement_time_s"),
        optional_fields=OPTIONAL_PHYSICS_COLUMNS,
    )
    fuel_density = _numeric_series(df, "fuel_density_m3")
    temperature = _numeric_series(df, "temperature_keV")
    confinement_time = _numeric_series(df, "confinement_time_s")
    df["ne_20"] = _derive_ne_20_series(df)

    df["triple_product"] = fuel_density * temperature * confinement_time
    triple_product = _numeric_series(df, "triple_product")
    df["lawson_ratio"] = triple_product / LAWSON_DT_IGNITION
    df["density_temp"] = fuel_density * temperature
    df["density_tau"] = fuel_density * confinement_time
    purity = _numeric_series(df, "fuel_purity") if "fuel_purity" in df.columns else pd.Series(1.0, index=df.index, dtype=float)
    df["purity_weighted_density"] = fuel_density * purity

    if "fuel_density_m3" in df.columns:
        df["log_fuel_density_m3"] = np.log1p(fuel_density)
    if "temperature_keV" in df.columns:
        df["log_temperature_keV"] = np.log1p(temperature)
    if "confinement_time_s" in df.columns:
        df["log_confinement_time_s"] = np.log1p(confinement_time)
    if "energy_input_MJ" in df.columns:
        df["log_energy_input_MJ"] = np.log1p(_numeric_series(df, "energy_input_MJ"))
    if "pressure_Pa" in df.columns:
        df["log_pressure_Pa"] = np.log1p(_numeric_series(df, "pressure_Pa"))
    df["log_triple_product"] = np.log1p(triple_product)

    if TARGET_COLUMN in df.columns:
        target = _numeric_series(df, TARGET_COLUMN)
        df[TARGET_LOG_COLUMN] = np.log1p(target.clip(lower=0))

    df = add_ipb98_proxy(df)
    return df


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    candidate_columns = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS
    available_columns = [column for column in candidate_columns if column in df.columns]
    return [
        column
        for column in available_columns
        if column not in LEAKAGE_COLUMNS
        and column != TARGET_COLUMN
        and _numeric_series(df, column).notna().any()
    ]


def align_to_feature_schema(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned_df = df.copy()
    for column in feature_columns:
        if column not in aligned_df.columns:
            aligned_df[column] = np.nan
    return aligned_df.loc[:, feature_columns]


def build_prepared_dataset_output_path() -> Path:
    ensure_project_directories()
    prepared_dir = config.get_data_processed_dir() / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    return prepared_dir / f"fusion_dataset_processed_{artifact_id}.csv"


def add_source_identity_columns(df: pd.DataFrame, *, start_index: int = 0) -> pd.DataFrame:
    conflicting_columns = [column for column in ROW_IDENTITY_COLUMNS if column in df.columns]
    if conflicting_columns:
        raise ValueError(
            "Input dataset already contains reserved row identity columns: "
            f"{', '.join(conflicting_columns)}."
        )
    identified_df = df.copy()
    row_indices = np.arange(start_index, start_index + len(identified_df), dtype=int)
    identified_df[ORIGINAL_ROW_INDEX_COLUMN] = row_indices
    identified_df[RAW_CSV_ROW_NUMBER_COLUMN] = row_indices + 2
    return identified_df


def _build_preprocessing_contract_compatibility_payload() -> dict[str, object]:
    return {
        "contract_version": PREPROCESSING_CONTRACT_VERSION,
        "required_physics_columns": list(REQUIRED_PHYSICS_COLUMNS),
        "optional_physics_columns": list(OPTIONAL_PHYSICS_COLUMNS),
        "base_feature_columns": list(BASE_FEATURE_COLUMNS),
        "engineered_feature_columns": list(ENGINEERED_FEATURE_COLUMNS),
        "leakage_columns": list(LEAKAGE_COLUMNS),
        "group_column": GROUP_COLUMN,
        "target_column": TARGET_COLUMN,
        "ne_20_reference_density_m3": NE_20_REFERENCE_DENSITY_M3,
        "ne_20_relative_tolerance": NE_20_RELATIVE_TOLERANCE,
        "ne_20_absolute_tolerance": NE_20_ABSOLUTE_TOLERANCE,
    }


def _normalize_fingerprint_value(value: object) -> object:
    if isinstance(value, CodeType):
        return {
            "argcount": value.co_argcount,
            "posonlyargcount": value.co_posonlyargcount,
            "kwonlyargcount": value.co_kwonlyargcount,
            "nlocals": value.co_nlocals,
            "stacksize": value.co_stacksize,
            "flags": value.co_flags,
            "code_hex": value.co_code.hex(),
            "names": list(value.co_names),
            "varnames": list(value.co_varnames),
            "freevars": list(value.co_freevars),
            "cellvars": list(value.co_cellvars),
            "consts": [_normalize_fingerprint_value(constant) for constant in value.co_consts],
        }
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, tuple):
        return [_normalize_fingerprint_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_fingerprint_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_normalize_fingerprint_value(item) for item in value]
        return sorted(normalized_items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, dict):
        return {
            str(key): _normalize_fingerprint_value(inner_value)
            for key, inner_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@lru_cache(maxsize=1)
def _features_module_source() -> str | None:
    try:
        return Path(__file__).read_text(encoding="utf-8")
    except OSError:
        return None


@lru_cache(maxsize=1)
def _features_module_definition_sources() -> dict[str, str]:
    module_source = _features_module_source()
    if module_source is None:
        return {}
    module_ast = ast.parse(module_source)
    definition_sources: dict[str, str] = {}
    for node in module_ast.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        source_segment = ast.get_source_segment(module_source, node)
        if source_segment is None:
            continue
        definition_sources[node.name] = source_segment
    return definition_sources


def _normalize_source_tokens(source_text: str) -> str:
    normalized_tokens: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source_text).readline):
        if token.type in {
            tokenize.COMMENT,
            tokenize.ENCODING,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
            tokenize.NEWLINE,
            tokenize.NL,
        }:
            continue
        normalized_tokens.append(f"{token.type}:{token.string}")
    return "\n".join(normalized_tokens)


def _fingerprint_callable_from_source(function: Callable[..., object]) -> str | None:
    definition_sources = _features_module_definition_sources()
    source_text = definition_sources.get(function.__name__)
    if source_text is None:
        return None
    normalized_source = _normalize_source_tokens(source_text)
    return _compute_preprocessing_contract_hash({"normalized_source": normalized_source})


def _fingerprint_callable_from_code_object(function: Callable[..., object]) -> str:
    payload = {
        "qualname": function.__qualname__,
        "module": function.__module__,
        "defaults": _normalize_fingerprint_value(function.__defaults__),
        "kwdefaults": _normalize_fingerprint_value(function.__kwdefaults__),
        "code": _normalize_fingerprint_value(function.__code__),
    }
    return _compute_preprocessing_contract_hash(payload)


def _fingerprint_callable(
    function: Callable[..., object],
    *,
    fingerprint_method: str,
) -> str:
    if fingerprint_method == PREPROCESSING_LOGIC_FINGERPRINT_METHOD:
        source_fingerprint = _fingerprint_callable_from_source(function)
        if source_fingerprint is not None:
            return source_fingerprint
    return _fingerprint_callable_from_code_object(function)


def _build_preprocessing_logic_fingerprints(*, fingerprint_method: str) -> dict[str, str]:
    return {
        "to_kev": _fingerprint_callable(to_kev, fingerprint_method=fingerprint_method),
        "_convert_temperature_series": _fingerprint_callable(
            _convert_temperature_series,
            fingerprint_method=fingerprint_method,
        ),
        "_temperature_series_with_units": _fingerprint_callable(
            _temperature_series_with_units,
            fingerprint_method=fingerprint_method,
        ),
        "standardize_temperature_column": _fingerprint_callable(
            standardize_temperature_column,
            fingerprint_method=fingerprint_method,
        ),
        "aggregate_time_resolved_shots_at_cutoff": _fingerprint_callable(
            aggregate_time_resolved_shots_at_cutoff,
            fingerprint_method=fingerprint_method,
        ),
        "add_ipb98_proxy": _fingerprint_callable(add_ipb98_proxy, fingerprint_method=fingerprint_method),
        "_derive_ne_20_series": _fingerprint_callable(_derive_ne_20_series, fingerprint_method=fingerprint_method),
        "align_to_feature_schema": _fingerprint_callable(align_to_feature_schema, fingerprint_method=fingerprint_method),
        "engineer_features": _fingerprint_callable(engineer_features, fingerprint_method=fingerprint_method),
        "get_model_feature_columns": _fingerprint_callable(
            get_model_feature_columns,
            fingerprint_method=fingerprint_method,
        ),
        "validate_physics_dataframe": _fingerprint_callable(
            validate_physics_dataframe,
            fingerprint_method=fingerprint_method,
        ),
    }


def _compute_preprocessing_contract_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_preprocessing_contract_payload(*, fingerprint_method: str) -> dict[str, object]:
    compatibility_payload = _build_preprocessing_contract_compatibility_payload()
    logic_fingerprints = _build_preprocessing_logic_fingerprints(fingerprint_method=fingerprint_method)
    return {
        **compatibility_payload,
        "lawson_dt_ignition": LAWSON_DT_IGNITION,
        "supported_temperature_units": list(SUPPORTED_TEMPERATURE_UNITS),
        "source_fingerprint_method": fingerprint_method,
        "logic_fingerprints": logic_fingerprints,
    }


PREPROCESSING_CONTRACT_COMPARISON_FIELDS = tuple(
    _build_preprocessing_contract_payload(
        fingerprint_method=PREPROCESSING_LOGIC_FINGERPRINT_METHOD,
    ).keys()
) + (
    "source_sha256",
    "sha256",
)
PREPROCESSING_CONTRACT_CORE_FIELDS = tuple(_build_preprocessing_contract_compatibility_payload().keys()) + (
    "lawson_dt_ignition",
    "supported_temperature_units",
)
PREPROCESSING_CONTRACT_LEGACY_OPTIONAL_CORE_FIELDS = (
    "ne_20_reference_density_m3",
    "ne_20_relative_tolerance",
    "ne_20_absolute_tolerance",
)
PREPROCESSING_CONTRACT_FINGERPRINT_FIELDS = (
    "source_fingerprint_method",
    "logic_fingerprints",
    "source_sha256",
    "sha256",
)


def normalize_preprocessing_contract(contract: Mapping[str, object]) -> dict[str, object]:
    return {
        key: contract.get(key)
        for key in PREPROCESSING_CONTRACT_COMPARISON_FIELDS
    }


def preprocessing_contract_matches(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return normalize_preprocessing_contract(left) == normalize_preprocessing_contract(right)


def describe_preprocessing_contract_differences(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> list[str]:
    left_contract = normalize_preprocessing_contract(left)
    right_contract = normalize_preprocessing_contract(right)
    return [
        field
        for field in PREPROCESSING_CONTRACT_COMPARISON_FIELDS
        if left_contract.get(field) != right_contract.get(field)
    ]


def assess_runtime_preprocessing_contract_compatibility(
    saved_contract: Mapping[str, object],
    current_contract: Mapping[str, object],
    *,
    legacy_runtime_contract: Mapping[str, object] | None = None,
) -> PreprocessingContractCompatibilityReport:
    compatibility_warnings: list[str] = []
    core_differences: list[str] = []
    for field in PREPROCESSING_CONTRACT_CORE_FIELDS:
        if saved_contract.get(field) == current_contract.get(field):
            continue
        if field in PREPROCESSING_CONTRACT_LEGACY_OPTIONAL_CORE_FIELDS and field not in saved_contract:
            compatibility_warnings.append(
                "Runtime source compatibility was accepted using legacy preprocessing contract defaults for "
                f"{field}; retrain to persist the explicit value."
            )
            continue
        core_differences.append(field)
    if core_differences:
        fingerprint_differences = [
            field
            for field in PREPROCESSING_CONTRACT_FINGERPRINT_FIELDS
            if saved_contract.get(field) != current_contract.get(field)
        ]
        return PreprocessingContractCompatibilityReport(
            compatible=False,
            differing_fields=tuple(core_differences + fingerprint_differences),
        )

    saved_method = saved_contract.get("source_fingerprint_method")
    current_method = current_contract.get("source_fingerprint_method")
    if saved_method == current_method:
        differing_fields = [
            field
            for field in PREPROCESSING_CONTRACT_FINGERPRINT_FIELDS
            if saved_contract.get(field) != current_contract.get(field)
        ]
        return PreprocessingContractCompatibilityReport(
            compatible=not differing_fields,
            differing_fields=tuple(differing_fields),
            warnings=tuple(compatibility_warnings),
        )

    if saved_method == LEGACY_PREPROCESSING_LOGIC_FINGERPRINT_METHOD and legacy_runtime_contract is not None:
        legacy_fingerprint_differences = [
            field
            for field in PREPROCESSING_CONTRACT_FINGERPRINT_FIELDS
            if saved_contract.get(field) != legacy_runtime_contract.get(field)
        ]
        if not legacy_fingerprint_differences:
            return PreprocessingContractCompatibilityReport(
                compatible=True,
                differing_fields=(),
                warnings=tuple(compatibility_warnings),
            )

    if saved_method == LEGACY_PREPROCESSING_LOGIC_FINGERPRINT_METHOD:
        return PreprocessingContractCompatibilityReport(
            compatible=True,
            differing_fields=("source_fingerprint_method",),
            warnings=(
                *compatibility_warnings,
                "Saved model uses legacy bytecode-based preprocessing fingerprints. "
                "Runtime source compatibility was accepted using the stable contract fields only; "
                "retrain to refresh the artifact with source-based fingerprints.",
            ),
        )

    differing_fields = ["source_fingerprint_method"]
    differing_fields.extend(
        field
        for field in ("logic_fingerprints", "source_sha256", "sha256")
        if saved_contract.get(field) != current_contract.get(field)
    )
    return PreprocessingContractCompatibilityReport(
        compatible=False,
        differing_fields=tuple(differing_fields),
    )


def build_preprocessing_contract(
    *,
    fingerprint_method: str = PREPROCESSING_LOGIC_FINGERPRINT_METHOD,
) -> dict[str, object]:
    contract_payload = _build_preprocessing_contract_payload(
        fingerprint_method=fingerprint_method,
    )
    logic_payload = cast(dict[str, object], contract_payload["logic_fingerprints"])
    return {
        **contract_payload,
        "source_sha256": _compute_preprocessing_contract_hash(logic_payload),
        "sha256": _compute_preprocessing_contract_hash(contract_payload),
    }


def prepare_dataset(
    dataset_path: Optional[Union[str, Path]] = None,
    *,
    allow_synthetic: bool = False,
    processed_output_path: Optional[Union[str, Path]] = None,
    assume_temperature_unit: str | None = None,
    shot_prediction_cutoff_rows: int = DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS,
    synthetic_output_path: Optional[Union[str, Path]] = None,
    synthetic_n_rows: int = SYNTHETIC_DATASET_ROWS,
    synthetic_random_state: int = RANDOM_STATE,
) -> PreparedDataset:
    raw_path, dataset_source_kind, requested_dataset_path = resolve_training_dataset_path(
        dataset_path,
        allow_synthetic=allow_synthetic,
        synthetic_output_path=(
            Path(synthetic_output_path).expanduser().resolve()
            if synthetic_output_path is not None
            else None
        ),
        synthetic_n_rows=synthetic_n_rows,
        synthetic_random_state=synthetic_random_state,
    )
    raw_df = pd.read_csv(raw_path)
    audit_summary = audit_dataframe(raw_df)
    raw_df = add_source_identity_columns(raw_df)
    prepared_frame = prepare_model_frame(
        raw_df,
        assume_temperature_unit=assume_temperature_unit,
        shot_prediction_cutoff_rows=shot_prediction_cutoff_rows,
        require_target=True,
    )
    canonical_df = prepared_frame.dataframe
    column_mapping = prepared_frame.column_mapping

    candidate_feature_columns = [
        column for column in BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS if column in canonical_df.columns
    ]
    processed_path = (
        Path(processed_output_path).expanduser().resolve()
        if processed_output_path is not None
        else build_prepared_dataset_output_path()
    )
    write_dataframe_csv_atomic(processed_path, canonical_df, index=False)

    return PreparedDataset(
        raw_path=raw_path,
        processed_path=processed_path,
        dataframe=canonical_df,
        audit_summary=audit_summary,
        column_mapping=column_mapping,
        candidate_feature_columns=candidate_feature_columns,
        dataset_source_kind=dataset_source_kind,
        synthetic_data_used=dataset_source_kind == "synthetic_generated",
        requested_dataset_path=requested_dataset_path,
        synthetic_random_state=synthetic_random_state if dataset_source_kind == "synthetic_generated" else None,
        synthetic_row_count=int(len(raw_df)) if dataset_source_kind == "synthetic_generated" else None,
    )
