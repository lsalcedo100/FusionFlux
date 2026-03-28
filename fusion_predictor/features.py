from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, cast

import numpy as np
import pandas as pd

from config import (
    BASE_FEATURE_COLUMNS,
    COLUMN_ALIASES,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    ENGINEERED_FEATURE_COLUMNS,
    GROUP_COLUMN,
    LAWSON_DT_IGNITION,
    LEAKAGE_COLUMNS,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    SYNTHETIC_DATASET_ROWS,
    TARGET_COLUMN,
    TARGET_LOG_COLUMN,
)
from lawson import to_kev
from validation import validate_physics_dataframe

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


@dataclass
class PreparedDataset:
    raw_path: Path
    processed_path: Path
    dataframe: pd.DataFrame
    audit_summary: dict[str, object]
    column_mapping: dict[str, str]
    feature_columns: list[str]
    dataset_source_kind: str
    synthetic_data_used: bool
    requested_dataset_path: Path | None


def ensure_project_directories() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def create_synthetic_dataset(
    output_path: Optional[Path] = None,
    n_rows: int = SYNTHETIC_DATASET_ROWS,
    random_state: int = RANDOM_STATE,
) -> Path:
    ensure_project_directories()
    output_path = output_path or (DATA_RAW_DIR / "synthetic_nuclear_fusion_experiment.csv")
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

    ne_20 = fuel_density_m3 / 1e20
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
    df.to_csv(output_path, index=False)
    return output_path


def resolve_training_dataset_path(
    dataset_path: Optional[Union[str, Path]] = None,
    *,
    allow_synthetic: bool = False,
) -> tuple[Path, str, Path | None]:
    ensure_project_directories()
    if dataset_path is not None:
        resolved = Path(dataset_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset not found: {resolved}")
        return resolved, "provided_path", resolved

    if allow_synthetic:
        synthetic_path = create_synthetic_dataset()
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
                break
    return rename_map


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


def standardize_temperature_column(df: pd.DataFrame) -> pd.DataFrame:
    if "temperature_keV" in df.columns:
        return df
    if "temperature_eV" in df.columns:
        temperature_ev = _numeric_series(df, "temperature_eV")
        df["temperature_keV"] = temperature_ev / 1e3
        return df
    if "temperature_K" in df.columns:
        temperature_k = _numeric_series(df, "temperature_K")
        df["temperature_keV"] = temperature_k.apply(
            lambda value: to_kev(value, "K") if pd.notna(value) else np.nan
        )
        return df
    if "temperature" in df.columns and "temperature_unit" in df.columns:
        df["temperature_keV"] = [
            to_kev(value, unit) if pd.notna(value) and pd.notna(unit) else np.nan
            for value, unit in zip(df["temperature"], df["temperature_unit"])
        ]
        return df
    if "temperature" in df.columns:
        warnings.warn(
            "A generic 'temperature' column was found without units. Assuming keV.",
            stacklevel=2,
        )
        df["temperature_keV"] = pd.to_numeric(df["temperature"], errors="coerce")
        return df
    raise ValueError("No temperature column could be mapped to a canonical temperature_keV field.")


def aggregate_time_resolved_shots(df: pd.DataFrame) -> pd.DataFrame:
    time_column = None
    if "time_s" in df.columns:
        time_column = "time_s"
    elif "time_ms" in df.columns:
        time_column = "time_ms"
        time_ms = _numeric_series(df, "time_ms")
        df["time_s"] = time_ms / 1e3
        time_column = "time_s"

    if GROUP_COLUMN not in df.columns or time_column is None:
        return df

    if df[GROUP_COLUMN].nunique() == len(df):
        return df

    aggregated_rows: list[dict[str, object]] = []
    for shot_id, group in df.sort_values(time_column).groupby(GROUP_COLUMN):
        cutoff = max(1, int(np.ceil(len(group) * 0.8)))
        pre_target_window = group.iloc[:cutoff]
        record: dict[str, object] = {GROUP_COLUMN: shot_id}

        for column in group.columns:
            if column == GROUP_COLUMN:
                continue
            group_column = _series(group, column)
            pre_target_column = _series(pre_target_window, column)
            if pd.api.types.is_numeric_dtype(group_column):
                if column == TARGET_COLUMN:
                    record[column] = float(group_column.max())
                else:
                    record[column] = float(pre_target_column.median())
            else:
                mode = pre_target_column.mode(dropna=True)
                record[column] = mode.iloc[0] if not mode.empty else pre_target_column.iloc[0]
        aggregated_rows.append(record)

    return pd.DataFrame(aggregated_rows)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    validate_physics_dataframe(
        df,
        required_fields=REQUIRED_PHYSICS_COLUMNS,
        optional_fields=OPTIONAL_PHYSICS_COLUMNS,
    )
    return df


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
    ne_20 = _numeric_series(df, "ne_20") if "ne_20" in df.columns else _numeric_series(df, "fuel_density_m3") / 1e20
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
    return [column for column in available_columns if column not in LEAKAGE_COLUMNS and column != TARGET_COLUMN]


def prepare_dataset(
    dataset_path: Optional[Union[str, Path]] = None,
    *,
    allow_synthetic: bool = False,
) -> PreparedDataset:
    raw_path, dataset_source_kind, requested_dataset_path = resolve_training_dataset_path(
        dataset_path,
        allow_synthetic=allow_synthetic,
    )
    raw_df = pd.read_csv(raw_path)
    audit_summary = audit_dataframe(raw_df)

    rename_map = resolve_column_mapping(raw_df)
    canonical_df = raw_df.rename(columns=rename_map).copy()
    canonical_df = _coerce_numeric_columns(canonical_df)
    canonical_df = standardize_temperature_column(canonical_df)
    canonical_df = aggregate_time_resolved_shots(canonical_df)
    canonical_df = clean_dataframe(canonical_df)
    canonical_df = engineer_features(canonical_df)

    feature_columns = get_model_feature_columns(canonical_df)
    processed_path = DATA_PROCESSED_DIR / "fusion_dataset_processed.csv"
    canonical_df.to_csv(processed_path, index=False)

    return PreparedDataset(
        raw_path=raw_path,
        processed_path=processed_path,
        dataframe=canonical_df,
        audit_summary=audit_summary,
        column_mapping=rename_map,
        feature_columns=feature_columns,
        dataset_source_kind=dataset_source_kind,
        synthetic_data_used=dataset_source_kind == "synthetic_generated",
        requested_dataset_path=requested_dataset_path,
    )
