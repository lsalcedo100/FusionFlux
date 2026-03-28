from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from config import TARGET_COLUMN

NE_20_RELATIVE_TOLERANCE = 1e-3
NE_20_ABSOLUTE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class NumericRule:
    minimum: float | None = None
    maximum: float | None = None
    min_inclusive: bool = False
    max_inclusive: bool = True
    description: str = "a finite number"


PHYSICS_INPUT_RULES: dict[str, NumericRule] = {
    "fuel_density_m3": NumericRule(minimum=0.0, description="a positive finite number"),
    "temperature_keV": NumericRule(minimum=0.0, description="a positive finite number"),
    "confinement_time_s": NumericRule(minimum=0.0, description="a positive finite number"),
    "fuel_purity": NumericRule(
        minimum=0.0,
        maximum=1.0,
        min_inclusive=True,
        max_inclusive=True,
        description="a finite number between 0 and 1 inclusive",
    ),
    "energy_input_MJ": NumericRule(minimum=0.0, description="a positive finite number"),
    "pressure_Pa": NumericRule(minimum=0.0, description="a positive finite number"),
    "Ip_MA": NumericRule(minimum=0.0, description="a positive finite number"),
    "Bt_T": NumericRule(minimum=0.0, description="a positive finite number"),
    "R_m": NumericRule(minimum=0.0, description="a positive finite number"),
    "a_m": NumericRule(minimum=0.0, description="a positive finite number"),
    "kappa": NumericRule(minimum=0.0, description="a positive finite number"),
    "ne_20": NumericRule(minimum=0.0, description="a positive finite number"),
    "M_amu": NumericRule(minimum=0.0, description="a positive finite number"),
    "Pin_MW": NumericRule(minimum=0.0, description="a positive finite number"),
    TARGET_COLUMN: NumericRule(
        minimum=0.0,
        min_inclusive=True,
        description="a finite number greater than or equal to 0",
    ),
}


def validate_physics_value(value: object, name: str, allow_none: bool = False) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required.")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be {PHYSICS_INPUT_RULES[name].description}.")

    rule = PHYSICS_INPUT_RULES[name]
    if rule.minimum is not None:
        if rule.min_inclusive and numeric_value < rule.minimum:
            raise ValueError(f"{name} must be {rule.description}.")
        if not rule.min_inclusive and numeric_value <= rule.minimum:
            raise ValueError(f"{name} must be {rule.description}.")
    if rule.maximum is not None:
        if rule.max_inclusive and numeric_value > rule.maximum:
            raise ValueError(f"{name} must be {rule.description}.")
        if not rule.max_inclusive and numeric_value >= rule.maximum:
            raise ValueError(f"{name} must be {rule.description}.")
    return numeric_value


def validate_positive_finite(value: object, name: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return numeric_value


def validate_physics_inputs(
    values: Mapping[str, object],
    *,
    required_fields: tuple[str, ...],
    optional_fields: tuple[str, ...] = (),
) -> dict[str, float | None]:
    validated: dict[str, float | None] = {}
    for field in required_fields:
        if field not in PHYSICS_INPUT_RULES:
            raise KeyError(f"Unknown validation field: {field}")
        validated[field] = validate_physics_value(values.get(field), field)
    for field in optional_fields:
        if field not in PHYSICS_INPUT_RULES:
            raise KeyError(f"Unknown validation field: {field}")
        validated[field] = validate_physics_value(values.get(field), field, allow_none=True)

    major_radius = validated.get("R_m")
    minor_radius = validated.get("a_m")
    if major_radius is not None and minor_radius is not None and minor_radius >= major_radius:
        raise ValueError("a_m must be smaller than R_m.")
    _validate_or_derive_ne_20_mapping(validated)
    return validated


def validate_physics_dataframe(
    df: pd.DataFrame,
    *,
    required_fields: tuple[str, ...],
    optional_fields: tuple[str, ...] = (),
) -> None:
    missing_columns = [field for field in required_fields if field not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns after mapping: {missing_columns}")

    invalid_messages: list[str] = []
    for field in (*required_fields, *optional_fields):
        if field not in df.columns:
            continue
        series = pd.to_numeric(df[field], errors="coerce")
        missing_mask = series.isna()
        if field in required_fields and missing_mask.any():
            invalid_messages.append(
                _format_invalid_message(field, df.index[missing_mask].tolist(), "missing or non-numeric")
            )
        series_to_check = series
        if field in optional_fields:
            series_to_check = series[~missing_mask]
            if series_to_check.empty:
                continue
        elif missing_mask.any():
            series_to_check = series[~missing_mask]
            if series_to_check.empty:
                continue

        rule = PHYSICS_INPUT_RULES[field]
        invalid_mask = ~series_to_check.map(math.isfinite)
        if rule.minimum is not None:
            if rule.min_inclusive:
                invalid_mask |= series_to_check < rule.minimum
            else:
                invalid_mask |= series_to_check <= rule.minimum
        if rule.maximum is not None:
            if rule.max_inclusive:
                invalid_mask |= series_to_check > rule.maximum
            else:
                invalid_mask |= series_to_check >= rule.maximum
        if invalid_mask.any():
            invalid_messages.append(
                _format_invalid_message(field, series_to_check.index[invalid_mask].tolist(), rule.description)
            )

    if {"R_m", "a_m"}.issubset(df.columns):
        geometry_mask = (
            pd.to_numeric(df["R_m"], errors="coerce").notna()
            & pd.to_numeric(df["a_m"], errors="coerce").notna()
            & (pd.to_numeric(df["a_m"], errors="coerce") >= pd.to_numeric(df["R_m"], errors="coerce"))
        )
        if geometry_mask.any():
            invalid_messages.append(
                _format_invalid_message("a_m", df.index[geometry_mask].tolist(), "smaller than R_m")
            )

    if {"fuel_density_m3", "ne_20"}.issubset(df.columns):
        density = pd.to_numeric(df["fuel_density_m3"], errors="coerce")
        ne_20 = pd.to_numeric(df["ne_20"], errors="coerce")
        comparable_mask = density.notna() & ne_20.notna()
        if comparable_mask.any():
            expected_ne_20 = density[comparable_mask] / 1e20
            consistent_mask = expected_ne_20.combine(
                ne_20[comparable_mask],
                lambda expected, actual: math.isclose(
                    float(expected),
                    float(actual),
                    rel_tol=NE_20_RELATIVE_TOLERANCE,
                    abs_tol=NE_20_ABSOLUTE_TOLERANCE,
                ),
            )
            inconsistent_rows = expected_ne_20.index[~consistent_mask.to_numpy(dtype=bool)].tolist()
            if inconsistent_rows:
                invalid_messages.append(
                    _format_invalid_message(
                        "ne_20",
                        inconsistent_rows,
                        "consistent with fuel_density_m3 / 1e20",
                    )
                )

    if invalid_messages:
        raise ValueError("Invalid physics inputs in dataset: " + "; ".join(invalid_messages))


def _format_invalid_message(field: str, rows: list[int], expectation: str) -> str:
    displayed_rows = ", ".join(str(row) for row in rows[:5])
    if len(rows) > 5:
        displayed_rows = f"{displayed_rows}, ..."
    return f"{field} rows [{displayed_rows}] must be {expectation}"


def _validate_or_derive_ne_20_mapping(values: dict[str, float | None]) -> None:
    fuel_density = values.get("fuel_density_m3")
    if fuel_density is None:
        return

    derived_ne_20 = fuel_density / 1e20
    provided_ne_20 = values.get("ne_20")
    if provided_ne_20 is None:
        values["ne_20"] = derived_ne_20
        return

    if not math.isclose(
        provided_ne_20,
        derived_ne_20,
        rel_tol=NE_20_RELATIVE_TOLERANCE,
        abs_tol=NE_20_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(
            "ne_20 must match fuel_density_m3 / 1e20 within tolerance "
            f"(expected {derived_ne_20:.6g}, got {provided_ne_20:.6g})."
        )
    values["ne_20"] = derived_ne_20
