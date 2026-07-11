from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, overload

from config import (
    NE_20_ABSOLUTE_TOLERANCE,
    NE_20_REFERENCE_DENSITY_M3,
    NE_20_RELATIVE_TOLERANCE,
    TARGET_COLUMN,
)

if TYPE_CHECKING:
    import pandas as pd

class _PandasModule(Protocol):
    def isna(self, value: object) -> object: ...

    def to_numeric(self, arg: object, errors: str = "raise") -> Any: ...


def _format_ne_20_reference_density() -> str:
    return f"{NE_20_REFERENCE_DENSITY_M3:.0e}".replace("+", "")


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


def _get_numeric_rule(name: str) -> NumericRule:
    try:
        return PHYSICS_INPUT_RULES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown validation field: {name}") from exc


def is_boolean_like(value: object) -> bool:
    value_type = type(value)
    return isinstance(value, bool) or (
        value_type.__module__ == "numpy" and value_type.__name__ in {"bool", "bool_"}
    )


@overload
def validate_physics_value(value: object, name: str, allow_none: Literal[False] = False) -> float: ...


@overload
def validate_physics_value(value: object, name: str, allow_none: Literal[True]) -> float | None: ...


def validate_physics_value(value: object, name: str, allow_none: bool = False) -> float | None:
    rule = _get_numeric_rule(name)
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required.")

    numeric_value = _coerce_float(value, name)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be {rule.description}.")

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
    numeric_value = _coerce_float(value, name)
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return numeric_value


def _coerce_float(value: object, name: str) -> float:
    if is_boolean_like(value):
        raise ValueError(f"{name} must be a finite number, not a boolean.")
    try:
        return float(value if isinstance(value, (str, bytes, bytearray)) else cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc


def _is_missing_dataframe_value(value: object, *, pandas_module: _PandasModule | None = None) -> bool:
    pandas_module = _require_pandas() if pandas_module is None else pandas_module
    if isinstance(value, str):
        return value.strip() == ""
    return bool(pandas_module.isna(value))


def _require_pandas() -> _PandasModule:
    try:
        import pandas as pandas_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for dataframe validation. Install project dependencies "
            "or avoid validate_physics_dataframe in minimal Lawson-only environments."
        ) from exc
    return cast(_PandasModule, pandas_module)


def validate_physics_inputs(
    values: Mapping[str, object],
    *,
    required_fields: tuple[str, ...],
    optional_fields: tuple[str, ...] = (),
) -> dict[str, float | None]:
    validated: dict[str, float | None] = {}
    for field in required_fields:
        validated[field] = validate_physics_value(values.get(field), field)
    for field in optional_fields:
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
    pandas = _require_pandas()
    missing_columns = [field for field in required_fields if field not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns after mapping: {missing_columns}")

    invalid_messages: list[str] = []
    for field in (*required_fields, *optional_fields):
        if field not in df.columns:
            continue
        rule = _get_numeric_rule(field)
        raw_series = df[field]
        series = pandas.to_numeric(raw_series, errors="coerce")
        missing_mask = raw_series.map(lambda value: _is_missing_dataframe_value(value, pandas_module=pandas))
        boolean_mask = raw_series.map(is_boolean_like) & ~missing_mask
        invalid_numeric_mask = series.isna() & ~missing_mask & ~boolean_mask

        if field in required_fields and missing_mask.any():
            invalid_messages.append(
                _format_invalid_message(field, df.index[missing_mask].tolist(), "present")
            )
        if boolean_mask.any():
            invalid_messages.append(
                _format_invalid_message(field, df.index[boolean_mask].tolist(), "numeric, not boolean")
            )
        if invalid_numeric_mask.any():
            invalid_messages.append(
                _format_invalid_message(field, df.index[invalid_numeric_mask].tolist(), "numeric when provided")
            )

        series_to_check = series[~missing_mask & ~boolean_mask & ~invalid_numeric_mask]
        if series_to_check.empty:
            continue

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
            pandas.to_numeric(df["R_m"], errors="coerce").notna()
            & pandas.to_numeric(df["a_m"], errors="coerce").notna()
            & (
                pandas.to_numeric(df["a_m"], errors="coerce")
                >= pandas.to_numeric(df["R_m"], errors="coerce")
            )
        )
        if geometry_mask.any():
            invalid_messages.append(
                _format_invalid_message("a_m", df.index[geometry_mask].tolist(), "smaller than R_m")
            )

    if {"fuel_density_m3", "ne_20"}.issubset(df.columns):
        density = pandas.to_numeric(df["fuel_density_m3"], errors="coerce")
        ne_20 = pandas.to_numeric(df["ne_20"], errors="coerce")
        comparable_mask = density.notna() & ne_20.notna()
        if comparable_mask.any():
            expected_ne_20 = density[comparable_mask] / NE_20_REFERENCE_DENSITY_M3
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
                        f"consistent with fuel_density_m3 / {_format_ne_20_reference_density()}",
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

    derived_ne_20 = fuel_density / NE_20_REFERENCE_DENSITY_M3
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
            f"ne_20 must match fuel_density_m3 / {_format_ne_20_reference_density()} within tolerance "
            f"(expected {derived_ne_20:.6g}, got {provided_ne_20:.6g})."
        )
    values["ne_20"] = derived_ne_20
