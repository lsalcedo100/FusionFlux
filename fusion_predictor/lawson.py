from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from config import LAWSON_DT_IGNITION
from validation import validate_physics_value, validate_positive_finite

EV_TO_K = 1.160451812e4
KEV_TO_K = 1e3 * EV_TO_K


@dataclass(frozen=True)
class LawsonResult:
    triple_product: float
    lawson_ratio: float
    status: str

def ev_to_kelvin(temperature_ev: float) -> float:
    return validate_positive_finite(temperature_ev, "temperature_ev") * EV_TO_K


def kelvin_to_ev(temperature_k: float) -> float:
    return validate_positive_finite(temperature_k, "temperature_k") / EV_TO_K


def kev_to_kelvin(temperature_kev: float) -> float:
    return validate_positive_finite(temperature_kev, "temperature_kev") * KEV_TO_K


def kelvin_to_kev(temperature_k: float) -> float:
    return validate_positive_finite(temperature_k, "temperature_k") / KEV_TO_K


def ev_to_kev(temperature_ev: float) -> float:
    return validate_positive_finite(temperature_ev, "temperature_ev") / 1e3


def kev_to_ev(temperature_kev: float) -> float:
    return validate_positive_finite(temperature_kev, "temperature_kev") * 1e3


def to_kev(temperature: float, unit: str = "keV") -> float:
    normalized_unit = unit.strip().lower()
    if normalized_unit == "kev":
        return float(validate_physics_value(temperature, "temperature_keV"))
    if normalized_unit == "ev":
        return ev_to_kev(temperature)
    if normalized_unit == "k":
        return kelvin_to_kev(temperature)
    raise ValueError("unit must be 'keV', 'eV', or 'K'")


def calculate_lawson_status(
    density_m3: float,
    temperature: float,
    confinement_time_s: float,
    temp_unit: str = "keV",
) -> LawsonResult:
    density_m3 = float(validate_physics_value(density_m3, "fuel_density_m3"))
    confinement_time_s = float(validate_physics_value(confinement_time_s, "confinement_time_s"))
    temperature_kev = to_kev(temperature, temp_unit)

    triple_product = density_m3 * temperature_kev * confinement_time_s
    lawson_ratio = triple_product / LAWSON_DT_IGNITION
    status = "IGNITION REACHED" if lawson_ratio >= 1.0 else "SUB-CRITICAL"
    return LawsonResult(
        triple_product=triple_product,
        lawson_ratio=lawson_ratio,
        status=status,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lawson Criterion calculator for D-T fusion.")
    parser.add_argument("--density-m3", type=float, required=True, help="Ion density in m^-3.")
    parser.add_argument("--temperature", type=float, required=True, help="Ion temperature.")
    parser.add_argument(
        "--temp-unit",
        type=str,
        default="keV",
        choices=["keV", "eV", "K"],
        help="Temperature unit.",
    )
    parser.add_argument(
        "--confinement-time-s",
        type=float,
        required=True,
        help="Energy confinement time in seconds.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = calculate_lawson_status(
        density_m3=args.density_m3,
        temperature=args.temperature,
        confinement_time_s=args.confinement_time_s,
        temp_unit=args.temp_unit,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
