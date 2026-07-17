import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

import lawson
from lawson import (
    build_parser,
    calculate_lawson_status,
    ev_to_kelvin,
    ev_to_kev,
    kelvin_to_ev,
    kelvin_to_kev,
    kev_to_ev,
    kev_to_kelvin,
    main,
    to_kev,
)


def test_calculate_lawson_status_iter_reference_case():
    result = calculate_lawson_status(1e20, 15, 4, temp_unit="keV")
    assert math.isclose(result.triple_product, 6e21, rel_tol=1e-12)
    assert math.isclose(result.lawson_ratio, 2.0, rel_tol=1e-12)
    assert result.status == "IGNITION REACHED"


def test_temperature_conversion_round_trip():
    temperature_kev = 12.5
    temperature_k = kev_to_kelvin(temperature_kev)
    assert math.isclose(kelvin_to_kev(temperature_k), temperature_kev, rel_tol=1e-12)
    assert math.isclose(to_kev(12500, "eV"), 12.5, rel_tol=1e-12)


def test_ev_kelvin_conversions_round_trip():
    temperature_ev = 15000.0
    temperature_k = ev_to_kelvin(temperature_ev)
    assert math.isclose(kelvin_to_ev(temperature_k), temperature_ev, rel_tol=1e-12)


def test_ev_kev_conversions_round_trip():
    temperature_kev = 15.0
    temperature_ev = kev_to_ev(temperature_kev)
    assert math.isclose(temperature_ev, 15000.0, rel_tol=1e-12)
    assert math.isclose(ev_to_kev(temperature_ev), temperature_kev, rel_tol=1e-12)


def test_to_kev_accepts_kelvin_and_is_case_insensitive():
    temperature_kev = 12.5
    temperature_k = kev_to_kelvin(temperature_kev)
    assert math.isclose(to_kev(temperature_k, "K"), temperature_kev, rel_tol=1e-12)
    # Unit parsing is normalized, so mixed case resolves the same branch.
    assert math.isclose(to_kev(temperature_k, "k"), temperature_kev, rel_tol=1e-12)


def test_to_kev_rejects_unknown_unit():
    with pytest.raises(ValueError, match="unit must be"):
        to_kev(15.0, "furlongs")


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        calculate_lawson_status(-1e20, 15, 4)
    with pytest.raises(ValueError):
        calculate_lawson_status(1e20, 0, 4)


def test_build_parser_parses_cli_arguments():
    args = build_parser().parse_args(
        ["--density-m3", "1e20", "--temperature", "15", "--temp-unit", "keV", "--confinement-time-s", "4"]
    )
    assert args.density_m3 == 1e20
    assert args.temperature == 15.0
    assert args.temp_unit == "keV"
    assert args.confinement_time_s == 4.0


def test_main_reports_lawson_status_as_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["lawson.py", "--density-m3", "1e20", "--temperature", "15", "--confinement-time-s", "4"],
    )
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "IGNITION REACHED"
    assert math.isclose(payload["lawson_ratio"], 2.0, rel_tol=1e-12)
    assert math.isclose(payload["triple_product"], 6e21, rel_tol=1e-12)


def test_module_exposes_callable_entrypoint():
    # main() is the console entrypoint guarded by ``if __name__ == '__main__'``.
    assert callable(lawson.main)


def test_lawson_import_does_not_require_pandas():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import sys
sys.path.insert(0, '.')
sys.modules['pandas'] = None
import lawson
result = lawson.calculate_lawson_status(1e20, 15, 4)
print(result.status)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "IGNITION REACHED"
