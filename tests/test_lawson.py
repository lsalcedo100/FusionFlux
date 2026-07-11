import math
import subprocess
import sys
from pathlib import Path

import pytest

from lawson import calculate_lawson_status, kelvin_to_kev, kev_to_kelvin, to_kev


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


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        calculate_lawson_status(-1e20, 15, 4)
    with pytest.raises(ValueError):
        calculate_lawson_status(1e20, 0, 4)


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
