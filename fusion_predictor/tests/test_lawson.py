import math

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
