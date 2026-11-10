import pytest

from gauge_reader.units import convert


@pytest.mark.parametrize("v, src, dst, expected", [
    (1, "bar", "kPa", 100.0),
    (1, "psi", "kPa", 6.894757),
    (1, "kgf/cm²", "kPa", 98.0665),
    (1, "kg/cm2", "bar", 0.980665),
    (1, "MPa", "bar", 10.0),
    (1, "atm", "hPa", 1013.25),
    (760, "mmHg", "hPa", 1013.25),
    (29.92, "inHg", "hPa", 1013.21),
    (1, "mbar", "hPa", 1.0),
    (1000, "mmH2O", "kPa", 9.80665),
    (100, "°C", "°F", 212.0),
    (32, "Fahrenheit", "℃", 0.0),
])
def test_conversion_factors(v, src, dst, expected):
    assert convert(v, src, dst) == pytest.approx(expected, rel=1e-4)


def test_incompatible_units():
    assert convert(1, "bar", "°C") is None
    assert convert(1, "A", "V") is None
