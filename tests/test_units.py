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


@pytest.mark.parametrize("u", ["inches of water", "inch of water", "inH2O", "in H2O", "in. w.c.", '"H2O'])
def test_inches_of_water_is_not_inhg(u):
    from gauge_reader.units import norm_unit, unit_kind

    assert unit_kind(u) == "pressure"
    assert norm_unit(u) not in ("inhg", "inchofmercury", "inchesofmercury")
    assert convert(1, u, "Pa") == pytest.approx(249.08891)
    assert convert(1, u, "inHg") == pytest.approx(0.073556, rel=1e-4)


def test_incompatible_units():
    assert convert(1, "bar", "°C") is None
    assert convert(1, "A", "V") is None


def test_meterfl_plurals_and_typos():
    from gauge_reader.units import convert, norm_unit, unit_kind

    for u in ["megapascals", "megapascals ", "magepascals", "megapascasl", "kilopascals"]:
        assert unit_kind(u) == "pressure"
    assert norm_unit("magepascals") == "megapascal"
    assert convert(1.1, "kilopascals", "megapascals") == 1.1e-3
    # Plain abbreviations ending in s are not stripped.
    assert norm_unit("psi") == "psi" and norm_unit("mmHg") == "mmhg"
