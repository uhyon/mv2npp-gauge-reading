import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "filter_measurebench", Path(__file__).resolve().parents[1] / "scripts" / "filter_measurebench.py"
)
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)


def test_flatten_handles_strings_and_nesting():
    assert fm.flatten("bar") == ["bar"]
    assert fm.flatten(["bar", ["psi", ["kPa"]]]) == ["bar", "psi", "kPa"]


@pytest.mark.parametrize(
    "u",
    ["bar", "kgf/cm²", "kgf/cm2", "kg/cm2", "KG/CM²", "mbar", "MPa", "kPa", "Pa", "inHg", "psi",
     "mmHg", "millimeter of mercury", "°C", "℃", "degree Celsius", "°F", "Fahrenheit", "mmH2O"],
)
def test_unit_variants_accepted(u):
    assert fm.unit_kind(u) is not None


@pytest.mark.parametrize("u", ["A", "Ampere", "V", "rpm", "km/h", "%"])
def test_other_units_rejected(u):
    assert fm.unit_kind(u) is None


def test_unit_ok_flat_and_nested():
    assert fm.unit_ok(["bar"])
    assert fm.unit_ok([["A", "Ampere"], ["°C"]])
    assert not fm.unit_ok([["A", "Ampere"]])
    assert not fm.unit_ok([1, None])


def test_first_interval():
    assert fm.first_interval([3, 3.5]) == (3.0, 3.5)
    assert fm.first_interval([[10, 11], [22, 23]]) == (10.0, 11.0)
    assert fm.first_interval([]) is None


def test_pick_single_scale():
    assert fm.pick({"interval": [3, 3.5], "units": ["bar"]}) == ("bar", (3.0, 3.5), 1, "ok")


def test_pick_dual_scale_pairs_interval_with_unit():
    kw = {"interval": [[40, 45], [3, 3.1]], "units": [["psi"], ["bar"]]}
    assert fm.pick(kw) == ("psi", (40.0, 45.0), 2, "ok_paired")
    kw = {"interval": [[1, 2], [3, 3.1]], "units": [["A"], ["bar"]]}
    assert fm.pick(kw) == ("bar", (3.0, 3.1), 2, "ok_paired")


def test_pick_alternative_intervals_one_unit_group():
    kw = {"interval": [[3, 3.5], [3.2, 3.4]], "units": ["bar", "Bar"]}
    assert fm.pick(kw) == ("bar", (3.0, 3.5), 2, "ok")


def test_pick_reasons():
    assert fm.pick({"interval": [1, 2], "units": ["A"]})[3] == "unit"
    assert fm.pick({"interval": [], "units": ["bar"]})[3] == "no_interval"
    kw = {"interval": [[1, 2], [3, 4], [5, 6]], "units": [["bar"], ["psi"]]}
    assert fm.pick(kw)[3] == "interval_unit_count_mismatch"
