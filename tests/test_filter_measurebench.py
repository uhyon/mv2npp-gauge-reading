import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "filter_measurebench", Path(__file__).resolve().parents[1] / "scripts" / "filter_measurebench.py"
)
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)


def test_flatten_handles_strings_and_nesting():
    assert fm.flatten("bar") == ["bar"]
    assert fm.flatten(["bar", ["psi", ["kPa"]]]) == ["bar", "psi", "kPa"]


def test_unit_ok_flat_and_nested():
    assert fm.unit_ok(["bar"])
    assert fm.unit_ok([["A", "Ampere"], ["°C"]])
    assert not fm.unit_ok([["A", "Ampere"]])
    assert not fm.unit_ok([1, None])


def test_first_interval():
    assert fm.first_interval([3, 3.5]) == (3.0, 3.5)
    assert fm.first_interval([[10, 11], [22, 23]]) == (10.0, 11.0)
    assert fm.first_interval([]) is None
