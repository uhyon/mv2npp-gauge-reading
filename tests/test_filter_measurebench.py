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


def test_single_value_answers_become_point_intervals():
    assert fm.intervals_of(3.5) == [(3.5, 3.5)]
    assert fm.intervals_of("120") == [(120.0, 120.0)]
    assert fm.pick({"value": 120, "units": ["mmHg"]}) == ("mmHg", (120.0, 120.0), 1, "ok")
    assert fm.pick({"intervals": [[3, 3.5]], "units": ["bar"]}) == ("bar", (3.0, 3.5), 1, "ok")


def test_default_types_exclude_sphygmomanometer():
    assert fm.KEEP_TYPES == {"pressure_gauge", "thermometer"}


def test_alternatives_dual_scale_keeps_both():
    kw = {"interval": [[3, 3.1], [43, 45]], "units": [["bar"], ["psi"]]}
    alts = fm.alternatives(kw)
    assert [a["units"] for a in alts] == [["bar"], ["psi"]]
    assert [a["interval"] for a in alts] == [[3.0, 3.1], [43.0, 45.0]]
    assert not any(a["point"] for a in alts)


def test_alternatives_single_scale_and_point():
    assert fm.alternatives({"interval": [3, 3.5], "units": ["bar", "Bar"]}) == [
        {"units": ["bar", "Bar"], "interval": [3.0, 3.5], "point": False, "tol": "interval"}]
    assert fm.alternatives({"value": 80, "units": ["°C"]})[0]["point"] is True


def test_zero_width_interval_is_point():
    alts = fm.alternatives({"interval": [6000, 6000], "units": ["kPa"]})
    assert alts[0]["point"] is True and alts[0]["tol"] == "point_tol"
    assert fm.tol_method(alts) == "point_tol"
    alts = fm.alternatives({"interval": [3, 3.5], "units": ["bar"]})
    assert alts[0]["point"] is False and fm.tol_method(alts) == "interval"


def test_point_widened_from_other_scale_psi_to_kpa():
    alts = fm.alternatives({"interval": [[800, 900], [6000, 6000]], "units": [["psi"], ["kPa"]]})
    kpa = alts[1]
    assert kpa["point"] is False and kpa["tol"] == "converted:psi"
    assert kpa["interval"] == pytest.approx([5515.81, 6205.28], abs=0.01)
    assert fm.tol_method(alts) == "converted+interval"


def test_point_widened_from_other_scale_mmhg_to_hpa():
    alts = fm.alternatives({"interval": [[985, 985], [738, 739]], "units": [["hPa"], ["mmHg"]]})
    assert alts[0]["interval"] == pytest.approx([983.92, 985.25], abs=0.01)
    assert alts[0]["tol"] == "converted:mmHg"


def test_point_without_convertible_partner_keeps_point_tol():
    alts = fm.alternatives({"interval": [[850, 850], [10, 20]], "units": [["bar"], ["°C"]]})
    assert alts[0]["tol"] == "point_tol" and alts[0]["point"] is True


@pytest.mark.parametrize("kw", [
    {"interval": [985, 985], "units": ["hPa"]},
    {"interval": [1017, 1017], "units": ["hPa"]},
    {"interval": [29.8, 30.0], "units": ["inHg"]},
    {"interval": [755, 760], "units": ["mmHg"]},
    {"interval": [[1010, 1012], [29.8, 29.9]], "units": [["mbar"], ["inHg"]]},
])
def test_barometer_detected(kw):
    assert fm.is_barometer(fm.alternatives(kw))


@pytest.mark.parametrize("kw", [
    {"interval": [6000, 6000], "units": ["kPa"]},
    {"interval": [850, 850], "units": ["bar"]},
    {"interval": [3, 3.5], "units": ["bar"]},
    {"interval": [120, 125], "units": ["mmHg"]},
])
def test_process_gauges_not_barometer(kw):
    assert not fm.is_barometer(fm.alternatives(kw))


def test_sanity_flags():
    alt = lambda u, lo, hi: fm.alternatives({"interval": [lo, hi], "units": [u]})
    assert fm.sanity_flags("pressure_gauge", alt("°C", 60, 62)) == ["temperature_unit_on_pressure_gauge"]
    assert fm.sanity_flags("pressure_gauge", alt("mmHg", 298, 298)) == ["possible_sphygmomanometer"]
    assert fm.sanity_flags("pressure_gauge", alt("mmHg", 755, 760)) == []
    assert fm.sanity_flags("pressure_gauge", alt("bar", 3, 3.5)) == []
    assert fm.sanity_flags("pressure_gauge", alt("mmHg", -300, -290)) == []  # vacuum gauge
    assert fm.sanity_flags("pressure_gauge", alt("mmHg", 320, 320)) == []


def test_manual_excludes():
    assert set(fm.EXCLUDE_IDS) == {"pressure_gauge_6", "pressure_gauge_39"}


def test_apply_manual_minor_div():
    alts = fm.alternatives({"interval": [2100, 2100], "units": ["psi"]})
    assert alts[0]["tol"] == "point_tol"
    new, scale = fm.apply_manual(alts, {"scale_min": "0", "scale_max": "5000", "minor_div": "100"})
    assert new[0]["interval"] == [2050.0, 2150.0] and new[0]["tol"] == "minor_div"
    assert scale == {"scale_min": 0.0, "scale_max": 5000.0}
    same, scale = fm.apply_manual(alts, {"scale_min": "", "scale_max": "", "minor_div": ""})
    assert same == alts and scale == {}


def _row(qid, image_type, kw, design="dial"):
    from PIL import Image

    return {"question_id": qid, "question": "What is the reading?", "image": Image.new("RGB", (8, 8)),
            "image_type": image_type, "design": design, "evaluator": "interval_matching",
            "evaluator_kwargs": __import__("json").dumps(kw), "meta_info": {"source": "fake"}}


def test_main_end_to_end_report(tmp_path, capsys):
    """Runs main() to the end with --report and --label-template on fake rows (no download)."""
    import csv

    rows = [
        _row("pressure_gauge_1", "pressure_gauge", {"interval": [[3, 3.1], [43, 45]], "units": [["bar"], ["psi"]]}),
        _row("pressure_gauge_2", "pressure_gauge", {"interval": [[800, 900], [6000, 6000]],
                                                    "units": [["psi"], ["kPa"]]}),
        _row("pressure_gauge_3", "pressure_gauge", {"interval": [6000, 6000], "units": ["kPa"]}),
        _row("pressure_gauge_4", "pressure_gauge", {"interval": [985, 990], "units": ["hPa"]}),
        _row("pressure_gauge_5", "pressure_gauge", {"interval": [20, 22], "units": ["°C"]}),
        _row("pressure_gauge_6", "pressure_gauge", {"interval": [20, 22], "units": ["°C"]}),
        _row("thermometer_1", "thermometer", {"interval": [36, 37], "units": ["°C"]}),
        _row("clock_1", "clock", {"interval": [3, 4], "units": ["h"]}),
        _row("pressure_gauge_7", "pressure_gauge", {"interval": [3, 4], "units": ["bar"]}, design="digital"),
    ]
    tmpl = tmp_path / "template.csv"
    fm.main(["--out", str(tmp_path), "--report", "--label-template", str(tmpl)], rows=rows)
    out = capsys.readouterr().out

    with open(tmp_path / "labels.csv", encoding="utf-8") as f:
        labels = {r["question_id"]: r for r in csv.DictReader(f)}
    assert set(labels) == {"pressure_gauge_1", "pressure_gauge_2", "pressure_gauge_3", "pressure_gauge_4",
                           "pressure_gauge_5", "thermometer_1"}
    assert labels["pressure_gauge_4"]["subtype"] == "barometer"
    assert labels["pressure_gauge_2"]["tol_method"].startswith("converted")
    assert "point_tol" in labels["pressure_gauge_3"]["tol_method"]
    assert labels["pressure_gauge_5"]["flags"] == "temperature_unit_on_pressure_gauge"
    assert (tmp_path / "images" / "thermometer_1.png").exists()

    with open(tmpl, encoding="utf-8-sig") as f:
        todo = list(csv.DictReader(f))
    assert [r["question_id"] for r in todo] == ["pressure_gauge_3"]

    assert "kept: 6" in out
    assert "temperature_unit_on_pressure_gauge: 1 ['pressure_gauge_5']" in out
    assert "possible_sphygmomanometer: 0 []" in out
    assert "barometer  pressure_gauge_4" in out
    assert "label template for 1 point_tol samples" in out


def test_main_exclude_barometer(tmp_path, capsys):
    rows = [_row("pressure_gauge_4", "pressure_gauge", {"interval": [985, 990], "units": ["hPa"]}),
            _row("pressure_gauge_1", "pressure_gauge", {"interval": [3, 4], "units": ["bar"]})]
    fm.main(["--out", str(tmp_path), "--exclude-type", "barometer", "--report"], rows=rows)
    assert "kept: 1" in capsys.readouterr().out
