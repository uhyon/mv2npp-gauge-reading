import csv
import json

import pytest

from gauge_reader.eval.datasets import load_measurebench, load_meterfl, range_group, write_manifest
from gauge_reader.eval.report import Prediction, load_predictions, report, score


def _meterfl(tmp_path, items):
    d = tmp_path / "Meter-FL" / "data" / "readings"
    d.mkdir(parents=True)
    for sid, client, unit, lo, hi, gt in items:
        (d / f"{sid}.json").write_text(json.dumps({
            "unit": unit, "scale_min": lo, "scale_max": hi, "reading_gt": gt, "tick_values": [],
            "seg_anomaly": False, "annot_issue": False, "notes": "", "file": f"{sid}.png",
            "client": client, "side": "test"}), encoding="utf-8")
    return tmp_path / "Meter-FL"


def _corrections(tmp_path, rows):
    p = tmp_path / "corr.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "field", "meterfl_value", "corrected_value", "checked_by", "date", "note"])
        w.writerows(rows)
    return p


def test_meterfl_corrections_and_filter(tmp_path):
    root = _meterfl(tmp_path, [
        ("a", "pressure_gauge_handheld", "megapascals", 0, 0.1, 0.035),
        ("b", "pressure_gauge_industrial", "magepascals", 0, 10, 6.1),
        ("c", "ammeter_lab", "ampere", 0, 5, 1.0)])
    corr = _corrections(tmp_path, [("a", "reading_gt", "0.035", "0.036", "", "", ""),
                                   ("b", "unit", "magepascals", "megapascals", "", "", "")])
    s = {x.id: x for x in load_meterfl(root, corr)}
    assert set(s) == {"a", "b"}                      # ammeter left out
    assert s["a"].reading == 0.036 and s["a"].corrected == ["reading_gt"]
    assert s["b"].unit == "megapascals" and s["b"].group == "0–10 MPa"
    assert s["a"].image.as_posix().endswith("pressure_gauge_handheld/test/a.png")
    assert len(load_meterfl(root, corr, pressure_only=False)) == 3


def test_meterfl_unknown_correction_id_fails(tmp_path):
    root = _meterfl(tmp_path, [("a", "pressure_gauge_lab", "bar", 0, 10, 1)])
    with pytest.raises(KeyError):
        load_meterfl(root, _corrections(tmp_path, [("zz", "reading_gt", "1", "2", "", "", "")]))


def test_percent_fs_with_unit_conversion_and_declines(tmp_path):
    root = _meterfl(tmp_path, [("a", "pressure_gauge_lab", "bar", 0, 10, 5.0),
                               ("b", "pressure_gauge_lab", "bar", 0, 10, 2.0),
                               ("c", "pressure_gauge_lab", "bar", 0, 10, 2.0)])
    samples = load_meterfl(root, None)
    preds = {"a": Prediction(5.1), "b": Prediction(0.2, "MPa"), "c": Prediction(None)}
    rows = {r.sample.id: r for r in score(samples, preds)}
    assert rows["a"].fs_error == pytest.approx(1.0)
    assert rows["b"].fs_error == pytest.approx(0.0)  # 0.2 MPa = 2 bar
    assert rows["c"].fs_error is None and not rows["c"].answered
    o = report(list(rows.values()))["overall"]
    assert o["coverage"] == pytest.approx(2 / 3)
    assert o["within_2fs"] == 1.0 and o["within_2fs_all"] == pytest.approx(2 / 3)


def _measurebench(tmp_path):
    p = tmp_path / "mb" / "labels.csv"
    p.parent.mkdir()
    alts_dual = [{"units": ["bar"], "interval": [2.0, 2.2], "point": False, "tol": "interval"},
                 {"units": ["psi"], "interval": [29, 32], "point": False, "tol": "interval"}]
    alts_hand = [{"units": ["°C"], "interval": [57.5, 62.5], "point": False, "tol": "minor_div"}]
    alts_baro = [{"units": ["hPa"], "interval": [1010, 1015], "point": False, "tol": "interval"}]
    fields = ["question_id", "image_type", "subtype", "unit", "interval_low", "interval_high",
              "n_intervals", "tol_method", "flags", "alternatives", "source", "scale_min", "scale_max"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"question_id": "pg_1", "image_type": "pressure_gauge", "subtype": "pressure_gauge",
                        "unit": "bar", "interval_low": 2.0, "interval_high": 2.2, "n_intervals": 2,
                        "tol_method": "interval", "flags": "", "alternatives": json.dumps(alts_dual),
                        "source": "", "scale_min": "", "scale_max": ""})
        w.writerow({"question_id": "th_1", "image_type": "thermometer", "subtype": "thermometer",
                        "unit": "°C", "interval_low": 57.5, "interval_high": 62.5, "n_intervals": 1,
                        "tol_method": "minor_div", "flags": "nonlinear_scale",
                        "alternatives": json.dumps(alts_hand), "source": "", "scale_min": 0, "scale_max": 120})
        w.writerow({"question_id": "pg_5", "image_type": "pressure_gauge", "subtype": "barometer",
                        "unit": "hPa", "interval_low": 1010, "interval_high": 1015, "n_intervals": 1,
                        "tol_method": "interval", "flags": "", "alternatives": json.dumps(alts_baro),
                        "source": "", "scale_min": "", "scale_max": ""})
    return p


def test_measurebench_hits_and_fs_subset(tmp_path):
    p = _measurebench(tmp_path)
    samples = load_measurebench(p)
    assert len(samples) == 3
    assert len(load_measurebench(p.parent, exclude_barometers=True)) == 2
    assert len(load_measurebench(p, exclude_flags=("nonlinear_scale",))) == 2
    s = {x.id: x for x in samples}
    assert s["th_1"].reading == 60 and s["th_1"].has_range and not s["pg_1"].has_range
    preds = {"pg_1": Prediction(30, "psi"), "th_1": Prediction(61.2, "°C"), "pg_5": Prediction(1020)}
    rows = {r.sample.id: r for r in score(samples, preds)}
    assert rows["pg_1"].hit and rows["pg_1"].fs_error is None
    assert rows["th_1"].hit and rows["th_1"].fs_error == pytest.approx(1.0)
    assert rows["pg_5"].hit is False
    rep = report(list(rows.values()))
    assert rep["overall"]["hit_rate"] == pytest.approx(2 / 3)
    assert rep["overall"]["fs_n"] == 1
    assert rep["without_flagged"]["n"] == 2
    assert set(rep["by_tol_method"]) == {"interval", "minor_div"}


def test_manifest_round_trip(tmp_path):
    samples = load_measurebench(_measurebench(tmp_path))
    m = tmp_path / "manifest.csv"
    write_manifest(samples, m)
    with open(m, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows[0]["value"], rows[0]["pred_unit"] = "2.1", "bar"
    with open(m, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    preds = load_predictions(m)
    assert preds["pg_1"] == Prediction(2.1, "bar") and preds["th_1"].value is None


def test_range_group_names():
    assert range_group(0, 0.1, "megapascals") == "0–0.1 MPa"
    assert range_group(-1, 0, "kilogram-force per square centimeter") == "-1–0 kgf/cm2"
    assert range_group(None, 5, "bar") == "no range"
