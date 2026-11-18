import math

import pytest
from test_predict import FakeModel
from test_reader import _img

from gauge_reader.reader import GaugeReader
from gauge_reader.registry import compare_scale, load_registry


def test_load_registry(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("# comment\ntag,scale_min,scale_max,unit,alarm_low,alarm_high\n"
                 "PI-1,0,1.6,MPa,0.4,1.2\nTI-1,0,120,°C,,90\n", encoding="utf-8")
    reg = load_registry(p)
    assert reg["PI-1"].scale == (0.0, 1.6, "MPa")
    assert reg["PI-1"].band.low == 0.4 and reg["PI-1"].band.high == 1.2
    assert reg["TI-1"].band.low == -math.inf and reg["TI-1"].band.high == 90


def test_load_registry_rejects_bad_rows(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("tag,scale_min,scale_max,unit\nA,0,,bar\n")
    with pytest.raises(ValueError):
        load_registry(p)
    p.write_text("tag,scale_min,scale_max,unit\nA,0,1,bar\nA,0,2,bar\n")
    with pytest.raises(ValueError):
        load_registry(p)


def test_example_registry_loads():
    from pathlib import Path
    reg = load_registry(Path(__file__).resolve().parents[1] / "configs" / "gauges.csv")
    assert "PI-101" in reg


def test_compare_scale():
    assert compare_scale((0, 1.6, "MPa"), 0, 1.6, "MPa") == "match"
    assert compare_scale((0, 1.6, "MPa"), 0.01, 1.6, "") == "match"
    assert compare_scale((0, 1.6, "MPa"), 0, 2.5, "MPa") == "mismatch"
    assert compare_scale((0, 1.6, "MPa"), 0, 1.6, "bar") == "mismatch"
    assert compare_scale((0, 1.6, "MPa"), None, None) == "no_ocr"
    assert compare_scale((0, 1.6, "megapascals"), 0, 1.6, "MPA") == "match"
    assert compare_scale((0, 60, "pounds per square inch"), 0, 60, "PSI") == "match"
    assert compare_scale((0, 1.6, "kilopascals"), 0, 1.6, "MPa") == "mismatch"


def test_given_scale_skips_ocr_unless_checked():
    calls = []

    def ocr(img):
        calls.append(1)
        return []

    model = FakeModel([(270, 0.5)] * 4)
    r = GaugeReader(model=model, ocr=ocr).read(_img(), scale=(0, 10, "bar"))
    assert r.status == "ok" and not calls and r.scale_check == "" and "ocr" not in r.ms

    model = FakeModel([(270, 0.5)] * 4)
    r = GaugeReader(model=model, ocr=ocr, check_scale=True).read(_img(), scale=(0, 10, "bar"))
    assert r.status == "ok" and calls and r.scale_check == "no_ocr"
    assert r.value == pytest.approx(5.0, abs=1e-3) and r.scale_source == "given"
