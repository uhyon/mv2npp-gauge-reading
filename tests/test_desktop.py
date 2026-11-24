import sys
from pathlib import Path

import pytest

from gauge_reader.reader import Reading
from gauge_reader.reading.geometry import AlarmBand

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))
from meter import meter_svg, status_of  # noqa: E402


def test_status_of():
    assert status_of(Reading("retake")) == "RETAKE"
    assert status_of(Reading("ok", alarm="HIGH", value=1.0)) == "HIGH"
    assert status_of(Reading("ok", alarm="", value=1.0)) == ""


def test_meter_svg_shows_value_band_and_status():
    r = Reading("ok", alarm="NORMAL", value=6.3, unit="bar", scale_min=0, scale_max=10,
                fraction=0.63)
    svg = meter_svg(r, AlarmBand(2, 8))
    assert svg.startswith("<svg") and "6.3" in svg and "NORMAL" in svg
    assert svg.count("<path") == 4          # track, low, normal and high bands
    assert "RETAKE" in meter_svg(Reading("retake"))


def test_app_tags_from_file_names():
    pytest.importorskip("PySide6.QtSvgWidgets")
    app = pytest.importorskip("app")
    assert app.tag_from_name(Path("PI-301__synth_blur.jpg")) == "PI-301"
    assert app.tag_from_name(Path("XX-1__a.jpg")) is None
    assert app.tag_from_name(Path("roboflow_dim.jpg")) is None
