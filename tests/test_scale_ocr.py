import math

import pytest

from gauge_reader.reading.geometry import GaugeKeypoints
from gauge_reader.reading.scale_ocr import (
    alternatives,
    find_unit,
    fit_line,
    labels_on_dial,
    parse_number,
    read_scale,
)

# upright dial: centre (100,100), radius 80, min at 7:30 (225° clockwise from 12), sweep 270
C, R = (100.0, 100.0), 80.0


def at(frac, r=0.75):
    """Image point at ``frac`` of the scale, ``r`` scale radii from the centre."""
    ang = math.radians(225 + 270 * frac)            # clockwise from 12 o'clock
    return C[0] + r * R * math.sin(ang), C[1] - r * R * math.cos(ang)


KP = GaugeKeypoints(C, at(0.5, 0.9), at(0.0, 1.0), at(1.0, 1.0))


def box(text, frac, r=0.75, conf=0.9):
    x, y = at(frac, r)
    return text, [(x - 5, y - 4), (x + 5, y - 4), (x + 5, y + 4), (x - 5, y + 4)], conf


def test_parse_number():
    assert parse_number("0.1MPa") == (0.1, "MPa")
    assert parse_number("1,6") == (1.6, "")
    assert parse_number("08") == (0.8, "")
    assert parse_number("0.O6") == (0.06, "")
    assert parse_number("MPa")[0] is None
    assert parse_number("-0.1") == (-0.1, "")


def test_alternatives_restore_dropped_decimal_point():
    assert alternatives("16", 16.0) == [16.0, 1.6]
    assert alternatives("100", 100.0) == [100.0, 1.0, 10.0]
    assert alternatives("1.6", 1.6) == [1.6]
    assert alternatives("5", 5.0) == [5.0]


def test_find_unit():
    assert find_unit(["压力表", "MPa"]) == "MPa"
    assert find_unit(["Kg/cm", "MOTOR"]) == "kgf/cm2"
    assert find_unit(["Pa", "0.1"]) == ""          # prefix missed: ambiguous
    assert find_unit(["C", "100"]) == "°C"         # degree sign lost
    assert find_unit(["C", "MPa"]) == "MPa"        # a printed unit wins over a lone letter
    assert find_unit(["°℃"]) == "℃"
    assert find_unit(["°C"]) == "°C"
    assert find_unit(["0-60 psi"]) == "psi"


def test_read_scale_with_dropped_decimals_and_noise():
    boxes = [box("04", 0.25), box("08", 0.5), box("12", 0.75), box("16", 1.0),
             box("12810126", 0.5, r=0.3), box("MPa", 0.5, r=0.4), box("NO 2", 0.9, r=1.9)]
    s = read_scale(boxes, KP)
    assert s.ok and s.scale_min == 0.0 and s.scale_max == pytest.approx(1.6)
    assert s.unit == "MPa" and len(s.inliers) == 4


def test_read_scale_ignores_off_dial_numbers_and_outliers():
    boxes = [box("10", 0.2), box("20", 0.4), box("30", 0.6), box("40", 0.8), box("50", 1.0),
             box("2024", 0.5, r=2.0), box("7", 0.3)]
    s = read_scale(boxes, KP)
    assert (s.scale_min, s.scale_max) == (0.0, 50.0)
    assert s.rms < 0.01


def test_read_scale_needs_three_labels():
    s = read_scale([box("10", 0.5), box("16", 0.8)], KP)
    assert not s.ok


def test_vacuum_gauge_negative_to_zero():
    boxes = [box("-0.1", 0.0), box("-0.05", 0.5), box("0", 1.0), box("-0.075", 0.25)]
    s = read_scale(boxes, KP)
    assert (s.scale_min, s.scale_max) == (-0.1, 0.0)


def test_fit_line_prefers_values_as_read():
    labels = labels_on_dial([box("2", 0.2), box("4", 0.4), box("6", 0.6), box("8", 0.8)], KP, None)
    a, b, _inl = fit_line(labels)
    assert a == pytest.approx(0, abs=1e-6) and b == pytest.approx(10)


def test_decimal_line_needs_clearly_more_labels():
    # a thermometer 0..100: the /10 line explains the same labels, the printed values win
    boxes = [box(str(v), v / 100) for v in range(0, 101, 10)] + [box("-10", -0.1)]
    s = read_scale(boxes, KP)
    assert (s.scale_min, s.scale_max) == (0.0, 100.0)
