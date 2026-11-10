import math

import pytest

from gauge_reader.reading.geometry import (
    AlarmBand,
    GaugeKeypoints,
    check_alarm,
    clockwise_angle,
    full_scale_error,
    read_value,
)


def polar(deg_clockwise_from_up: float, r: float = 100.0, c=(200.0, 200.0)):
    """Point on a dial, angle measured clockwise from 12 o'clock (image coords)."""
    t = math.radians(deg_clockwise_from_up)
    return (c[0] + r * math.sin(t), c[1] - r * math.cos(t))


def make_kp(needle_deg: float, sweep: float = 270.0) -> GaugeKeypoints:
    # Classic 270 degree dial: min at 7:30 (-135), max at 4:30 (+135).
    return GaugeKeypoints(
        center=(200.0, 200.0),
        tip=polar(needle_deg),
        scale_min=polar(-sweep / 2),
        scale_max=polar(sweep / 2),
    )


def test_clockwise_angle_quarter_turn():
    assert clockwise_angle((0, 0), (0, -1), (1, 0)) == pytest.approx(90.0)


@pytest.mark.parametrize(
    "needle_deg, expected",
    [(-135, 0.0), (0, 5.0), (135, 10.0), (-67.5, 2.5), (67.5, 7.5)],
)
def test_read_value_270_dial(needle_deg, expected):
    assert read_value(make_kp(needle_deg), 0.0, 10.0) == pytest.approx(expected, abs=1e-6)


def test_needle_below_min_stop_reads_negative():
    # 5 degrees counter-clockwise of the min tick.
    assert read_value(make_kp(-140), 0.0, 10.0) < 0


def test_needle_past_max_reads_over_range():
    assert read_value(make_kp(140), 0.0, 10.0) > 10.0


def test_full_scale_error():
    assert full_scale_error(5.2, 5.0, 0.0, 10.0) == pytest.approx(2.0)


def test_alarm_states():
    band = AlarmBand(2.0, 8.0)
    assert check_alarm(5.0, band, 0.9) == "NORMAL"
    assert check_alarm(9.0, band, 0.9) == "HIGH"
    assert check_alarm(1.0, band, 0.9) == "LOW"
    assert check_alarm(5.0, band, 0.2) == "RETAKE"
