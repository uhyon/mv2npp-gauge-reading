"""Convert gauge keypoints into a reading.

Image coordinates: x to the right, y downward. Gauges sweep clockwise from the
minimum tick to the maximum tick, which is the convention for nearly all
pressure and temperature dials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]


def clockwise_angle(center: Point, start: Point, end: Point) -> float:
    """Clockwise angle in degrees [0, 360) from ray center->start to ray center->end."""
    a0 = math.atan2(start[1] - center[1], start[0] - center[0])
    a1 = math.atan2(end[1] - center[1], end[0] - center[0])
    # y points down, so increasing atan2 is clockwise on screen.
    return math.degrees(a1 - a0) % 360.0


@dataclass(frozen=True)
class GaugeKeypoints:
    center: Point
    tip: Point
    scale_min: Point
    scale_max: Point


def needle_fraction(kp: GaugeKeypoints) -> float:
    """Needle position as a fraction of the scale arc (0 at min, 1 at max).

    Values slightly outside [0, 1] are possible when the needle rests below the
    minimum stop or past the maximum; the dead zone is split evenly to decide.
    """
    sweep = clockwise_angle(kp.center, kp.scale_min, kp.scale_max)
    if sweep == 0:
        raise ValueError("scale_min and scale_max point in the same direction")
    needle = clockwise_angle(kp.center, kp.scale_min, kp.tip)
    if needle > sweep:  # needle is in the dead zone between max and min
        dead = 360.0 - sweep
        needle = needle if needle - sweep <= dead / 2 else needle - 360.0
    return needle / sweep


def read_value(kp: GaugeKeypoints, min_value: float, max_value: float) -> float:
    """Linear-scale reading from keypoints and the scale range."""
    return min_value + needle_fraction(kp) * (max_value - min_value)


def full_scale_error(pred: float, true: float, min_value: float, max_value: float) -> float:
    """Absolute error as a percentage of full scale (the project's main metric)."""
    return abs(pred - true) / abs(max_value - min_value) * 100.0


@dataclass(frozen=True)
class AlarmBand:
    low: float
    high: float


def check_alarm(value: float, band: AlarmBand, confidence: float, min_confidence: float = 0.5) -> str:
    """Return 'RETAKE', 'LOW', 'HIGH' or 'NORMAL'."""
    if confidence < min_confidence:
        return "RETAKE"
    if value < band.low:
        return "LOW"
    if value > band.high:
        return "HIGH"
    return "NORMAL"
