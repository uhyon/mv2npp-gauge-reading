import numpy as np
import pytest

from gauge_reader.keypoints.yolo import detections_from_arrays, pad_image, shift


def _gauge(cx=100.0, cy=100.0, r=50.0, frac=0.5):
    """Keypoints of a 270° dial: min at lower left, max at lower right, clockwise on screen."""
    import math
    a_min = math.radians(135)           # lower left, y down
    sweep = math.radians(270)
    pt = lambda a: (cx + r * math.cos(a), cy + r * math.sin(a))
    return [(cx, cy), pt(a_min + frac * sweep), pt(a_min), pt(a_min + sweep)]


def test_reading_and_sorting():
    k1, k2 = _gauge(frac=0.25), _gauge(frac=0.8)
    dets = detections_from_arrays(np.array([[0, 0, 10, 10], [50, 50, 150, 150]]),
                                  np.array([0.3, 0.9]), np.array([k1, k2]),
                                  np.array([[0.9] * 4, [0.8, 0.7, 0.9, 0.9]]))
    assert dets[0].box_conf == pytest.approx(0.9)
    assert dets[0].fraction() == pytest.approx(0.8)
    assert dets[0].reading(0, 10) == pytest.approx(8.0)
    assert dets[0].min_kpt_conf == pytest.approx(0.7)
    assert dets[1].reading(-1, 1) == pytest.approx(-0.5)


def test_pad_and_shift_back():
    img = np.zeros((40, 80, 3), np.uint8)
    padded, dx, dy = pad_image(img, 0.25)
    assert padded.shape == (60, 120, 3) and (dx, dy) == (20, 10)
    assert padded[0, 0, 0] == 114 and padded[dy, dx, 0] == 0
    k = [(p[0] + dx, p[1] + dy) for p in _gauge(frac=0.4)]
    det = detections_from_arrays(np.array([[dx, dy, dx + 10, dy + 10]]), np.array([1.0]),
                                 np.array([k]))[0]
    back = shift(det, dx, dy)
    assert back.box[:2] == (0.0, 0.0)
    assert back.keypoints.center == pytest.approx((100.0, 100.0))
    assert back.fraction() == pytest.approx(0.4)
    assert pad_image(img, 0)[0] is img
