import math

import numpy as np
import pytest

from gauge_reader.keypoints.yolo import detections_from_arrays, shift
from gauge_reader.reading.geometry import GaugeKeypoints, needle_fraction
from gauge_reader.reading.perspective import ARC_FRACTIONS, arc_points, fit_dial


def _rot(yaw, pitch, roll):
    y, p, r = map(math.radians, (yaw, pitch, roll))
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return rz @ ry @ rx


def dial(frac, *, sweep=270.0, a0=135.0, yaw=0.0, pitch=0.0, roll=0.0, noise=0.0, seed=0):
    """Keypoints of a unit dial seen by a pinhole camera; arc points at ARC_FRACTIONS."""
    rot, rng = _rot(yaw, pitch, roll), np.random.default_rng(seed)

    def img(a_deg, r=1.0):
        a = math.radians(a_deg)
        p = rot @ np.array([r * math.cos(a), r * math.sin(a), 0.0]) + [0, 0, 3.0]
        q = 600 * p[:2] / p[2] + 320 + rng.normal(0, noise, 2)
        return float(q[0]), float(q[1])

    center = img(0.0, 0.0)
    kp = GaugeKeypoints(center, img(a0 + frac * sweep, 0.8), img(a0), img(a0 + sweep))
    return kp, [img(a0 + f * sweep) for f in ARC_FRACTIONS]


@pytest.mark.parametrize("frac", [0.0, 0.31, 0.5, 0.87, 1.0])
def test_fit_is_exact_on_a_tilted_dial(frac):
    kp, arc = dial(frac, yaw=45, pitch=30, roll=10)
    fit = fit_dial(kp, arc)
    assert fit.fraction == pytest.approx(frac, abs=2e-4)
    assert fit.sweep == pytest.approx(270.0, abs=0.05)
    assert fit.rms < 1e-4


def test_fit_beats_image_plane_angles_on_tilted_dials():
    errs_img, errs_fit = [], []
    for k, frac in enumerate(np.linspace(0.05, 0.95, 10)):
        kp, arc = dial(frac, yaw=50, pitch=-25, roll=30 * k)
        errs_img.append(abs(needle_fraction(kp) - frac))
        errs_fit.append(abs(fit_dial(kp, arc).fraction - frac))
    assert max(errs_img) > 0.02          # the image plane is off by several % FS
    assert max(errs_fit) < 1e-3


@pytest.mark.parametrize("sweep", [90.0, 180.0, 300.0, 340.0])
def test_fit_finds_the_sweep(sweep):
    kp, arc = dial(0.4, sweep=sweep, a0=200, yaw=30, pitch=20)
    fit = fit_dial(kp, arc)
    assert fit.sweep == pytest.approx(sweep, abs=0.1)
    assert fit.fraction == pytest.approx(0.4, abs=1e-3)


def test_turned_dial_reads_the_same():
    reads = [fit_dial(*dial(0.62, yaw=35, pitch=15, roll=roll)).fraction for roll in (0, 90, 180, 270)]
    assert reads == pytest.approx([0.62] * 4, abs=1e-3)


def test_needle_in_the_dead_zone():
    below = fit_dial(*dial(-0.04, yaw=30, pitch=20))   # resting just under the min stop
    past = fit_dial(*dial(1.05, yaw=30, pitch=20))
    assert below.fraction == pytest.approx(-0.04, abs=1e-3)
    assert past.fraction == pytest.approx(1.05, abs=1e-3)


def test_noise_stays_bounded():
    errs = [abs(fit_dial(*dial(f, yaw=40, pitch=25, noise=1.0, seed=s)).fraction - f)
            for s, f in enumerate(np.linspace(0.1, 0.9, 9))]
    # 1 px of noise on a dial of about 200 px radius
    assert np.mean(errs) < 0.01


def test_points_off_any_round_dial_have_a_large_residual():
    kp, arc = dial(0.5, yaw=30, pitch=20)
    good = fit_dial(kp, arc).rms
    bad = fit_dial(kp, [arc[0], (arc[1][0] + 60, arc[1][1]), arc[2]]).rms
    assert good < 1e-4
    assert bad > 0.01


def test_arc_points_follow_the_homography():
    h = np.array([[2.0, 0, 10], [0, 2.0, 20], [0, 0, 1]])
    pts = arc_points(h, (256, 256), (256 - 100, 256), 180.0)   # min at 9 o'clock, clockwise
    # 25% -> 45 deg further clockwise (up-left on screen), 50% -> 12 o'clock, 75% -> up-right
    s = 100 / math.sqrt(2)
    want = [(256 - s, 256 - s), (256, 156), (256 + s, 256 - s)]
    np.testing.assert_allclose(pts, [(2 * x + 10, 2 * y + 20) for x, y in want], atol=1e-9)


def test_detections_with_arc_points():
    kp, arc = dial(0.3, yaw=50, pitch=30)
    kpts = np.array([[kp.center, kp.tip, kp.scale_min, kp.scale_max, *arc]])
    (d,) = detections_from_arrays(np.array([[0, 0, 640, 640]]), np.array([0.9]), kpts)
    assert len(d.arc) == 3
    assert d.fraction() == pytest.approx(0.3, abs=1e-3)
    assert abs(d.image_fraction() - 0.3) > 1e-3
    moved = shift(d, 5, 7)
    assert moved.arc[0] == pytest.approx((arc[0][0] - 5, arc[0][1] - 7))
    assert moved.fraction() == pytest.approx(0.3, abs=1e-3)


def test_four_keypoint_detections_read_in_the_image():
    kp, _ = dial(0.3)
    kpts = np.array([[kp.center, kp.tip, kp.scale_min, kp.scale_max]])
    (d,) = detections_from_arrays(np.array([[0, 0, 640, 640]]), np.array([0.9]), kpts)
    assert d.arc == () and d.dial_fit() is None
    assert d.fraction() == pytest.approx(0.3, abs=1e-6)


def test_wrong_keypoint_count():
    with pytest.raises(ValueError):
        detections_from_arrays(np.zeros((1, 4)), np.ones(1), np.zeros((1, 5, 2)))


class _T:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = _T(xyxy), _T(conf)

    def __len__(self):
        return 1


def test_read_rows_reads_arc_models_on_the_face(tmp_path):
    from types import SimpleNamespace

    import cv2

    from gauge_reader.predict import read_rows

    kp, arc = dial(0.3, yaw=50, pitch=30)
    pts = [[kp.center, kp.tip, kp.scale_min, kp.scale_max, *arc]]
    res = SimpleNamespace(boxes=_Boxes([[0, 0, 640, 640]], [0.9]),
                          keypoints=SimpleNamespace(xy=_T(pts), conf=_T([[0.9] * 7])))
    model = SimpleNamespace(predict=lambda img, **kw: [res])
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), np.full((640, 640, 3), 128, np.uint8))
    rows = [{"id": "a", "image": str(p), "scale_min": 0, "scale_max": 10, "unit": "bar"}]
    (rec,), counts = read_rows(model, rows, pad=0)
    assert counts["ok"] == 1
    assert float(rec["value"]) == pytest.approx(3.0, abs=0.01)
    assert abs(float(rec["fraction_img"]) - 0.3) > 0.005
    assert float(rec["fit_rms"]) < 1e-3
    assert rec["arc_50_x"]
