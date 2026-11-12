import math

import numpy as np
import pytest

from gauge_reader.reading.geometry import GaugeKeypoints, full_scale_error, read_value
from gauge_reader.synth.dial import DialSpec, format_label, nice_major_steps, render_dial
from gauge_reader.synth.scene import apply_h, make_sample

CFG = {"image_size": 320, "seed": 0,
       "dial": {"ranges": [[0, 10], [0, 16], [-1, 1], [0, 150]], "units": ["bar", "psi"]}}


def kp(d):
    return GaugeKeypoints(center=d["center"], tip=d["tip"], scale_min=d["scale_min"],
                          scale_max=d["scale_max"])


def test_nice_steps_and_labels():
    assert nice_major_steps(0, 16) == [2.0]
    assert 0.25 in nice_major_steps(-1, 1)
    assert format_label(-0.0, 0.5) == "0.0" and format_label(2.5, 2.5) == "2.5"


@pytest.mark.parametrize("sweep", [180.0, 270.0])
@pytest.mark.parametrize("value", [0.0, 3.7, 10.0])
def test_flat_dial_keypoints_read_back(sweep, value):
    spec = DialSpec(0, 10, "bar", sweep, 1, 5)
    r = render_dial(spec, value, size=256, rng=np.random.default_rng(1))
    k = GaugeKeypoints(r.center, r.tip, r.scale_min, r.scale_max)
    assert read_value(k, 0, 10) == pytest.approx(value, abs=1e-6)
    # The gap is centred at the bottom: min and max are mirror images.
    assert r.scale_min[1] == pytest.approx(r.scale_max[1])
    assert r.scale_min[0] + r.scale_max[0] == pytest.approx(2 * r.center[0])


@pytest.mark.parametrize("seed", range(12))
def test_sample_labels_consistent(seed):
    s = make_sample(np.random.default_rng(seed), CFG)
    sp = s.spec
    assert s.image.shape == (320, 320, 3)
    assert sp.min_value <= s.value <= sp.max_value
    # Image keypoints map back through the homography to the flat-dial keypoints,
    # where the reading is exact.
    Hinv = np.linalg.inv(s.homography)
    back = dict(zip(s.keypoints, apply_h(Hinv, list(s.keypoints.values()))))
    for name, p in s.dial_keypoints.items():
        assert back[name] == pytest.approx(p, abs=1e-3)
    pred = read_value(kp(back), sp.min_value, sp.max_value)
    assert full_scale_error(pred, s.value, sp.min_value, sp.max_value) < 1e-4
    # The whole gauge case lies inside the frame.
    x0, y0, x1, y1 = s.bbox
    assert 0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 320
    for x, y in s.keypoints.values():
        assert x0 <= x <= x1 and y0 <= y <= y1


def test_same_seed_same_image():
    a = make_sample(np.random.default_rng(5), CFG)
    b = make_sample(np.random.default_rng(5), CFG)
    assert np.array_equal(a.image, b.image) and a.value == b.value


def test_needle_tip_is_drawn_on_needle():
    spec = DialSpec(0, 10, "bar", 270, 1, 5, face_color=(255, 255, 255), needle_color=(255, 0, 0))
    r = render_dial(spec, 6.0, size=400, rng=np.random.default_rng(0))
    img = np.array(r.image)
    # Sample just inside the tip along the needle direction: must be red.
    dx, dy = r.center[0] - r.tip[0], r.center[1] - r.tip[1]
    n = math.hypot(dx, dy)
    x, y = r.tip[0] + dx / n * 8, r.tip[1] + dy / n * 8
    px = img[round(y), round(x)]
    assert px[0] > 150 and px[1] < 100


def test_case_wall_extends_bbox_at_oblique_view():
    cfg = dict(CFG, camera={"yaw_deg": [45, 45], "pitch_deg": [0, 0], "roll_deg": [0, 0],
                            "focal": [1.5, 1.5], "scale": [0.5, 0.5]})
    flat = make_sample(np.random.default_rng(3), dict(cfg, case={"depth": [0, 0], "recess": [0, 0]}))
    deep = make_sample(np.random.default_rng(3), dict(cfg, case={"depth": [0.4, 0.4],
                                                                "recess": [0.05, 0.05]}))
    w = lambda s: s.bbox[2] - s.bbox[0]
    assert w(deep) > w(flat) * 1.1          # the side wall shows
    assert deep.camera["case_depth"] > 0 and flat.camera["face_recess"] == 0
    # Recessed face: the homography is not the front-plane one, but labels stay exact.
    Hinv = np.linalg.inv(deep.homography)
    back = dict(zip(deep.keypoints, apply_h(Hinv, list(deep.keypoints.values()))))
    sp = deep.spec
    assert read_value(kp(back), sp.min_value, sp.max_value) == pytest.approx(deep.value, abs=1e-6)
