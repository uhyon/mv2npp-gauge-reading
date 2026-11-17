import math

import numpy as np
import pytest

from gauge_reader.eval.roboflow import (
    DetRecord,
    average_precision,
    iou,
    original_name,
    parse_obaw9_boxes,
    parse_v4tii,
    score_gauge,
    summarize_keypoints,
    unstretch,
    v4tii_images,
)
from gauge_reader.keypoints.yolo import detections_from_arrays


def _row(cls, box, pts=((0, 0, 0), (0, 0, 0))):
    vals = [cls, *box] + [v for p in pts for v in p]
    return " ".join(str(v) for v in vals)


def _labels():
    # one gauge filling the centre of a 100x100 image; pivot (50,50), tip at 12 o'clock
    return "\n".join([
        _row(0, (0.5, 0.5, 0.8, 0.8)),
        _row(2, (0.3, 0.7, 0.05, 0.05), ((0.3, 0.7, 2), (0, 0, 0))),       # min, lower left
        _row(1, (0.7, 0.7, 0.05, 0.05), ((0.7, 0.7, 2), (0, 0, 0))),       # max, lower right
        _row(3, (0.5, 0.35, 0.05, 0.3), ((0.5, 0.2, 2), (0.5, 0.5, 2))),   # tip first: swapped
    ])


def test_original_name():
    assert original_name("abc_jpg.rf.123def") == "abc_jpg"
    assert original_name("plain") == "plain"


def test_parse_v4tii_assigns_parts_and_pivot():
    (g,) = parse_v4tii(_labels(), 100, 100)
    assert g.complete
    assert g.needle_swapped
    assert g.center == pytest.approx((50, 50))
    assert g.tip == pytest.approx((50, 20))
    assert g.scale_min == pytest.approx((30, 70))
    assert g.scale_max == pytest.approx((70, 70))


def test_parse_v4tii_ignores_parts_outside_any_gauge():
    text = _row(0, (0.2, 0.2, 0.2, 0.2)) + "\n" + _row(2, (0.9, 0.9, 0.05, 0.05), ((0.9, 0.9, 2), (0, 0, 0)))
    (g,) = parse_v4tii(text, 100, 100)
    assert g.scale_min is None and not g.complete


def test_parse_obaw9_meter_box():
    text = "1 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n3 0 0 1 0 1 1"
    assert parse_obaw9_boxes(text, 200, 100) == [pytest.approx((20, 20, 100, 60))]


def test_unstretch_makes_box_square():
    sx, sy = unstretch((0, 0, 50, 100))
    assert (50 * sx, 100 * sy) == (100, 100)
    assert unstretch((0, 0, 80, 40)) == (1.0, 2.0)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0


def test_score_gauge_perfect_prediction():
    (g,) = parse_v4tii(_labels(), 100, 100)
    kp = np.array([[g.center, g.tip, g.scale_min, g.scale_max]])
    (d,) = detections_from_arrays(np.array([g.box]), np.array([0.9]), kp)
    r = score_gauge(g, d)
    assert r["err_tip"] == pytest.approx(0)
    assert r["frac_err_img"] == pytest.approx(0)
    assert r["true_fraction"] == pytest.approx(0.5)
    s = summarize_keypoints([r, score_gauge(g, None)])
    assert s["detected"] == 1 and s["gauges"] == 2 and s["tip_pck05"] == 1


def test_score_gauge_unstretches_before_angles():
    # a round dial squeezed to half width: tip at 45 deg on the dial shows at a smaller angle
    W = 50
    r = 40
    def p(angle_deg):   # clockwise from 12 o'clock on the round dial, then squeeze x
        a = math.radians(angle_deg)
        return (50 + r * math.sin(a)) * W / 100, 50 - r * math.cos(a)
    lab = "\n".join([
        _row(0, (0.5, 0.5, 0.8, 0.8)),
        _row(2, (0, 0, 0.01, 0.01), ((p(-135)[0] / W, p(-135)[1] / 100, 2), (0, 0, 0))),
        _row(1, (0, 0, 0.01, 0.01), ((p(135)[0] / W, p(135)[1] / 100, 2), (0, 0, 0))),
        _row(3, (0, 0, 0.01, 0.01), ((0.5, 0.5, 2), (p(45)[0] / W, p(45)[1] / 100, 2))),
    ])
    (g,) = parse_v4tii(lab, W, 100)
    kp = np.array([[g.center, g.tip, g.scale_min, g.scale_max]])
    (d,) = detections_from_arrays(np.array([g.box]), np.array([0.9]), kp)
    assert score_gauge(g, d)["true_fraction"] == pytest.approx(180 / 270)


def test_average_precision():
    recs = [DetRecord([(0, 0, 10, 10)], [(0.9, (0, 0, 10, 10)), (0.8, (50, 50, 60, 60))]),
            DetRecord([(0, 0, 10, 10)], [(0.3, (1, 1, 10, 10))])]
    r = average_precision(recs, 0.5)
    assert r["n_gt"] == 2
    assert r["recall@0.25"] == 1
    assert r["precision@0.25"] == pytest.approx(2 / 3)
    # TP at 0.9, FP at 0.8, TP at 0.3: precision envelope 1 up to recall 0.5, then 2/3
    assert r["ap"] == pytest.approx((51 * 1 + 50 * 2 / 3) / 101)
    assert average_precision([DetRecord([(0, 0, 1, 1)])])["ap"] == 0


def test_v4tii_images_dedupes_train(tmp_path):
    for split, names in {"train": ["a_jpg.rf.1", "a_jpg.rf.2", "b_jpg.rf.3"], "valid": ["c_jpg.rf.4"]}.items():
        d = tmp_path / split / "images"
        d.mkdir(parents=True)
        for n in names:
            (d / f"{n}.jpg").write_bytes(b"")
    got = v4tii_images(tmp_path)
    assert [(s, p.stem) for s, p in got] == [("train", "a_jpg.rf.1"), ("train", "b_jpg.rf.3"),
                                             ("valid", "c_jpg.rf.4")]
