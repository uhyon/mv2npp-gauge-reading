import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gauge_reader.predict import read_rows


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
        return len(self.xyxy.a)


class FakeModel:
    """Returns one gauge centred in the (padded) input with the given sweep and fraction."""

    def __init__(self, specs):
        self.specs = list(specs)  # (sweep_deg, fraction) per call
        self.sizes = []

    def predict(self, img, **kw):
        sweep, frac = self.specs.pop(0)
        h, w = img.shape[:2]
        self.sizes.append((h, w))
        cx, cy, r = w / 2, h / 2, min(w, h) / 4
        a0 = math.radians(135)
        pt = lambda a: (cx + r * math.cos(a), cy + r * math.sin(a))
        kps = [(cx, cy), pt(a0 + math.radians(frac * sweep)), pt(a0), pt(a0 + math.radians(sweep))]
        return [SimpleNamespace(boxes=_Boxes([[cx - r, cy - r, cx + r, cy + r]], [0.9]),
                                keypoints=SimpleNamespace(xy=_T([kps]), conf=_T([[0.9] * 4])))]


def _rows(tmp_path):
    img = np.full((100, 120, 3), 200, np.uint8)
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), img)
    base = {"image": str(p), "scale_min": 0, "scale_max": 10, "unit": "bar"}
    return [{**base, "id": "ok"}, {**base, "id": "collapsed"}, {**base, "id": "past_max"},
            {**base, "id": "no_range", "scale_max": ""},
            {**base, "id": "missing", "image": str(tmp_path / "nope.png")}]


def test_read_rows_checks_and_padding(tmp_path):
    model = FakeModel([(270, 0.5), (40, 0.5), (270, 1.12)])
    calls = []
    recs, counts = read_rows(model, _rows(tmp_path), pad=0.15,
                             transform=lambda img, row: calls.append(row["id"]) or img)
    r = {x["id"]: x for x in recs}
    assert model.sizes[0] == (130, 156)                  # 15% margin per side
    assert r["ok"]["status"] == "ok" and float(r["ok"]["value"]) == pytest.approx(5.0, abs=1e-3)
    assert float(r["ok"]["center_x"]) == pytest.approx(60.0, abs=0.1)  # back in crop coordinates
    assert r["collapsed"]["status"] == "bad_geometry" and r["collapsed"]["value"] == ""
    assert float(r["collapsed"]["sweep"]) == pytest.approx(40.0, abs=0.1)
    assert float(r["past_max"]["value"]) == pytest.approx(11.0)          # clipped to 1.1
    assert float(r["past_max"]["fraction"]) == pytest.approx(1.12, abs=1e-3)
    assert r["no_range"]["status"] == "no_range" and r["missing"]["status"] == "unreadable"
    assert counts == {"ok": 2, "no_range": 1, "no_detection": 0, "unreadable": 1, "bad_geometry": 1,
                      "retake": 0}
    assert calls == ["ok", "collapsed", "past_max"]


def test_read_rows_without_clip(tmp_path):
    rows = _rows(tmp_path)[2:3]
    recs, _ = read_rows(FakeModel([(270, 1.12)]), rows, pad=0, clip=None)
    assert float(recs[0]["value"]) == pytest.approx(11.2, abs=1e-3)


def test_combine_tta_median_spread_and_retake():
    from gauge_reader.predict import combine_tta
    rows = [{"id": i, "scale_min": "0", "scale_max": "10", "unit": "bar"} for i in "abc"]
    up = [{"status": "ok", "fraction": "0.50", "value": "5", "ms": "10"},
          {"status": "ok", "fraction": "0.50", "value": "5"}, {"status": "no_detection", "value": ""}]
    plus = [{"status": "ok", "fraction": "0.52", "ms": "11"}, {"status": "ok", "fraction": "0.60"},
            {"status": "ok", "fraction": "0.3"}]
    minus = [{"status": "ok", "fraction": "0.49", "ms": "12"}, {"status": "ok", "fraction": "1.40"},
             {"status": "no_detection"}]
    a, b, c = combine_tta(rows, [up, plus, minus])
    assert a["status"] == "ok" and float(a["value"]) == 5.0 and float(a["spread"]) == 3.0
    assert a["ms"] == "33.0" and a["value_0"] == "5"
    assert b["frac_m"] == "1.1000"                       # clipped
    assert b["status"] == "retake" and b["value"] == "" and float(b["value_median"]) == 6.0
    assert c["status"] == "no_detection" and c["value"] == ""
    (b2,) = combine_tta(rows[1:2], [up[1:2], plus[1:2], minus[1:2]], retake_spread=None)
    assert b2["status"] == "ok" and float(b2["value"]) == 6.0


def test_read_rows_tta_turns_images(tmp_path):
    from gauge_reader.predict import read_rows_tta
    rows = _rows(tmp_path)[:1]
    model = FakeModel([(270, 0.5), (270, 0.51), (270, 0.49)])
    recs, counts = read_rows_tta(model, rows, pad=0)
    assert counts["ok"] == 1 and float(recs[0]["value"]) == pytest.approx(5.0, abs=1e-3)
    assert float(recs[0]["spread"]) == pytest.approx(2.0, abs=0.01)
