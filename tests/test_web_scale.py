"""The browser demo's scale reading (demo/web/scale.js) agrees with the Python reader
(gauge_reader.reading.scale_ocr). Needs Node; skipped without it."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from gauge_reader.reading.geometry import GaugeKeypoints
from gauge_reader.reading.scale_ocr import alternatives, find_unit, parse_number, read_scale

NODE = shutil.which("node")
SCALE = Path(__file__).resolve().parents[1] / "demo" / "web" / "scale.js"
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

TEXTS = ["0.1MPa", "08", "1O", "0.O6", "-1", "16", "185", "004", "1,6", "bar", "EN837-1", "CL1.6",
         "kg/cm", "Kgf/cm²", "°C", "C", "MPa", "Pa", "psi", "x10", "+", "=1", "l16", "2.5bar"]


def _js(body: str):
    script = f"import * as s from {json.dumps(SCALE.as_uri())};\n{body}"
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True,
                         text=True, check=True)
    return json.loads(out.stdout)


def test_text_parsing_matches_python():
    js = _js(f"""const t = {json.dumps(TEXTS)};
console.log(JSON.stringify({{ num: t.map((x) => s.parseNumber(x)),
  alt: t.map((x) => {{ const [v] = s.parseNumber(x); return v === null ? null : s.alternatives(x, v); }}),
  unit: t.map((x) => s.findUnit([x])), units: s.findUnit(t) }}));""")
    for text, (v, rest), alt, unit in zip(TEXTS, js["num"], js["alt"], js["unit"]):
        pv, prest = parse_number(text)
        assert (v, rest) == (pv, prest), text
        if pv is not None:
            assert alt == pytest.approx(alternatives(text, pv)), text
        assert unit == find_unit([text]), text
    assert js["units"] == find_unit(TEXTS)


def _dial_boxes(lo, hi, n, sweep=270.0, start=225.0, r=100.0, extra=()):
    """Numbers lo..hi printed around a dial (image y down, clockwise from the min mark)."""
    c = (300.0, 300.0)
    pt = lambda ang, rad: (c[0] + rad * math.cos(math.radians(ang)),
                           c[1] - rad * math.sin(math.radians(ang)))
    boxes = []
    for i in range(n):
        f = i / (n - 1)
        x, y = pt(start - f * sweep, 0.75 * r)
        v = lo + f * (hi - lo)
        boxes.append([f"{v:g}", [(x - 8, y - 6), (x + 8, y - 6), (x + 8, y + 6), (x - 8, y + 6)], 0.9])
    boxes += [list(b) for b in extra]
    kp = GaugeKeypoints(c, pt(90, r), pt(start, r), pt(start - sweep, r))
    return kp, boxes


@pytest.mark.parametrize("lo,hi,n,extra", [
    (0, 10, 6, ()), (0, 1.6, 9, ()), (-1, 6, 8, ()),
    (0, 16, 5, [("EN837-1", [(290, 330), (310, 330), (310, 340), (290, 340)], 0.9),
                ("bar", [(292, 350), (308, 350), (308, 360), (292, 360)], 0.9)]),
])
def test_read_scale_matches_python(lo, hi, n, extra):
    kp, boxes = _dial_boxes(lo, hi, n, extra=extra)
    py = read_scale([(t, p, c) for t, p, c in boxes], kp, None)
    k = {"center": kp.center, "tip": kp.tip, "scale_min": kp.scale_min, "scale_max": kp.scale_max}
    js = _js(f"""const b = {json.dumps(boxes)}.map(([text, poly, conf]) => ({{ text, poly, conf }}));
const r = s.readScale(b, {json.dumps(k)}, null);
console.log(JSON.stringify([r.scaleMin, r.scaleMax, r.unit, r.inliers.length]));""")
    assert js[0] == pytest.approx(py.scale_min) and js[1] == pytest.approx(py.scale_max)
    assert js[2] == py.unit and js[3] == len(py.inliers)
    assert py.scale_min == pytest.approx(lo) and py.scale_max == pytest.approx(hi)
