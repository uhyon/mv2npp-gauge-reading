"""Read the scale range and unit from the numbers printed on the dial (stage 4, FR-07).

An OCR model returns text boxes. Each box that holds a number is placed on the scale by its
angle around the dial centre, as a fraction of the min-to-max arc: on the dial face when the
scale-arc keypoints give a ``DialFit`` (so a tilted view does not distort the angles), in the
image otherwise. Printed numbers grow linearly along the scale, so (fraction, value) pairs lie
on a line value = a + b * fraction; a RANSAC line through them ignores serial numbers, model
names and misread labels (for example "12" for "1.2"). The scale range is the line at fraction
0 and 1, snapped to a printed label or to the label grid. The unit is the first text that
names a pressure or temperature unit.

Works on plain lists of (text, box, confidence) so it can be tested without an OCR model;
``RapidOcr`` wraps rapidocr-onnxruntime (PP-OCRv3, CPU, models shipped with the package).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from gauge_reader.reading.geometry import GaugeKeypoints, Point, clockwise_angle
from gauge_reader.reading.perspective import DialFit, _project
from gauge_reader.units import UNIT_KIND, norm_unit

TextBox = tuple[str, Sequence[Point], float]   # text, polygon (4 points), confidence

NUM_RE = re.compile(r"^([-+]?\d+(?:[.,]\d+)?)\s*([^\d\s].*)?$")
RADIUS_RANGE = (0.3, 1.4)     # label distance from the centre, in scale radii
FRACTION_RANGE = (-0.15, 1.15)
INLIER_TOL = 0.04             # fraction of the scale
MIN_INLIERS = 3               # distinct labels on the line needed to trust a range
SNAP_TOL = 0.03               # fraction of the span
# superscript 2 lost by the OCR
TRUNCATED_UNITS = {"kg/cm": "kgf/cm2", "kgf/cm": "kgf/cm2", "kg/cm.": "kgf/cm2"}


@dataclass
class Label:
    text: str
    value: float
    fraction: float
    radius: float
    conf: float
    box: int = 0          # index of the text box (alternative readings share it)


@dataclass
class ScaleRead:
    scale_min: float | None
    scale_max: float | None
    unit: str = ""
    labels: list[Label] = field(default_factory=list)   # all numeric labels on the dial
    inliers: list[Label] = field(default_factory=list)  # labels on the fitted line
    rms: float = math.nan                               # fit residual, fraction of the scale

    @property
    def ok(self) -> bool:
        return self.scale_min is not None and self.scale_max is not None


def parse_number(text: str) -> tuple[float | None, str]:
    """Leading number and the rest (a possible unit): "0.1MPa" -> (0.1, "MPa")."""
    t = text.strip().replace(" ", "")
    # O/o read for a zero next to digits ("0.O6", "1O")
    if re.search(r"\d", t):
        t = re.sub(r"(?<=[\d.])[Oo]|[Oo](?=[\d.])", "0", t)
    m = NUM_RE.match(t)
    if not m:
        return None, ""
    num = m.group(1).replace(",", ".")
    # a leading zero before more digits means a dropped decimal point: "08" is 0.8
    if re.fullmatch(r"0\d+", num):
        num = "0." + num[1:]
    try:
        return float(num), (m.group(2) or "")
    except ValueError:
        return None, ""


def find_unit(texts: Sequence[str]) -> str:
    """First token that names a pressure or temperature unit. A lone "C" or "F" (degree sign
    lost) counts only when no other unit is printed."""
    u = _find_unit(texts)
    if u:
        return u
    for text in texts:
        t = text.strip().replace("°", "").replace("º", "")
        if t in ("C", "℃"):
            return "°C"
        if t in ("F", "℉"):
            return "°F"
    return ""


def _find_unit(texts: Sequence[str]) -> str:
    for text in texts:
        for tok in [text, *re.split(r"[\s()\[\]]+", text)]:
            _, rest = parse_number(tok)
            for cand in (tok, rest):
                if not cand:
                    continue
                n = norm_unit(cand.replace("°℃", "℃").replace("°℉", "℉"))
                n = TRUNCATED_UNITS.get(n, n)
                if n == "pa":       # usually "MPa" or "kPa" with the prefix missed
                    continue
                if n in UNIT_KIND and (len(n) > 1 or "°" in cand or "℃" in cand or "℉" in cand):
                    return TRUNCATED_UNITS.get(norm_unit(cand), cand.strip().replace("°℃", "℃"))
    return ""


def _center(poly: Sequence[Point]) -> Point:
    p = np.asarray(poly, dtype=float)
    return float(p[:, 0].mean()), float(p[:, 1].mean())


def place(point: Point, kp: GaugeKeypoints, fit: DialFit | None) -> tuple[float, float]:
    """(fraction of the scale, distance from the centre in scale radii) of an image point."""
    if fit is not None:
        q = _project(np.linalg.inv(fit.homography), np.array([point], dtype=float))[0]
        ang = math.degrees(math.atan2(q[1], q[0])) % 360.0
        r = float(np.hypot(*q))
        sweep = fit.sweep
    else:
        sweep = clockwise_angle(kp.center, kp.scale_min, kp.scale_max)
        ang = clockwise_angle(kp.center, kp.scale_min, point)
        rad = (math.dist(kp.center, kp.scale_min) + math.dist(kp.center, kp.scale_max)) / 2
        r = math.dist(kp.center, point) / rad if rad > 0 else math.inf
    if sweep <= 0:
        return math.nan, r
    if ang > sweep and ang - sweep > (360.0 - sweep) / 2:
        ang -= 360.0
    return ang / sweep, r


def alternatives(text: str, value: float) -> list[float]:
    """The value and, for a whole number of 2+ digits, the same digits with a decimal point
    the OCR may have dropped ("12" for "1.2", "16" for "1.6", "004" for "0.04")."""
    out = [value]
    digits = re.sub(r"\D", "", text.split(".")[0] if "." in text else text)
    if "." not in text and "," not in text and len(digits) >= 2 and not digits.startswith("0"):
        sign = -1.0 if value < 0 else 1.0
        for k in range(1, len(digits)):
            out.append(sign * float(digits[:k] + "." + digits[k:]))
    return list(dict.fromkeys(out))


def labels_on_dial(boxes: Sequence[TextBox], kp: GaugeKeypoints, fit: DialFit | None) -> list[Label]:
    """Numeric labels near the scale; a box may give several alternative values."""
    out = []
    for i, (text, poly, conf) in enumerate(boxes):
        v, _ = parse_number(text)
        if v is None:
            continue
        f, r = place(_center(poly), kp, fit)
        if (RADIUS_RANGE[0] <= r <= RADIUS_RANGE[1]
                and FRACTION_RANGE[0] <= f <= FRACTION_RANGE[1]):
            for alt in alternatives(text, v):
                out.append(Label(text, alt, f, r, float(conf), i))
    return out


def fit_line(labels: Sequence[Label], tol: float = INLIER_TOL,
             min_inliers: int = MIN_INLIERS) -> tuple[float, float, list[Label]] | None:
    """RANSAC over label pairs: value = a + b * fraction with b > 0. Returns a, b, inliers."""
    best = None
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            li, lj = labels[i], labels[j]
            if li.box == lj.box:
                continue
            df = lj.fraction - li.fraction
            if abs(df) < 0.08:
                continue
            b = (lj.value - li.value) / df
            if b <= 0:
                continue
            a = li.value - b * li.fraction
            # per box, the alternative closest to the line
            per_box: dict[int, tuple[float, Label]] = {}
            for lab in labels:
                e = abs(a + b * lab.fraction - lab.value) / b
                if e <= tol and (lab.box not in per_box or e < per_box[lab.box][0]):
                    per_box[lab.box] = (e, lab)
            inl = [lab for _, lab in per_box.values()]
            # distinct values only: the same number read twice is one label. A label read as
            # printed counts more than one with a restored decimal point, so a decimal line wins
            # only when it explains clearly more labels.
            k = len({round(lab.value, 9) for lab in inl})
            as_read = sum(lab.value == parse_number(lab.text)[0] for lab in inl)
            score = (k + 0.5 * as_read, k, -sum(e for e, _ in per_box.values()))
            if best is None or score > best[0]:
                best = (score, inl)
    if best is None or best[0][1] < max(min_inliers, 2):
        return None
    inl = best[1]
    f = np.array([lab.fraction for lab in inl])
    v = np.array([lab.value for lab in inl])
    b, a = np.polyfit(f, v, 1)
    if b <= 0:
        return None
    return float(a), float(b), inl


def _snap(x: float, span: float, values: Sequence[float], step: float | None) -> float:
    near = [v for v in values if abs(v - x) <= SNAP_TOL * span]
    if near:
        return min(near, key=lambda v: abs(v - x))
    if abs(x) <= SNAP_TOL * span:
        return 0.0
    if step:
        return round(x / step) * step
    return x


def read_scale(boxes: Sequence[TextBox], kp: GaugeKeypoints, fit: DialFit | None = None) -> ScaleRead:
    labels = labels_on_dial(boxes, kp, fit)
    unit = find_unit([t for t, _, _ in boxes])
    line = fit_line(labels)
    if line is None:
        return ScaleRead(None, None, unit, labels)
    a, b, inl = line
    vals = sorted({lab.value for lab in inl})
    step = float(np.min(np.diff(vals))) if len(vals) >= 2 else None
    lo, hi = _snap(a, b, vals, step), _snap(a + b, b, vals, step)
    rms = float(np.sqrt(np.mean([((a + b * lab.fraction - lab.value) / b) ** 2 for lab in inl])))
    if hi <= lo:
        return ScaleRead(None, None, unit, labels, inl, rms)
    return ScaleRead(lo, hi, unit, labels, inl, rms)


def dial_crop(img, kp, factor: float):
    """The dial around the predicted centre, ``factor`` scale radii on each side (0 = whole image)."""
    if factor <= 0:
        return img, (0, 0)
    cx, cy = kp.center
    r = max(np.hypot(kp.scale_min[0] - cx, kp.scale_min[1] - cy),
            np.hypot(kp.scale_max[0] - cx, kp.scale_max[1] - cy)) * factor
    h, w = img.shape[:2]
    x0, y0 = max(int(cx - r), 0), max(int(cy - r), 0)
    x1, y1 = min(int(cx + r) + 1, w), min(int(cy + r) + 1, h)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return img, (0, 0)
    return img[y0:y1, x0:x1], (x0, y0)


class RapidOcr:
    """rapidocr-onnxruntime; ``__call__(image) -> list[TextBox]``.

    The text-angle classifier is off by default: dial numbers are printed upright on the face,
    and skipping it cuts OCR time by about a quarter with the same MeterFL ranges (233 vs 234
    of 261 right, 2026-11-17). ``det_side`` sets the detector input size (package default 736,
    as the shorter side); smaller is faster and loses small labels."""

    def __init__(self, use_angle_cls: bool = False, det_side: int | None = None):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()
        self.engine.use_angle_cls = use_angle_cls      # read per call by rapidocr 1.2–1.4
        if det_side:
            try:
                self.engine.text_detector.preprocess_op[0].limit_side_len = det_side
            except (AttributeError, IndexError):   # other package layout: keep the default
                pass

    def __call__(self, img) -> list[TextBox]:
        res, _ = self.engine(img)
        return [(str(t), [tuple(map(float, p)) for p in box], float(c)) for box, t, c in (res or [])]
