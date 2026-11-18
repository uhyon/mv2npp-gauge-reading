"""Picture of a reading (FR-11): the gauge box, keypoints and scale arc drawn on the photo,
with the value or RETAKE as a label. Used by the demo and by ``read_gauge.py --draw``."""

from __future__ import annotations

import cv2
import numpy as np

from gauge_reader.reader import Reading

COLORS = {"NORMAL": (60, 170, 60), "LOW": (40, 140, 240), "HIGH": (40, 40, 220),
          "": (60, 170, 60), "RETAKE": (0, 160, 230)}   # BGR
KP_COLORS = {"center": (255, 255, 255), "tip": (40, 40, 220), "scale_min": (230, 160, 0),
             "scale_max": (230, 160, 0), "arc": (230, 200, 120)}


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def label_text(r: Reading) -> str:
    if r.value is None:
        return f"RETAKE ({r.status})"
    v = f"{r.value:.4g} {r.unit}".strip()
    return f"{v}  {r.alarm}" if r.alarm else v


def draw_reading(img: np.ndarray, r: Reading) -> np.ndarray:
    """A copy of ``img`` with the reading drawn on it (BGR)."""
    out = img.copy()
    h, w = out.shape[:2]
    s = max(1.0, max(h, w) / 640)                  # line width and font follow the photo size
    th, rad = max(1, round(2 * s)), max(2, round(4 * s))
    color = COLORS.get(r.alarm if r.value is not None else "RETAKE", (60, 170, 60))
    if r.box is not None:
        x0, y0, x1, y1 = map(int, map(round, r.box))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, th)
    k = r.keypoints
    if k is not None:
        ring = [k.scale_min, *r.arc, k.scale_max]
        if r.fit is not None:      # the fitted scale arc, through the dial-face homography
            from gauge_reader.reading.perspective import _project
            a = np.radians(np.linspace(0.0, r.fit.sweep, 90))
            ring = _project(np.asarray(r.fit.homography, dtype=np.float64),
                            np.stack([np.cos(a), np.sin(a)], axis=-1))
        cv2.polylines(out, [np.array([_pt(p) for p in ring], np.int32)], False,
                      KP_COLORS["arc"], th, cv2.LINE_AA)
        cv2.line(out, _pt(k.center), _pt(k.tip), KP_COLORS["tip"], th + 1, cv2.LINE_AA)
        for p in r.arc:
            cv2.circle(out, _pt(p), rad, KP_COLORS["arc"], -1, cv2.LINE_AA)
        for name in ("scale_min", "scale_max", "center", "tip"):
            cv2.circle(out, _pt(getattr(k, name)), rad + 1, KP_COLORS[name], -1, cv2.LINE_AA)
            cv2.circle(out, _pt(getattr(k, name)), rad + 1, (0, 0, 0), 1, cv2.LINE_AA)
    text = label_text(r)
    fs = 0.6 * s
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    x = 0 if r.box is None else max(0, min(int(r.box[0]), w - tw - 8))
    y = tht + 8 if r.box is None else max(tht + 8, int(r.box[1]) - 4)
    cv2.rectangle(out, (x, y - tht - 8), (x + tw + 8, y + base - 2), color, -1)
    cv2.putText(out, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th,
                cv2.LINE_AA)
    return out
