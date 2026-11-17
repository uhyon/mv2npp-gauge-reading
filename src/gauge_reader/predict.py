"""Run the pose model over a list of test images and turn each result into a reading.

Used by scripts/predict_readings.py (one manifest) and scripts/robustness.py (the same
images under each degradation). The scale range comes from the row (ground truth), so the
reading measures the keypoint stage only.

Models with the scale-arc keypoints are read on the dial face (``DialFit``, which undoes the
tilt of the view); ``fit_rms`` records how well their points fit a round dial, and
``fraction_img`` the image-plane reading for comparison. 4-keypoint models are read in the
image plane.

Two physical checks turn keypoint failures into bounded errors or declines:

- the scale arc from the min to the max keypoint must be at least ``min_sweep`` degrees
  (real dials sweep 90–340°; ours 180–270°). A smaller arc means the min and max points
  were confused or collapsed, and dividing by it can blow a reading up by 100x, so the
  row is declined with status ``bad_geometry``;
- the needle fraction is clipped to [-clip, 1 + clip]: a needle cannot sit far past its
  stops. The raw fraction is kept in the ``fraction`` column.

Rotation check and RETAKE (``read_rows_tta``): each image is read upright and turned by
+/- ``angle`` degrees. A reading does not change when the photo is turned, so the value is the
median of the three readings, and when they differ by more than ``retake_spread`` % of the
scale the model is unsure of the dial (turned far from upright, unusual design, blur) and the
row is declined with status ``retake``: the reader asks for another photo instead of giving a
wrong value. The median reading is still recorded in ``value_median``. Chosen 2026-11-17 from
MeterFL, MeasureBench and the synthetic val split (docs/RESULTS.md): spread > 5% flags about
9%, 55% and 1% of them, and the spread follows the error far better than keypoint confidence
or ``fit_rms``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from gauge_reader.keypoints.yolo import (
    ARC_NAMES,
    KP_NAMES,
    detections_from_result,
    pad_image,
    shift,
)

COLORS = {"center": (255, 255, 0), "tip": (0, 0, 255), "scale_min": (255, 0, 255),
          "scale_max": (0, 165, 255), "arc": (0, 255, 255)}
FIELDS = ["id", "value", "pred_unit", "box_conf", "min_kpt_conf", "fraction", "fraction_img", "sweep",
          "fit_rms", "n_det", "status", "ms"] + [f"{k}_{c}" for k in KP_NAMES + ARC_NAMES for c in "xy"]
TTA_FIELDS = ["value_median", "value_0", "frac_0", "frac_p", "frac_m", "spread"]
MIN_SWEEP_DEG = 90.0
CLIP = 0.1
STATUSES = ("ok", "no_range", "no_detection", "unreadable", "bad_geometry", "retake")
TTA_ANGLE = 25.0
RETAKE_SPREAD = 5.0   # % of scale


def _float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def draw(img: np.ndarray, det, text: str) -> np.ndarray:
    out = img.copy()
    s = max(1, round(max(img.shape[:2]) / 300))
    x0, y0, x1, y1 = map(int, det.box)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), s)
    c = tuple(map(int, det.keypoints.center))
    cv2.line(out, c, tuple(map(int, det.keypoints.tip)), (0, 0, 255), s)
    for name in KP_NAMES:
        p = tuple(map(int, getattr(det.keypoints, name)))
        cv2.circle(out, p, 3 * s, COLORS[name], -1)
    for p in det.arc:
        cv2.circle(out, tuple(map(int, p)), 2 * s, COLORS["arc"], -1)
    cv2.putText(out, text, (4, 14 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (0, 255, 0), s)
    return out


def read_rows(model, rows: list[dict], *, imgsz: int = 640, conf: float = 0.1, pad: float = 0.15,
              device=None, overlay: Path | None = None, min_sweep: float = MIN_SWEEP_DEG,
              clip: float | None = CLIP,
              transform: Callable[[np.ndarray, dict], np.ndarray] | None = None,
              progress: int = 0) -> tuple[list[dict], dict[str, int]]:
    """Read every row (keys id, image, scale_min, scale_max, unit).

    transform(img, row) is applied to each loaded image first (e.g. a degradation).
    min_sweep and clip: see the module docstring (clip None = no clipping).
    Returns one record per row (FIELDS) and the count per status. ``ms`` is the model
    time for the image (preprocess, inference and postprocess, without file reading).
    """
    if overlay:
        overlay.mkdir(parents=True, exist_ok=True)
    out, counts = [], dict.fromkeys(STATUSES, 0)
    for i, row in enumerate(rows):
        rec = {"id": row["id"], "value": "", "pred_unit": "", "status": "ok"}
        lo, hi = _float(row.get("scale_min")), _float(row.get("scale_max"))
        img = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
        if img is None:
            rec["status"] = "unreadable"
        elif lo is None or hi is None:
            rec["status"] = "no_range"
        else:
            if transform is not None:
                img = transform(img, row)
            padded, dx, dy = pad_image(img, pad)
            t0 = time.perf_counter()
            res = model.predict(padded, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            rec["ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
            dets = [shift(d, dx, dy) for d in detections_from_result(res)]
            rec["n_det"] = len(dets)
            if not dets:
                rec["status"] = "no_detection"
            else:
                d = dets[0]
                rec.update(box_conf=f"{d.box_conf:.4f}", min_kpt_conf=f"{d.min_kpt_conf:.4f}")
                for k in KP_NAMES:
                    x, y = getattr(d.keypoints, k)
                    rec[f"{k}_x"], rec[f"{k}_y"] = f"{x:.1f}", f"{y:.1f}"
                for k, (x, y) in zip(ARC_NAMES, d.arc):
                    rec[f"{k}_x"], rec[f"{k}_y"] = f"{x:.1f}", f"{y:.1f}"
                fit = d.dial_fit()
                if fit is not None:
                    rec["fit_rms"] = f"{fit.rms:.4f}"
                try:
                    rec["fraction_img"] = f"{d.image_fraction():.4f}"
                except ValueError:  # min and max in the same direction
                    pass
                sweep = d.sweep()
                rec["sweep"] = f"{sweep:.1f}"
                if sweep < max(min_sweep, 1e-6):
                    rec["status"] = "bad_geometry"
                else:
                    raw = d.fraction()
                    frac = raw if clip is None else min(max(raw, -clip), 1.0 + clip)
                    rec.update(value=f"{lo + frac * (hi - lo):.6g}", pred_unit=row.get("unit", ""),
                               fraction=f"{raw:.4f}")
                if overlay:
                    text = f"{rec['value'] or '?'} {row.get('unit', '')} conf {d.box_conf:.2f}"
                    cv2.imwrite(str(overlay / f"{row['id']}.jpg"), draw(img, d, text))
        counts[rec["status"]] += 1
        out.append(rec)
        if progress and (i + 1) % progress == 0:
            print(f"  {i + 1}/{len(rows)}")
    return out, counts


def rotate(angle: float) -> Callable[[np.ndarray, dict], np.ndarray]:
    """Transform that turns an image by ``angle`` degrees (counter-clockwise) about its centre."""
    def f(img: np.ndarray, row: dict) -> np.ndarray:
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, m, (w, h), borderValue=(114, 114, 114))
    return f


def combine_tta(rows: list[dict], passes: list[list[dict]], *, clip: float | None = CLIP,
                retake_spread: float | None = RETAKE_SPREAD) -> list[dict]:
    """Merge the upright, +angle and -angle passes of ``read_rows`` (records in row order).

    Keeps the upright record (keypoints, confidences) and adds TTA_FIELDS. ``spread`` is
    max - min needle fraction in % of scale (99 when fewer than two passes gave a fraction).
    ``value`` becomes the median reading, or blank with status ``retake`` when the spread is
    over ``retake_spread`` (None = never). ``ms`` is the sum over the passes."""
    c = float("inf") if clip is None else clip
    out = []
    for row, recs in zip(rows, zip(*passes)):
        rec = dict(recs[0])
        fr = []
        for r in recs:
            try:
                fr.append(min(max(float(r["fraction"]), -c), 1.0 + c))
            except (KeyError, ValueError, TypeError):
                fr.append(None)
        for k, f in zip(("frac_0", "frac_p", "frac_m"), fr):
            rec[k] = "" if f is None else f"{f:.4f}"
        got = [f for f in fr if f is not None]
        spread = (max(got) - min(got)) * 100 if len(got) >= 2 else 99.0
        rec["spread"] = f"{spread:.2f}"
        rec["value_0"] = recs[0].get("value", "")
        ms = [float(r["ms"]) for r in recs if r.get("ms")]
        if ms:
            rec["ms"] = f"{sum(ms):.1f}"
        lo, hi = _float(row.get("scale_min")), _float(row.get("scale_max"))
        if got and lo is not None and hi is not None and rec["status"] == "ok":
            med = lo + float(np.median(got)) * (hi - lo)
            rec["value_median"] = f"{med:.6g}"
            if retake_spread is not None and spread > retake_spread:
                rec.update(status="retake", value="", pred_unit="")
            else:
                rec.update(status="ok", value=f"{med:.6g}", pred_unit=row.get("unit", ""))
        out.append(rec)
    return out


def read_rows_tta(model, rows: list[dict], *, angle: float = TTA_ANGLE,
                  retake_spread: float | None = RETAKE_SPREAD,
                  transform: Callable[[np.ndarray, dict], np.ndarray] | None = None,
                  **kw) -> tuple[list[dict], dict[str, int]]:
    """``read_rows`` upright and turned by +/- ``angle``, merged by ``combine_tta``.

    ``transform`` (e.g. a degradation) is applied before the turn; other keywords go to
    ``read_rows`` (the overlay is drawn for the upright pass only)."""
    passes = []
    for i, ang in enumerate((0.0, angle, -angle)):
        def tf(img, row, ang=ang):
            img = transform(img, row) if transform is not None else img
            return rotate(ang)(img, row) if ang else img
        recs, _ = read_rows(model, rows, transform=tf, **(kw if i == 0 else
                                                           {k: v for k, v in kw.items() if k != "overlay"}))
        passes.append(recs)
    out = combine_tta(rows, passes, clip=kw.get("clip", CLIP), retake_spread=retake_spread)
    counts = dict.fromkeys(STATUSES, 0)
    for r in out:
        counts[r["status"]] += 1
    return out, counts


def median_ms(records: list[dict]) -> float | None:
    ms = sorted(float(r["ms"]) for r in records if r.get("ms"))
    return ms[len(ms) // 2] if ms else None


def write_records(records: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        extra = [k for k in TTA_FIELDS if any(k in r for r in records)]
        w = csv.DictWriter(f, fieldnames=FIELDS + extra)
        w.writeheader()
        w.writerows(records)
