"""Run the pose model over a list of test images and turn each result into a reading.

Used by scripts/predict_readings.py (one manifest) and scripts/robustness.py (the same
images under each degradation). The scale range comes from the row (ground truth), so the
reading measures the keypoint stage only.

Two physical checks turn keypoint failures into bounded errors or declines:

- the scale arc from the min to the max keypoint must be at least ``min_sweep`` degrees
  (real dials sweep 90–340°; ours 180–270°). A smaller arc means the min and max points
  were confused or collapsed, and dividing by it can blow a reading up by 100x, so the
  row is declined with status ``bad_geometry``;
- the needle fraction is clipped to [-clip, 1 + clip]: a needle cannot sit far past its
  stops. The raw fraction is kept in the ``fraction`` column.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from gauge_reader.keypoints.yolo import KP_NAMES, detections_from_result, pad_image, shift
from gauge_reader.reading.geometry import clockwise_angle

COLORS = {"center": (255, 255, 0), "tip": (0, 0, 255), "scale_min": (255, 0, 255),
          "scale_max": (0, 165, 255)}
FIELDS = ["id", "value", "pred_unit", "box_conf", "min_kpt_conf", "fraction", "sweep", "n_det",
          "status", "ms"] + [f"{k}_{c}" for k in KP_NAMES for c in "xy"]
MIN_SWEEP_DEG = 90.0
CLIP = 0.1
STATUSES = ("ok", "no_range", "no_detection", "unreadable", "bad_geometry")


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
                kp = d.keypoints
                sweep = clockwise_angle(kp.center, kp.scale_min, kp.scale_max)
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


def median_ms(records: list[dict]) -> float | None:
    ms = sorted(float(r["ms"]) for r in records if r.get("ms"))
    return ms[len(ms) // 2] if ms else None


def write_records(records: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)
