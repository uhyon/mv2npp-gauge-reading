"""Evaluation 3 (keypoints, Roboflow hustlegrad v4tii) and 4 (detection, Roboflow obaw9).

Neither set has a reading ground truth, so they test the stages before the reading:

* v4tii (``data/raw/roboflow/gauge-detection-2-2ynzm``): Ultralytics pose labels with classes
  Gauge / Max / Min / Needle and 2 keypoints per row. Gauge rows carry the box only, Min and Max
  carry the tick position in point 1, Needle carries its two ends (pivot and tip). Images were
  exported with a 640x640 stretch resize; the stretch is undone per gauge by making the Gauge box
  square (dials are round, assuming a roughly frontal view).
* obaw9 (``data/raw/roboflow/analog-gauge-obaw9``): YOLO segmentation polygons with classes
  center / meter / needle / scale, train split only. The box of a ``meter`` polygon is taken as
  the gauge box. None of it is used for training, so the whole set is held out.

Everything here works on plain arrays and label files, so it can be tested without a model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gauge_reader.reading.geometry import GaugeKeypoints, Point, needle_fraction

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
V4TII_NAMES = ("Gauge", "Max", "Min", "Needle")
OBAW9_NAMES = ("center", "meter", "needle", "scale")

Box = tuple[float, float, float, float]   # x0, y0, x1, y1 in px


# --------------------------------------------------------------------------- files

def original_name(stem: str) -> str:
    """Roboflow file stem without the export suffix: ``foo_jpg.rf.abc123`` -> ``foo_jpg``."""
    return re.split(r"\.rf\.", stem, maxsplit=1)[0]


def list_images(split_dir: Path) -> list[Path]:
    d = Path(split_dir) / "images"
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


def label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / (image.stem + ".txt")


def v4tii_images(root: Path, splits: tuple[str, ...] = ("train", "valid", "test")) -> list[tuple[str, Path]]:
    """(split, image) pairs; train is deduped to one file per original (it is tripled by
    brightness augmentation), valid and test are not augmented."""
    out: list[tuple[str, Path]] = []
    for split in splits:
        seen: set[str] = set()
        for img in list_images(Path(root) / split):
            key = original_name(img.stem)
            if split == "train":
                if key in seen:
                    continue
                seen.add(key)
            out.append((split, img))
    return out


# --------------------------------------------------------------------------- labels

@dataclass
class GaugeLabel:
    box: Box
    center: Point | None = None     # needle pivot
    tip: Point | None = None
    scale_min: Point | None = None
    scale_max: Point | None = None
    needle_swapped: bool = False    # the pivot was the needle row's second point

    @property
    def complete(self) -> bool:
        return None not in (self.center, self.tip, self.scale_min, self.scale_max)

    @property
    def size(self) -> float:
        """Gauge size used to normalise point errors: sqrt of the box area."""
        x0, y0, x1, y1 = self.box
        return math.sqrt(max(x1 - x0, 1e-9) * max(y1 - y0, 1e-9))

    def keypoints(self) -> GaugeKeypoints:
        return GaugeKeypoints(self.center, self.tip, self.scale_min, self.scale_max)


def _xywhn_to_xyxy(cx, cy, w, h, W, H) -> Box:
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def _inside(p: Point, b: Box, margin: float = 0.1) -> bool:
    x0, y0, x1, y1 = b
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return x0 - mx <= p[0] <= x1 + mx and y0 - my <= p[1] <= y1 + my


def _box_center(b: Box) -> Point:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_v4tii(text: str, width: int, height: int) -> list[GaugeLabel]:
    """Gauges of one image, each with the Min / Max / Needle rows that fall in its box.

    The needle row's end nearer to the box centre is taken as the pivot (``needle_swapped``
    records when that is the second point, against the dataset notes)."""
    gauges, parts = [], []
    for line in text.splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        cls = int(float(v[0]))
        nums = [float(x) for x in v[1:]]
        box = _xywhn_to_xyxy(*nums[:4], width, height)
        pts = []
        for i in range(4, len(nums) - 2, 3):
            x, y, vis = nums[i], nums[i + 1], nums[i + 2]
            pts.append((x * width, y * height) if vis > 0 and (x > 0 or y > 0) else None)
        if cls == 0:
            gauges.append(GaugeLabel(box))
        else:
            parts.append((cls, box, pts))
    for cls, box, pts in parts:
        p = [q for q in pts if q is not None]
        if not p:
            continue
        ref = p[0]
        owners = [g for g in gauges if _inside(ref, g.box)]
        if not owners:
            continue
        g = min(owners, key=lambda g: _dist(ref, _box_center(g.box)))
        if cls == 1:
            g.scale_max = p[0]
        elif cls == 2:
            g.scale_min = p[0]
        elif cls == 3 and len(p) >= 2:
            c = _box_center(g.box)
            a, b = p[0], p[1]
            if _dist(b, c) < _dist(a, c):
                a, b = b, a
                g.needle_swapped = True
            g.center, g.tip = a, b
    return gauges


def parse_obaw9_boxes(text: str, width: int, height: int, cls: int = 1) -> list[Box]:
    """Boxes of the polygons of class ``cls`` (1 = meter)."""
    out = []
    for line in text.splitlines():
        v = line.split()
        if len(v) < 7 or int(float(v[0])) != cls:
            continue
        xy = np.array([float(x) for x in v[1:]]).reshape(-1, 2)
        out.append((xy[:, 0].min() * width, xy[:, 1].min() * height,
                    xy[:, 0].max() * width, xy[:, 1].max() * height))
    return out


# --------------------------------------------------------------------------- geometry

def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def unstretch(box: Box) -> tuple[float, float]:
    """(sx, sy) that make the box square by enlarging its shorter side."""
    w, h = box[2] - box[0], box[3] - box[1]
    if w <= 0 or h <= 0:
        return 1.0, 1.0
    return (h / w, 1.0) if w < h else (1.0, w / h)


def scale_kp(k: GaugeKeypoints, sx: float, sy: float) -> GaugeKeypoints:
    s = lambda p: (p[0] * sx, p[1] * sy)
    return GaugeKeypoints(s(k.center), s(k.tip), s(k.scale_min), s(k.scale_max))


def fraction_error(pred: float, true: float) -> float:
    """|pred - true| in % of the scale."""
    return abs(pred - true) * 100.0


# --------------------------------------------------------------------------- detection AP

@dataclass
class DetRecord:
    """Detections and ground truth of one image for AP."""
    gt: list[Box]
    pred: list[tuple[float, Box]] = field(default_factory=list)   # (conf, box)


def average_precision(records: list[DetRecord], thr: float = 0.5) -> dict:
    """COCO-style AP (101-point interpolation) of one class at IoU ``thr``.

    Also returns precision and recall at confidence 0.25 (the deployed threshold)."""
    n_gt = sum(len(r.gt) for r in records)
    scored = []   # (conf, is_tp)
    for r in records:
        used = [False] * len(r.gt)
        for conf, box in sorted(r.pred, key=lambda t: -t[0]):
            best, j = 0.0, -1
            for k, g in enumerate(r.gt):
                if used[k]:
                    continue
                o = iou(box, g)
                if o > best:
                    best, j = o, k
            if j >= 0 and best >= thr:
                used[j] = True
                scored.append((conf, 1))
            else:
                scored.append((conf, 0))
    if n_gt == 0:
        return {"ap": float("nan"), "n_gt": 0, "precision@0.25": float("nan"),
                "recall@0.25": float("nan")}
    scored.sort(key=lambda t: -t[0])
    tp = np.cumsum([s[1] for s in scored]) if scored else np.zeros(0)
    fp = np.cumsum([1 - s[1] for s in scored]) if scored else np.zeros(0)
    rec = tp / n_gt if len(tp) else np.zeros(0)
    prec = tp / np.maximum(tp + fp, 1e-9) if len(tp) else np.zeros(0)
    # precision envelope, then sample at 101 recall points
    env = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        idx = np.searchsorted(rec, t, side="left")
        ap += env[idx] if idx < len(env) else 0.0
    ap /= 101
    hi = [s for s in scored if s[0] >= 0.25]
    tp25 = sum(s[1] for s in hi)
    return {"ap": float(ap), "n_gt": n_gt,
            "precision@0.25": tp25 / len(hi) if hi else float("nan"),
            "recall@0.25": tp25 / n_gt}


def ap_range(records: list[DetRecord]) -> float:
    """Mean AP over IoU 0.50:0.95."""
    return float(np.mean([average_precision(records, t)["ap"] for t in np.arange(0.5, 0.96, 0.05)]))


# --------------------------------------------------------------------------- keypoint scoring

KP_FIELDS = ("center", "tip", "scale_min", "scale_max")


def match_gauges(gts: list[GaugeLabel], dets: list, thr: float = 0.3) -> list[tuple[GaugeLabel, object | None]]:
    """Greedy one-to-one IoU matching of ground-truth gauges to detections (by confidence)."""
    pairs, free = [], list(dets)
    for g in sorted(gts, key=lambda g: -g.size):
        best, j = 0.0, -1
        for k, d in enumerate(free):
            o = iou(g.box, d.box)
            if o > best:
                best, j = o, k
        if j >= 0 and best >= thr:
            pairs.append((g, free.pop(j)))
        else:
            pairs.append((g, None))
    return pairs


def score_gauge(g: GaugeLabel, det) -> dict:
    """Point errors (fraction of gauge size, after undoing the stretch) and needle fraction."""
    row: dict = {"complete": g.complete, "needle_swapped": g.needle_swapped,
                 "detected": det is not None}
    if det is None or not g.complete:
        return row
    sx, sy = unstretch(g.box)
    size = math.sqrt((g.box[2] - g.box[0]) * sx * (g.box[3] - g.box[1]) * sy)
    gk, pk = scale_kp(g.keypoints(), sx, sy), scale_kp(det.keypoints, sx, sy)
    row["box_iou"] = iou(g.box, det.box)
    row["box_conf"] = det.box_conf
    for f in KP_FIELDS:
        row[f"err_{f}"] = _dist(getattr(gk, f), getattr(pk, f)) / size
    try:
        true = needle_fraction(gk)
    except ValueError:
        return row
    row["true_fraction"] = true
    try:
        row["frac_err_img"] = fraction_error(needle_fraction(pk), true)
    except ValueError:
        pass
    fit = det.dial_fit() if hasattr(det, "dial_fit") else None
    if fit is not None:   # a homography fit is unaffected by the stretch
        row["frac_err_face"] = fraction_error(fit.fraction, true)
    return row


def summarize_keypoints(rows: list[dict]) -> dict:
    """Aggregate per-gauge rows from ``score_gauge``."""
    n = len(rows)
    det = [r for r in rows if r["detected"]]
    scored = [r for r in det if "err_center" in r]
    out: dict = {"gauges": n, "detected": len(det),
                 "detection_rate": len(det) / n if n else float("nan"),
                 "scored": len(scored),
                 "complete_labels": sum(r["complete"] for r in rows),
                 "needle_swapped": sum(r["needle_swapped"] for r in rows)}
    for f in KP_FIELDS:
        e = np.array([r[f"err_{f}"] for r in scored])
        if len(e):
            out[f"{f}_err_median"] = float(np.median(e))
            out[f"{f}_pck05"] = float(np.mean(e <= 0.05))
            out[f"{f}_pck10"] = float(np.mean(e <= 0.10))
    for key in ("frac_err_img", "frac_err_face"):
        e = np.array([r[key] for r in scored if key in r])
        if len(e):
            out[f"{key}_n"] = len(e)
            out[f"{key}_mean"] = float(e.mean())
            out[f"{key}_median"] = float(np.median(e))
            out[f"{key}_within2"] = float(np.mean(e <= 2))
            out[f"{key}_within5"] = float(np.mean(e <= 5))
    return out
