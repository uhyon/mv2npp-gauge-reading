"""Turn YOLO pose output into gauge keypoints and a reading.

The pose model predicts one box per gauge and four keypoints in the order used by the
synthetic labels (``scripts/generate_synth.py``): center, needle tip, scale min, scale max.
Models trained on the arc labels (``scripts/arc_labels.py``) add three more points on the
scale arc, at 25, 50 and 75% of the scale; their readings are taken on the dial face
(``gauge_reader.reading.perspective``). This module works on plain arrays so it can be tested
without Ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gauge_reader.reading.geometry import GaugeKeypoints, Point, clockwise_angle, needle_fraction
from gauge_reader.reading.perspective import DialFit, fit_dial

KP_NAMES = ("center", "tip", "scale_min", "scale_max")
ARC_NAMES = ("arc_25", "arc_50", "arc_75")   # scale-arc points of the arc models


@dataclass
class GaugeDetection:
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 in image px
    box_conf: float
    keypoints: GaugeKeypoints
    kpt_conf: tuple[float, ...]
    arc: tuple[Point, ...] = ()              # ARC_NAMES points; empty for 4-keypoint models
    _fit: DialFit | bool | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def min_kpt_conf(self) -> float:
        return float(min(self.kpt_conf))

    def dial_fit(self) -> DialFit | None:
        """Reading on the dial face (arc models only); None if the points fit no dial."""
        if not self.arc:
            return None
        if self._fit is None:
            try:
                self._fit = fit_dial(self.keypoints, self.arc)
            except (ValueError, np.linalg.LinAlgError):
                self._fit = False
        return self._fit or None

    def image_fraction(self) -> float:
        """Needle fraction from angles in the image plane (the 4-keypoint method)."""
        return needle_fraction(self.keypoints)

    def sweep(self) -> float:
        """Min-to-max scale arc: on the dial face when fitted, else in the image."""
        fit = self.dial_fit()
        if fit is not None:
            return fit.sweep
        k = self.keypoints
        return clockwise_angle(k.center, k.scale_min, k.scale_max)

    def fraction(self) -> float:
        fit = self.dial_fit()
        return fit.fraction if fit is not None else self.image_fraction()

    def reading(self, min_value: float, max_value: float) -> float:
        return min_value + self.fraction() * (max_value - min_value)


def detections_from_arrays(boxes: np.ndarray, box_conf: np.ndarray, kpts: np.ndarray,
                           kpt_conf: np.ndarray | None = None) -> list[GaugeDetection]:
    """boxes (N,4) xyxy, box_conf (N,), kpts (N,K,2) xy, kpt_conf (N,K). Sorted by box_conf.

    K is 4 (KP_NAMES) or 7 (KP_NAMES then ARC_NAMES)."""
    boxes, box_conf, kpts = np.asarray(boxes), np.asarray(box_conf), np.asarray(kpts)
    if kpts.ndim != 3 or kpts.shape[1] not in (len(KP_NAMES), len(KP_NAMES) + len(ARC_NAMES)):
        raise ValueError(f"keypoints of shape {kpts.shape}, expected (N, 4 or 7, 2)")
    if kpt_conf is None:
        kpt_conf = np.ones(kpts.shape[:2])
    out = []
    for b, c, k, kc in zip(boxes, box_conf, kpts, np.asarray(kpt_conf)):
        pts = [tuple(map(float, p[:2])) for p in k]
        out.append(GaugeDetection(tuple(map(float, b)), float(c), GaugeKeypoints(*pts[:4]),
                                  tuple(map(float, kc)), tuple(pts[4:])))
    return sorted(out, key=lambda d: -d.box_conf)


def detections_from_result(result) -> list[GaugeDetection]:
    """From one ``ultralytics`` Results object of a pose model."""
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return []
    kp = result.keypoints
    conf = kp.conf.cpu().numpy() if kp.conf is not None else None
    return detections_from_arrays(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy(),
                                  kp.xy.cpu().numpy(), conf)


def pad_image(img: np.ndarray, frac: float, value: int = 114) -> tuple[np.ndarray, int, int]:
    """Pad a tight crop on every side by ``frac`` of its size. Returns image and (dx, dy).

    The synthetic training images show the gauge at 25–90% of the frame with background
    around it; test crops (MeterFL) fill the frame, so a margin brings them closer to that.
    """
    if frac <= 0:
        return img, 0, 0
    h, w = img.shape[:2]
    dx, dy = round(w * frac), round(h * frac)
    out = np.full((h + 2 * dy, w + 2 * dx) + img.shape[2:], value, dtype=img.dtype)
    out[dy:dy + h, dx:dx + w] = img
    return out, dx, dy


def shift(det: GaugeDetection, dx: float, dy: float) -> GaugeDetection:
    """Move a detection made on a padded image back to the original image coordinates."""
    k = det.keypoints
    mv = lambda p: (p[0] - dx, p[1] - dy)
    x0, y0, x1, y1 = det.box
    return GaugeDetection((x0 - dx, y0 - dy, x1 - dx, y1 - dy), det.box_conf,
                          GaugeKeypoints(mv(k.center), mv(k.tip), mv(k.scale_min), mv(k.scale_max)),
                          det.kpt_conf, tuple(mv(p) for p in det.arc))
