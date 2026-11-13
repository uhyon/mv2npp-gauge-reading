"""Turn YOLO pose output into gauge keypoints and a reading.

The pose model predicts one box per gauge and four keypoints in the order used by the
synthetic labels (``scripts/generate_synth.py``): center, needle tip, scale min, scale max.
This module works on plain arrays so it can be tested without Ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gauge_reader.reading.geometry import GaugeKeypoints, needle_fraction

KP_NAMES = ("center", "tip", "scale_min", "scale_max")


@dataclass
class GaugeDetection:
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 in image px
    box_conf: float
    keypoints: GaugeKeypoints
    kpt_conf: tuple[float, float, float, float]

    @property
    def min_kpt_conf(self) -> float:
        return float(min(self.kpt_conf))

    def fraction(self) -> float:
        return needle_fraction(self.keypoints)

    def reading(self, min_value: float, max_value: float) -> float:
        return min_value + self.fraction() * (max_value - min_value)


def detections_from_arrays(boxes: np.ndarray, box_conf: np.ndarray, kpts: np.ndarray,
                           kpt_conf: np.ndarray | None = None) -> list[GaugeDetection]:
    """boxes (N,4) xyxy, box_conf (N,), kpts (N,4,2) xy, kpt_conf (N,4). Sorted by box_conf."""
    boxes, box_conf, kpts = np.asarray(boxes), np.asarray(box_conf), np.asarray(kpts)
    if kpt_conf is None:
        kpt_conf = np.ones(kpts.shape[:2])
    out = []
    for b, c, k, kc in zip(boxes, box_conf, kpts, np.asarray(kpt_conf)):
        pts = [tuple(map(float, p[:2])) for p in k]
        out.append(GaugeDetection(tuple(map(float, b)), float(c),
                                  GaugeKeypoints(*pts), tuple(map(float, kc))))
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
                          det.kpt_conf)
