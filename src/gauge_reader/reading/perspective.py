"""Read the needle on the dial face rather than in the image plane.

``geometry.needle_fraction`` measures angles around the centre in the image. On a tilted dial
those angles are distorted: with the true keypoints of the synthetic val set the reading is
already off by 1.75% FS on average, 3.1% for faces tilted 40° or more (docs/RESULTS.md).

Models trained with the scale-arc keypoints (``ARC_FRACTIONS`` of the way from the min to the
max tick) give five points on the scale arc plus the centre. Those are the image of a circle and
its centre, with the five points evenly spread in angle over the sweep. ``fit_dial`` finds the
homography from that model to the image and the sweep that fit best, then reads the tip's angle
on the face, which the view no longer distorts. The fit residual (``rms``) says how well the
keypoints agree with a round dial.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gauge_reader.reading.geometry import GaugeKeypoints, Point

ARC_FRACTIONS = (0.25, 0.5, 0.75)   # scale-arc keypoints between the min and max ticks
SWEEP_RANGE = (20.0, 359.0)          # sweeps searched, degrees


@dataclass(frozen=True)
class DialFit:
    fraction: float    # needle position on the scale: 0 at the min tick, 1 at the max tick
    sweep: float       # min-to-max scale arc on the face, degrees
    rms: float         # fit residual, as a share of the scale radius in the image
    homography: np.ndarray  # face (unit circle, min tick at angle 0) to image


def _project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homographies ``h`` (..., 3, 3) to points ``pts`` (..., N, 2)."""
    v = np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1) @ np.swapaxes(h, -1, -2)
    return v[..., :2] / v[..., 2:3]


def _face_points(sweeps: np.ndarray, fractions: Sequence[float]) -> np.ndarray:
    """(B, N, 2): scale points at ``fractions`` of each sweep on a unit circle, then the centre."""
    a = np.radians(np.asarray(sweeps)[:, None] * np.asarray(fractions)[None, :])
    pts = np.stack([np.cos(a), np.sin(a)], axis=-1)
    return np.concatenate([pts, np.zeros((len(a), 1, 2))], axis=1)


def _fit_batch(sweeps: np.ndarray, img: np.ndarray, fractions: Sequence[float]):
    """Least-squares (DLT) homography from the face model to the image for every sweep.

    Returns the homographies (B, 3, 3) and the summed squared reprojection error in px (B,)."""
    # normalise the image points (Hartley) so the linear solve is well conditioned
    c = img.mean(axis=0)
    scale = math.sqrt(2) / max(float(np.mean(np.linalg.norm(img - c, axis=1))), 1e-9)
    t = np.array([[scale, 0, -scale * c[0]], [0, scale, -scale * c[1]], [0, 0, 1.0]])
    u, v = ((img - c) * scale).T
    face = _face_points(sweeps, fractions)
    x, y = face[..., 0], face[..., 1]
    o, z = np.ones_like(x), np.zeros_like(x)
    a = np.concatenate([np.stack([x, y, o, z, z, z, -u * x, -u * y, -u * o], axis=-1),
                        np.stack([z, z, z, x, y, o, -v * x, -v * y, -v * o], axis=-1)], axis=1)
    h = np.linalg.inv(t) @ np.linalg.svd(a)[2][:, -1, :].reshape(-1, 3, 3)
    err = np.sum((_project(h, face) - img) ** 2, axis=(1, 2))
    return h, np.where(np.isfinite(err), err, np.inf)


def fit_dial(kp: GaugeKeypoints, arc: Sequence[Point],
             fractions: Sequence[float] = ARC_FRACTIONS) -> DialFit:
    """Needle fraction read on the dial face from the centre, tip, min, ``arc`` and max points.

    ``arc`` holds the keypoints at ``fractions`` of the scale. Raises ValueError when no
    homography fits (e.g. all scale points on one line).
    """
    if len(arc) != len(fractions):
        raise ValueError(f"{len(arc)} arc points for {len(fractions)} fractions")
    fr = (0.0, *fractions, 1.0)
    img = np.array([kp.scale_min, *arc, kp.scale_max, kp.center], dtype=np.float64)
    if not np.all(np.isfinite(img)):
        raise ValueError("keypoints are not finite")

    # every sweep on a 1° grid, then a 0.02° grid around the best one
    sweeps = np.arange(SWEEP_RANGE[0], SWEEP_RANGE[1] + 1e-9, 1.0)
    _, err = _fit_batch(sweeps, img, fr)
    if not np.isfinite(err).any():
        raise ValueError("no homography fits these keypoints")
    best = sweeps[int(np.argmin(err))]
    sweeps = np.clip(np.arange(best - 1.0, best + 1.0 + 1e-9, 0.02), *SWEEP_RANGE)
    hs, err = _fit_batch(sweeps, img, fr)
    k = int(np.argmin(err))
    sweep, h, sq = float(sweeps[k]), hs[k], float(err[k])
    if abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("degenerate homography")

    q = _project(np.linalg.inv(h), np.array([kp.tip], dtype=np.float64))[0]
    t = math.degrees(math.atan2(q[1], q[0])) % 360.0
    if t > sweep and t - sweep > (360.0 - sweep) / 2:  # dead zone: closer to the min stop
        t -= 360.0
    radius = float(np.mean(np.linalg.norm(img[:-1] - img[-1], axis=1)))
    rms = math.sqrt(sq / len(img)) / radius if radius > 0 else math.inf
    return DialFit(fraction=t / sweep, sweep=sweep, rms=rms, homography=h)


def arc_points(homography: np.ndarray, dial_center: Point, dial_scale_min: Point, sweep_deg: float,
               fractions: Sequence[float] = ARC_FRACTIONS) -> list[Point]:
    """Image points of the scale arc at ``fractions`` of the sweep, for the synthetic labels.

    ``homography`` maps the flat dial image (``dial_*`` coordinates, y down) to the photo, as
    stored in a generated set's labels.jsonl; the scale runs clockwise from ``dial_scale_min``.
    """
    cx, cy = dial_center
    r = math.dist(dial_center, dial_scale_min)
    a0 = math.atan2(dial_scale_min[1] - cy, dial_scale_min[0] - cx)
    a = a0 + np.radians(np.asarray(fractions) * sweep_deg)
    pts = _project(np.asarray(homography, dtype=np.float64),
                   np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=-1))
    return [(float(x), float(y)) for x, y in pts]
