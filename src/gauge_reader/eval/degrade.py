"""Test-time degradations for the robustness evaluation (ER-05, FR-13).

Each condition has three fixed severities and is applied alone to the clean test images,
with a seed derived from the image id, so every model sees exactly the same degraded
images. Parameters follow docs/DATASETS.md, "Robustness protocol":

| condition         | severity 1 / 2 / 3                                                   |
| ----------------- | -------------------------------------------------------------------- |
| low_light         | gamma 1.8 / 2.5 / 3.2, then Poisson shot noise on the darkened image |
| radiation_speckle | 0.05% / 0.2% / 0.5% of pixels saturated; 10% of hits are short streaks |
| motion_blur       | linear kernel 7 / 15 / 25 px at native resolution, random direction  |

The training generator uses randomized versions of the same effects
(``gauge_reader.synth.scene``), so this measures whether the model learned to cope with
them on real photos, not whether it has seen these images.
"""

from __future__ import annotations

import math
import zlib

import cv2
import numpy as np

CONDITIONS = ("low_light", "radiation_speckle", "motion_blur")
SEVERITIES = (1, 2, 3)

GAMMA = {1: 1.8, 2: 2.5, 3: 3.2}
SPECKLE_FRACTION = {1: 0.0005, 2: 0.002, 3: 0.005}
STREAK_SHARE = 0.1
BLUR_PX = {1: 7, 2: 15, 3: 25}
PHOTONS_AT_WHITE = 400.0  # photon count of a white pixel in the shot-noise model


def rng_for(image_id: str, condition: str, severity: int, seed: int = 0) -> np.random.Generator:
    """Generator fixed by (seed, image id, condition, severity)."""
    return np.random.default_rng([seed, zlib.crc32(image_id.encode("utf-8")),
                                  CONDITIONS.index(condition), severity])


def low_light(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Darken with a gamma curve, then add photon (Poisson) noise, which is relatively
    stronger the darker the pixel is."""
    f = (img.astype(np.float32) / 255.0) ** GAMMA[severity]
    noisy = rng.poisson(f * PHOTONS_AT_WHITE).astype(np.float32) / PHOTONS_AT_WHITE
    return np.clip(noisy * 255.0 + 0.5, 0, 255).astype(np.uint8)


def radiation_speckle(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Saturated single-pixel hits; one in ten is drawn as a short streak (2–8 px)."""
    out = img.copy()
    h, w = img.shape[:2]
    n = max(1, round(h * w * SPECKLE_FRACTION[severity]))
    n_streak = round(n * STREAK_SHARE)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    out[ys[n_streak:], xs[n_streak:]] = 255
    white = (255,) * (out.shape[2] if out.ndim == 3 else 1)
    for y, x in zip(ys[:n_streak].tolist(), xs[:n_streak].tolist()):
        length, a = rng.uniform(2, 8), rng.uniform(0, math.pi)
        end = (round(x + length * math.cos(a)), round(y + length * math.sin(a)))
        cv2.line(out, (x, y), end, white, 1)
    return out


def motion_blur(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Linear motion blur of BLUR_PX[severity] pixels in a random direction."""
    n = BLUR_PX[severity]
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1.0
    M = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), rng.uniform(0, 180), 1.0)
    k = cv2.warpAffine(k, M, (n, n))
    k /= max(float(k.sum()), 1e-6)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


_FUNCS = {"low_light": low_light, "radiation_speckle": radiation_speckle,
          "motion_blur": motion_blur}


def degrade(img: np.ndarray, condition: str, severity: int,
            rng: np.random.Generator) -> np.ndarray:
    if condition not in _FUNCS:
        raise ValueError(f"unknown condition {condition!r}, expected one of {CONDITIONS}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    return _FUNCS[condition](img, severity, rng)
