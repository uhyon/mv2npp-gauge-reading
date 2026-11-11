"""Place a rendered dial in a plant-like scene and apply NPP field degradations.

Glass effects (glare, dirt) are drawn on the dial before the perspective warp so
they move with the gauge; camera effects (low light, motion blur, radiation
speckle, sensor noise, JPEG) are applied to the whole frame afterwards.
"""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .dial import DialRender, DialSpec, render_dial, sample_spec, sample_value

Point = tuple[float, float]


# ----------------------------------------------------------------- backgrounds

def procedural_background(rng: np.random.Generator, size: int) -> np.ndarray:
    """Painted wall or panel with pipes, brackets and grime (RGB uint8)."""
    wall = [(150, 160, 150), (185, 180, 165), (120, 130, 135), (200, 200, 195),
            (90, 110, 95), (160, 150, 130), (70, 75, 80)]
    base = np.array(wall[rng.integers(len(wall))], dtype=np.float32)
    base = np.clip(base + rng.normal(0, 15, 3), 0, 255)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    g = rng.uniform(-0.25, 0.25) * (xx * math.cos(rng.uniform(0, 6.28)) + yy)
    img = base[None, None] * (1 + g[..., None])

    # Pipes with cylindrical shading.
    for _ in range(rng.integers(0, 5)):
        color = np.array([(180, 180, 185), (60, 120, 70), (190, 170, 60), (90, 90, 95),
                          (170, 60, 50), (210, 210, 210)][rng.integers(6)], dtype=np.float32)
        width = rng.uniform(0.04, 0.2) * size
        horizontal = rng.random() < 0.5
        pos = rng.uniform(0, size)
        coord = (yy if horizontal else xx) * size
        t = (coord - pos) / (width / 2)
        inside = np.abs(t) < 1
        shade = np.sqrt(np.clip(1 - t**2, 0, 1)) * 0.8 + 0.2 + 0.3 * np.exp(-((t + 0.4) / 0.2) ** 2)
        img = np.where(inside[..., None], color[None, None] * shade[..., None], img)

    # Panels / brackets.
    for _ in range(rng.integers(0, 4)):
        x0, y0 = rng.uniform(0, size, 2)
        w, h = rng.uniform(0.05, 0.4, 2) * size
        gray = rng.uniform(40, 220)
        col = np.clip(gray + rng.normal(0, 18, 3), 0, 255)  # mostly desaturated
        img[int(y0):int(y0 + h), int(x0):int(x0 + w)] = col

    # Grime and texture.
    noise = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), size / 40)
    noise /= max(1e-6, np.abs(noise).max())
    img *= (1 + 0.15 * noise[..., None])
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def load_background(rng: np.random.Generator, size: int, bg_paths: list[str],
                    photo_prob: float = 0.7) -> np.ndarray:
    if bg_paths and rng.random() < photo_prob:
        img = cv2.imread(bg_paths[rng.integers(len(bg_paths))], cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            s = rng.uniform(0.3, 1.0) * min(h, w)
            x, y = rng.uniform(0, w - s), rng.uniform(0, h - s)
            crop = img[int(y):int(y + s), int(x):int(x + s)]
            return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return procedural_background(rng, size)


def list_backgrounds(bg_dir: str | None) -> list[str]:
    if not bg_dir or not os.path.isdir(bg_dir):
        return []
    out = []
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        out += glob.glob(os.path.join(bg_dir, "**", f"*.{ext}"), recursive=True)
    return sorted(out)


# ------------------------------------------------------------------ glass layer

def add_glare(rng: np.random.Generator, dial: np.ndarray, center: Point, radius: float) -> None:
    """Specular reflection on the cover glass: a soft ellipse or a window band."""
    h, w = dial.shape[:2]
    mask = np.zeros((h, w), np.float32)
    if rng.random() < 0.5:
        cx, cy = center[0] + rng.uniform(-0.6, 0.6) * radius, center[1] + rng.uniform(-0.6, 0.6) * radius
        axes = (int(radius * rng.uniform(0.15, 0.6)), int(radius * rng.uniform(0.05, 0.3)))
        cv2.ellipse(mask, (int(cx), int(cy)), axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), radius * rng.uniform(0.05, 0.2))
    else:  # band from a window or ceiling light
        ang = rng.uniform(0, math.pi)
        off = rng.uniform(-0.7, 0.7) * radius
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = (xx - center[0]) * math.cos(ang) + (yy - center[1]) * math.sin(ang) - off
        mask = np.exp(-(dist / (radius * rng.uniform(0.05, 0.25))) ** 2)
    disc = np.zeros((h, w), np.float32)
    cv2.circle(disc, (int(center[0]), int(center[1])), int(radius), 1.0, -1)
    a = (mask * disc * rng.uniform(0.5, 1.0))[..., None]
    tint = np.array([255, 255, rng.uniform(220, 255)], np.float32)
    dial[..., :3] = (dial[..., :3] * (1 - a) + tint * a).astype(np.uint8)


def add_dirt(rng: np.random.Generator, dial: np.ndarray, center: Point, radius: float) -> None:
    """Dust, smudges and stains on the glass."""
    h, w = dial.shape[:2]
    field_ = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), radius * 0.08)
    field_ = (field_ - field_.min()) / max(1e-6, np.ptp(field_))
    a = np.clip((field_ - rng.uniform(0.4, 0.7)) * 2.5, 0, 1) * rng.uniform(0.25, 0.6)
    specks = (rng.random((h, w)) < rng.uniform(0.0005, 0.004)).astype(np.float32)
    specks = cv2.GaussianBlur(specks, (0, 0), 1.0) * 4
    a = np.clip(a + specks, 0, 0.9)
    disc = np.zeros((h, w), np.float32)
    cv2.circle(disc, (int(center[0]), int(center[1])), int(radius), 1.0, -1)
    a = (a * disc)[..., None]
    col = np.array([(110, 95, 70), (80, 80, 80), (140, 120, 90)][rng.integers(3)], np.float32)
    dial[..., :3] = (dial[..., :3] * (1 - a) + col * a).astype(np.uint8)


# ----------------------------------------------------------------- perspective

def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def camera_homography(src_size: int, diameter: float, yaw: float, pitch: float, roll: float,
                      focal: float, center_px: Point) -> np.ndarray:
    """Homography from the flat dial image to the camera image.

    The dial (``src_size`` px square, gauge spanning it) is a plane of physical
    width ``diameter`` px at distance ``focal``, rotated by yaw/pitch/roll, and
    its centre projects to ``center_px``.
    """
    R = rotation(yaw, pitch, roll)
    half = src_size / 2.0
    s = diameter / src_size
    src = np.array([[0, 0], [src_size, 0], [src_size, src_size], [0, src_size]], np.float32)
    dst = []
    for x, y in src:
        P = R @ np.array([(x - half) * s, (y - half) * s, 0.0]) + np.array([0, 0, focal])
        dst.append([focal * P[0] / P[2] + center_px[0], focal * P[1] / P[2] + center_px[1]])
    return cv2.getPerspectiveTransform(src, np.array(dst, np.float32))


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# --------------------------------------------------------------- camera effects

def low_light(rng, img: np.ndarray) -> np.ndarray:
    """Dim scene with an optional flashlight spot, then amplified sensor noise."""
    h, w = img.shape[:2]
    f = img.astype(np.float32) / 255.0
    gain = rng.uniform(0.08, 0.4)
    illum = np.full((h, w), gain, np.float32)
    if rng.random() < 0.5:  # torch / helmet lamp
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        sig = rng.uniform(0.2, 0.5) * w
        illum += rng.uniform(0.3, 0.8) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig**2))
    f = f * illum[..., None]
    f = f * np.array([1.0, rng.uniform(0.9, 1.05), rng.uniform(0.7, 1.0)], np.float32)  # warm cast
    # Camera auto-exposure brightens the frame, amplifying shot + read noise.
    boost = min(1 / max(f.mean(), 1e-3) * 0.35, 6.0) if rng.random() < 0.7 else 1.0
    shot = rng.normal(0, 1, f.shape).astype(np.float32) * np.sqrt(np.clip(f, 0, 1) / 300.0)
    read = rng.normal(0, rng.uniform(0.005, 0.02), f.shape).astype(np.float32)
    f = (f + shot + read) * boost
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def motion_blur(rng, img: np.ndarray) -> np.ndarray:
    length = int(rng.integers(5, 26))
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k)


def radiation_speckle(rng, img: np.ndarray) -> np.ndarray:
    """Gamma-ray hits on the sensor: hot pixels, small blobs, short streaks."""
    h, w = img.shape[:2]
    out = img.copy()
    density = rng.uniform(0.0005, 0.01)
    n = int(h * w * density)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    out[ys, xs] = rng.integers(180, 256, (n, 1))
    # salt-and-pepper share
    m = int(n * rng.uniform(0, 0.3))
    out[rng.integers(0, h, m), rng.integers(0, w, m)] = 0
    for _ in range(int(rng.integers(0, max(2, n // 200)))):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        L, a = rng.uniform(2, 12), rng.uniform(0, math.pi)
        cv2.line(out, (int(x0), int(y0)), (int(x0 + L * math.cos(a)), int(y0 + L * math.sin(a))),
                 (255, 255, 255), 1)
    for _ in range(int(rng.integers(0, max(2, n // 100)))):
        cv2.circle(out, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 3)),
                   (255, 255, 255), -1)
    return out


def jpeg(rng, img: np.ndarray, q_lo: int = 35, q_hi: int = 95) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(q_lo, q_hi))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy() if ok else img


# --------------------------------------------------------------------- samples

@dataclass
class Sample:
    image: np.ndarray                 # RGB uint8
    value: float
    spec: DialSpec
    keypoints: dict[str, Point]       # center, tip, scale_min, scale_max (image px)
    bbox: tuple[float, float, float, float]   # x0, y0, x1, y1 of the gauge case
    homography: np.ndarray            # dial image -> camera image
    dial_keypoints: dict[str, Point]  # same points on the flat dial image
    dial_size: int
    conditions: list[str] = field(default_factory=list)
    camera: dict = field(default_factory=dict)


def make_sample(rng: np.random.Generator, cfg: dict, bg_paths: list[str] | None = None) -> Sample:
    size = int(cfg.get("image_size", 640))
    cam = cfg.get("camera", {})
    npp = cfg.get("nuclear_conditions", {})

    spec = sample_spec(rng, cfg)
    value = sample_value(rng, spec)
    dial_size = 512
    r: DialRender = render_dial(spec, value, size=dial_size, rng=rng)
    dial = np.array(r.image)  # RGBA

    conditions = []
    if rng.random() < npp.get("dirt_prob", 0.2):
        add_dirt(rng, dial, r.center, r.face_radius)
        conditions.append("dirt")
    if rng.random() < npp.get("glare_prob", 0.3):
        add_glare(rng, dial, r.center, r.face_radius)
        conditions.append("glare")

    yaw = rng.uniform(*cam.get("yaw_deg", [-50, 50]))
    pitch = rng.uniform(*cam.get("pitch_deg", [-40, 40]))
    roll = rng.uniform(*cam.get("roll_deg", [-15, 15]))
    diameter = rng.uniform(*cam.get("scale", [0.25, 0.9])) * size
    focal = rng.uniform(*cam.get("focal", [0.8, 2.5])) * size

    # Place the gauge so its whole case stays in the frame.
    rim = [(r.center[0] + r.outer_radius * math.cos(t), r.center[1] + r.outer_radius * math.sin(t))
           for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)]
    H0 = camera_homography(dial_size, diameter, yaw, pitch, roll, focal, (0.0, 0.0))
    pr = apply_h(H0, rim)
    lo, hi = pr.min(0), pr.max(0)
    span = hi - lo
    if (span > size * 0.98).any():  # too big after projection: shrink
        k = (size * 0.98) / span.max()
        diameter *= k
        H0 = camera_homography(dial_size, diameter, yaw, pitch, roll, focal, (0.0, 0.0))
        pr = apply_h(H0, rim)
        lo, hi = pr.min(0), pr.max(0)
    cx = rng.uniform(-lo[0] + 1, size - hi[0] - 1)
    cy = rng.uniform(-lo[1] + 1, size - hi[1] - 1)
    H = camera_homography(dial_size, diameter, yaw, pitch, roll, focal, (cx, cy))

    bg = load_background(rng, size, bg_paths or [])

    # Stem / process connection below (or behind) the case.
    if rng.random() < 0.6:
        stem_w = r.outer_radius * rng.uniform(0.15, 0.3)
        stem_pts = [(r.center[0] - stem_w, r.center[1]), (r.center[0] + stem_w, r.center[1]),
                    (r.center[0] + stem_w, dial_size * 1.6), (r.center[0] - stem_w, dial_size * 1.6)]
        sp = apply_h(H, stem_pts).astype(np.int32)
        col = [(180, 160, 90), (170, 170, 175), (120, 120, 120)][rng.integers(3)]
        cv2.fillConvexPoly(bg, sp, col)

    warped = cv2.warpPerspective(dial, H, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # Contact shadow of the case.
    alpha = warped[..., 3:4].astype(np.float32) / 255.0
    sh = cv2.GaussianBlur(alpha[..., 0], (0, 0), max(1.0, diameter * 0.03))
    sh = np.roll(sh, (int(diameter * 0.02), int(diameter * 0.02)), axis=(0, 1))
    img = bg.astype(np.float32) * (1 - 0.45 * sh[..., None])
    img = img * (1 - alpha) + warped[..., :3].astype(np.float32) * alpha

    # Global brightness / contrast jitter (always on, mild).
    img = (img - 128) * rng.uniform(0.8, 1.2) + 128 + rng.uniform(-20, 20)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < npp.get("low_light_prob", 0.3):
        img = low_light(rng, img)
        conditions.append("low_light")
    if rng.random() < npp.get("motion_blur_prob", 0.2):
        img = motion_blur(rng, img)
        conditions.append("motion_blur")
    elif rng.random() < npp.get("defocus_prob", 0.15):
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 2.5))
        conditions.append("defocus")
    if rng.random() < npp.get("radiation_speckle_prob", 0.2):
        img = radiation_speckle(rng, img)
        conditions.append("radiation_speckle")
    img = jpeg(rng, img)

    names = ["center", "tip", "scale_min", "scale_max"]
    dial_kp = dict(zip(names, [r.center, r.tip, r.scale_min, r.scale_max]))
    img_kp = {k: tuple(map(float, v)) for k, v in zip(names, apply_h(H, list(dial_kp.values())))}
    pr = apply_h(H, rim)
    bbox = (float(max(0, pr[:, 0].min())), float(max(0, pr[:, 1].min())),
            float(min(size, pr[:, 0].max())), float(min(size, pr[:, 1].max())))
    return Sample(image=img, value=value, spec=spec, keypoints=img_kp, bbox=bbox, homography=H,
                  dial_keypoints=dial_kp, dial_size=dial_size, conditions=conditions,
                  camera={"yaw": yaw, "pitch": pitch, "roll": roll, "focal": focal,
                          "diameter": diameter})
