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

def _over(layer: np.ndarray, a: np.ndarray, col: np.ndarray) -> None:
    """Composite colour ``col`` with alpha ``a`` (HxW) over a float RGBA ``layer`` in place."""
    a = a[..., None]
    la = layer[..., 3:4]
    out_a = a + la * (1 - a)
    rgb = (col * a + layer[..., :3] * la * (1 - a)) / np.maximum(out_a, 1e-6)
    layer[..., :3] = rgb
    layer[..., 3:4] = out_a


def add_glare(rng: np.random.Generator, glass: np.ndarray, center: Point, radius: float) -> None:
    """Specular reflection on the cover glass: a soft ellipse or a window band.

    ``glass`` is a float RGBA layer (alpha 0-1) the size of the flat dial image.
    """
    h, w = glass.shape[:2]
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
    a = mask * disc * rng.uniform(0.5, 1.0)
    tint = np.array([255, 255, rng.uniform(220, 255)], np.float32)
    _over(glass, a, tint)


def add_dirt(rng: np.random.Generator, glass: np.ndarray, center: Point, radius: float) -> None:
    """Dust, smudges and stains on the glass (float RGBA layer, see ``add_glare``)."""
    h, w = glass.shape[:2]
    field_ = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), radius * 0.08)
    field_ = (field_ - field_.min()) / max(1e-6, np.ptp(field_))
    a = np.clip((field_ - rng.uniform(0.4, 0.7)) * 2.5, 0, 1) * rng.uniform(0.25, 0.6)
    specks = (rng.random((h, w)) < rng.uniform(0.0005, 0.004)).astype(np.float32)
    specks = cv2.GaussianBlur(specks, (0, 0), 1.0) * 4
    a = np.clip(a + specks, 0, 0.9)
    disc = np.zeros((h, w), np.float32)
    cv2.circle(disc, (int(center[0]), int(center[1])), int(radius), 1.0, -1)
    col = np.array([(110, 95, 70), (80, 80, 80), (140, 120, 90)][rng.integers(3)], np.float32)
    _over(glass, a * disc, col)


# ----------------------------------------------------------------- perspective

def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project(R: np.ndarray, focal: float, center_px: Point, pts3) -> np.ndarray:
    """Pinhole projection of gauge-frame points (x right, y down, z into the wall).

    The gauge frame is centred on the dial, in the same pixel units as ``focal``;
    the gauge front (bezel and glass) is the plane z = 0.
    """
    P = np.asarray(pts3, np.float64).reshape(-1, 3) @ R.T + np.array([0.0, 0.0, focal])
    return focal * P[:, :2] / P[:, 2:3] + np.asarray(center_px, np.float64)


def camera_homography(src_size: int, diameter: float, yaw: float, pitch: float, roll: float,
                      focal: float, center_px: Point, z: float = 0.0) -> np.ndarray:
    """Homography from the flat dial image to the camera image.

    The dial (``src_size`` px square, gauge spanning it) is a plane of physical
    width ``diameter`` px at distance ``focal``, rotated by yaw/pitch/roll, and
    the gauge front centre projects to ``center_px``. ``z`` moves the plane
    behind the front (e.g. the recessed face under the glass).
    """
    R = rotation(yaw, pitch, roll)
    half = src_size / 2.0
    s = diameter / src_size
    src = np.array([[0, 0], [src_size, 0], [src_size, src_size], [0, src_size]], np.float32)
    pts3 = [[(x - half) * s, (y - half) * s, z] for x, y in src]
    dst = project(R, focal, center_px, pts3)
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32))


def draw_case_wall(size: int, R: np.ndarray, focal: float, center_px: Point, radius: float,
                   depth: float, color, rng: np.random.Generator, n: int = 96,
                   ss: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Render the cylindrical side wall of the case (radius/depth in gauge-frame px).

    Only wall segments facing the camera are drawn, far ones first, shaded by a
    light from above. Drawn ``ss`` times larger and downsampled, so edges are
    smooth and segments leave no seams. Returns (RGB float32, alpha float32).
    """
    S = size * ss
    rgb = np.zeros((S, S, 3), np.float32)
    mask = np.zeros((S, S), np.float32)
    if depth > 0:
        t = np.linspace(0, 2 * math.pi, n + 1)
        ring = np.stack([radius * np.cos(t), radius * np.sin(t)], 1)
        cpx = (center_px[0] * ss, center_px[1] * ss)
        front = project(R, focal * ss, cpx, np.c_[ring * ss, np.zeros(n + 1)])
        back = project(R, focal * ss, cpx, np.c_[ring * ss, np.full(n + 1, depth * ss)])
        light = np.array([rng.uniform(-0.5, 0.5), -1.0, -0.6])
        light /= np.linalg.norm(light)
        base = np.asarray(color, np.float32)
        quads = []
        for i in range(n):
            tm = (t[i] + t[i + 1]) / 2
            normal = R @ np.array([math.cos(tm), math.sin(tm), 0.0])
            mid = R @ np.array([radius * math.cos(tm), radius * math.sin(tm), depth / 2])
            mid = mid + np.array([0.0, 0.0, focal])
            if normal @ mid >= 0:          # facing away from the camera
                continue
            lit = max(0.0, -float(normal @ light))
            col = np.clip(base * (0.35 + 0.75 * lit) + 255 * 0.4 * lit ** 12, 0, 255)
            poly = np.array([front[i], front[i + 1], back[i + 1], back[i]])
            quads.append((mid[2], poly, col))
        for _, poly, col in sorted(quads, key=lambda q: -q[0]):
            pts = np.round(poly).astype(np.int32)
            cv2.fillConvexPoly(rgb, pts, tuple(float(c) for c in col))
            cv2.fillConvexPoly(mask, pts, 1.0)
    if ss > 1:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    return rgb, mask


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


def _disc(size: int, center: Point, radius: float, ss: int = 4) -> np.ndarray:
    """Anti-aliased filled disc mask (float32 0-1)."""
    m = np.zeros((size * ss, size * ss), np.float32)
    cv2.circle(m, (round(center[0] * ss), round(center[1] * ss)), round(radius * ss), 1.0, -1)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)


def _warp(layer: np.ndarray, H: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpPerspective(layer, H, (size, size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def make_sample(rng: np.random.Generator, cfg: dict, bg_paths: list[str] | None = None) -> Sample:
    """One labelled image.

    The gauge is a short cylinder: bezel and glass at the front (z = 0), the
    printed face recessed behind the glass (z = recess) and the case wall going
    back to z = depth. At oblique views the wall shows and the face shifts
    against the bezel (parallax). Keypoints and the homography refer to the face.
    """
    size = int(cfg.get("image_size", 640))
    cam = cfg.get("camera", {})
    npp = cfg.get("nuclear_conditions", {})
    case = cfg.get("case", {})

    spec = sample_spec(rng, cfg)
    value = sample_value(rng, spec)
    dial_size = 512
    r: DialRender = render_dial(spec, value, size=dial_size, rng=rng)
    dial = np.array(r.image).astype(np.float32)  # RGBA
    dial[..., 3] /= 255.0

    conditions = []
    glass = np.zeros((dial_size, dial_size, 4), np.float32)
    if rng.random() < npp.get("dirt_prob", 0.2):
        add_dirt(rng, glass, r.center, r.face_radius)
        conditions.append("dirt")
    if rng.random() < npp.get("glare_prob", 0.3):
        add_glare(rng, glass, r.center, r.face_radius)
        conditions.append("glare")

    yaw = rng.uniform(*cam.get("yaw_deg", [-50, 50]))
    pitch = rng.uniform(*cam.get("pitch_deg", [-40, 40]))
    roll = rng.uniform(*cam.get("roll_deg", [-15, 15]))
    diameter = rng.uniform(*cam.get("scale", [0.25, 0.9])) * size
    focal = rng.uniform(*cam.get("focal", [0.8, 2.5])) * size
    depth_frac = rng.uniform(*case.get("depth", [0.15, 0.5]))      # / gauge diameter
    recess_frac = rng.uniform(*case.get("recess", [0.02, 0.08]))   # / gauge diameter
    R = rotation(yaw, pitch, roll)

    # Place the gauge so its whole case (front rim and back rim) stays in the frame.
    t = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    ring = np.stack([np.cos(t), np.sin(t)], 1) * 0.5  # unit-diameter circle

    def rims(d: float, c: Point) -> np.ndarray:
        rr = ring * d * (r.outer_radius * 2 / dial_size)
        return np.r_[project(R, focal, c, np.c_[rr, np.zeros(len(t))]),
                     project(R, focal, c, np.c_[rr, np.full(len(t), depth_frac * d)])]

    pr = rims(diameter, (0.0, 0.0))
    span = pr.max(0) - pr.min(0)
    if (span > size * 0.98).any():  # too big after projection: shrink
        diameter *= (size * 0.98) / span.max()
        pr = rims(diameter, (0.0, 0.0))
    lo, hi = pr.min(0), pr.max(0)
    cx = rng.uniform(-lo[0] + 1, size - hi[0] - 1)
    cy = rng.uniform(-lo[1] + 1, size - hi[1] - 1)
    depth, recess = depth_frac * diameter, recess_frac * diameter
    H_front = camera_homography(dial_size, diameter, yaw, pitch, roll, focal, (cx, cy))
    H = camera_homography(dial_size, diameter, yaw, pitch, roll, focal, (cx, cy), z=recess)
    H_mid = camera_homography(dial_size, diameter, yaw, pitch, roll, focal, (cx, cy), z=depth / 2)

    bg = load_background(rng, size, bg_paths or [])

    # Stem / process connection below (or behind) the case.
    if rng.random() < 0.6:
        stem_w = r.outer_radius * rng.uniform(0.15, 0.3)
        stem_pts = [(r.center[0] - stem_w, r.center[1]), (r.center[0] + stem_w, r.center[1]),
                    (r.center[0] + stem_w, dial_size * 1.6), (r.center[0] - stem_w, dial_size * 1.6)]
        sp = apply_h(H_mid, stem_pts).astype(np.int32)
        col = [(180, 160, 90), (170, 170, 175), (120, 120, 120)][rng.integers(3)]
        cv2.fillConvexPoly(bg, sp, col)

    # Split the flat render into the face (recessed) and the bezel (front).
    face_m = _disc(dial_size, r.center, r.face_radius)
    face = dial.copy()
    face[..., 3] *= face_m
    bezel = dial.copy()
    bezel[..., 3] *= 1 - face_m
    lip_col = np.clip(np.array(spec.bezel_color, np.float32) * 0.35, 0, 255)
    lip = np.dstack([np.broadcast_to(lip_col, (dial_size, dial_size, 3)), face_m * dial[..., 3]])

    phys_r = r.outer_radius * diameter / dial_size
    wall_col = np.array(spec.bezel_color, np.float32) * rng.uniform(0.6, 0.95)
    wall_rgb, wall_a = draw_case_wall(size, R, focal, (cx, cy), phys_r, depth, wall_col, rng)
    w_bezel, w_lip = (_warp(x, H_front, size) for x in (bezel, lip))
    w_face = _warp(face, H, size)
    w_glass = _warp(glass, H_front, size)

    # Contact shadow of the case.
    front_a = np.maximum(w_bezel[..., 3], w_lip[..., 3])
    alpha = np.maximum(front_a, wall_a)
    sh = cv2.GaussianBlur(alpha, (0, 0), max(1.0, diameter * 0.03))
    sh = np.roll(sh, (int(diameter * 0.02), int(diameter * 0.02)), axis=(0, 1))
    img = bg.astype(np.float32) * (1 - 0.45 * sh[..., None])

    def over(rgb, a):
        nonlocal img
        img = img * (1 - a[..., None]) + rgb * a[..., None]

    over(wall_rgb, wall_a)
    over(w_lip[..., :3], w_lip[..., 3])              # inner wall seen through the opening
    over(w_face[..., :3], w_face[..., 3] * w_lip[..., 3])  # face only inside the opening
    over(w_bezel[..., :3], w_bezel[..., 3])
    over(w_glass[..., :3], w_glass[..., 3])

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
    pr = rims(diameter, (cx, cy))
    bbox = (float(max(0, pr[:, 0].min())), float(max(0, pr[:, 1].min())),
            float(min(size, pr[:, 0].max())), float(min(size, pr[:, 1].max())))
    return Sample(image=img, value=value, spec=spec, keypoints=img_kp, bbox=bbox, homography=H,
                  dial_keypoints=dial_kp, dial_size=dial_size, conditions=conditions,
                  camera={"yaw": yaw, "pitch": pitch, "roll": roll, "focal": focal,
                          "diameter": diameter, "case_depth": depth, "face_recess": recess})
