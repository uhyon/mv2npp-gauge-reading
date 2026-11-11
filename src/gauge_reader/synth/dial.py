"""Render a flat, front-facing analog dial and its ground-truth keypoints.

Angles are screen angles in degrees, measured clockwise from +x (image y points
down), the same convention as ``gauge_reader.reading.geometry``. The scale runs
clockwise from ``min_value`` to ``max_value`` and its gap is centred at the
bottom of the dial (6 o'clock), as on nearly all pressure and temperature gauges.
"""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

Point = tuple[float, float]

# Used when the config's fonts_dir is empty or missing.
SYSTEM_FONT_GLOBS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans*.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans*.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans*.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito*.ttf",
    "C:/Windows/Fonts/arial*.ttf",
    "C:/Windows/Fonts/segoeui*.ttf",
    "C:/Windows/Fonts/calibri*.ttf",
    "/System/Library/Fonts/Supplemental/Arial*.ttf",
    "/Library/Fonts/Arial*.ttf",
]


@lru_cache(maxsize=8)
def find_fonts(fonts_dir: str | None = None) -> tuple[str, ...]:
    """TrueType fonts from ``fonts_dir``, else common system fonts. May be empty."""
    paths: list[str] = []
    if fonts_dir and os.path.isdir(fonts_dir):
        for ext in ("ttf", "otf", "TTF", "OTF"):
            paths += glob.glob(os.path.join(fonts_dir, "**", f"*.{ext}"), recursive=True)
    if not paths:
        for pattern in SYSTEM_FONT_GLOBS:
            paths += glob.glob(pattern)
    # Oblique/italic faces are rare on gauges.
    paths = [p for p in paths if not any(k in os.path.basename(p).lower()
                                         for k in ("italic", "oblique"))]
    return tuple(sorted(set(paths)))


@lru_cache(maxsize=256)
def load_font(path: str | None, size: int) -> ImageFont.ImageFont:
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


@dataclass
class DialSpec:
    min_value: float
    max_value: float
    unit: str
    sweep_deg: float
    major_step: float
    minor_per_major: int
    face_color: tuple[int, int, int] = (245, 245, 240)
    ink_color: tuple[int, int, int] = (20, 20, 20)
    needle_color: tuple[int, int, int] = (20, 20, 20)
    bezel_color: tuple[int, int, int] = (170, 170, 175)
    bezel_frac: float = 0.08          # bezel width / outer radius
    tick_radius_frac: float = 0.86    # outer end of ticks / face radius
    label_size_frac: float = 0.11     # label font size / face radius
    needle_style: str = "taper"       # "taper", "line" or "arrow"
    needle_len_frac: float = 0.95     # needle tip radius / tick radius
    needle_tail_frac: float = 0.2     # tail behind the centre / face radius
    font_path: str | None = None
    red_zone: tuple[float, float] | None = None
    green_zone: tuple[float, float] | None = None
    brand_text: str | None = None
    second_scale: tuple[float, str] | None = None   # (factor, unit), inner scale
    labels_outside_ticks: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def start_deg(self) -> float:
        """Screen angle of the minimum tick (gap centred at 6 o'clock = 90 deg)."""
        return 90.0 + (360.0 - self.sweep_deg) / 2.0

    def value_to_angle(self, value: float) -> float:
        frac = (value - self.min_value) / (self.max_value - self.min_value)
        return self.start_deg + frac * self.sweep_deg

    def major_values(self) -> list[float]:
        n = round((self.max_value - self.min_value) / self.major_step)
        return [self.min_value + i * self.major_step for i in range(n + 1)]


def polar(center: Point, radius: float, angle_deg: float) -> Point:
    t = math.radians(angle_deg)
    return (center[0] + radius * math.cos(t), center[1] + radius * math.sin(t))


def _decimals(step: float) -> int:
    for d in range(4):
        if abs(step * 10**d - round(step * 10**d)) < 1e-9:
            return d
    return 3


def format_label(value: float, step: float) -> str:
    d = _decimals(step)
    text = f"{value:.{d}f}"
    if text.startswith("-") and float(text) == 0:
        text = text[1:]
    return text


NICE_STEPS = [1, 2, 2.5, 5]


def nice_major_steps(lo: float, hi: float, min_n: int = 4, max_n: int = 10) -> list[float]:
    """Steps that divide [lo, hi] into min_n..max_n equal major intervals."""
    span = hi - lo
    out = []
    for k in range(-3, 5):
        for m in NICE_STEPS:
            step = m * 10.0**k
            n = span / step
            if min_n <= round(n) <= max_n and abs(n - round(n)) < 1e-6 \
                    and abs(lo / step - round(lo / step)) < 1e-6:
                out.append(step)
    return out


@dataclass
class DialRender:
    image: Image.Image          # RGBA, transparent outside the gauge case
    center: Point
    tip: Point
    scale_min: Point
    scale_max: Point
    face_radius: float          # radius of the printed dial face
    outer_radius: float         # radius including the bezel


def _vertical_gradient_disc(size: int, c0, c1) -> Image.Image:
    ramp = np.linspace(0.0, 1.0, size)[:, None, None]
    arr = (np.array(c0)[None, None] * (1 - ramp) + np.array(c1)[None, None] * ramp)
    arr = np.repeat(arr, size, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def render_dial(spec: DialSpec, value: float, size: int = 512, ss: int = 2,
                rng: np.random.Generator | None = None) -> DialRender:
    """Draw the gauge at ``size`` x ``size`` (supersampled by ``ss``).

    Keypoints follow the project convention: ``scale_min``/``scale_max`` are the
    outer ends of the first and last major ticks, ``tip`` is the needle tip.
    """
    rng = rng or np.random.default_rng()
    S = size * ss
    c = (S / 2.0, S / 2.0)
    R_out = S / 2.0 - 1
    R_face = R_out * (1.0 - spec.bezel_frac)
    R_tick = R_face * spec.tick_radius_frac

    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Case and bezel: metallic vertical gradient ring with a darker inner lip.
    b = np.array(spec.bezel_color, dtype=float)
    hi_c = tuple(np.clip(b * 1.35 + 20, 0, 255).astype(int))
    lo_c = tuple(np.clip(b * 0.55, 0, 255).astype(int))
    bezel = _vertical_gradient_disc(S, hi_c, lo_c)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse([c[0] - R_out, c[1] - R_out, c[0] + R_out, c[1] + R_out], fill=255)
    img.paste(bezel, (0, 0), mask)
    lip = R_face + max(2, 0.012 * S)
    d.ellipse([c[0] - lip, c[1] - lip, c[0] + lip, c[1] + lip], fill=lo_c + (255,))

    # Face with a faint radial shade.
    d.ellipse([c[0] - R_face, c[1] - R_face, c[0] + R_face, c[1] + R_face],
              fill=spec.face_color + (255,))

    ink = spec.ink_color + (255,)
    a0 = spec.start_deg

    # Coloured zones (drawn under the ticks).
    def zone(lo, hi, color):
        w = R_face * 0.06
        r = R_tick - w / 2
        box = [c[0] - r, c[1] - r, c[0] + r, c[1] + r]
        d.arc(box, spec.value_to_angle(lo), spec.value_to_angle(hi), fill=color, width=int(w))

    if spec.green_zone:
        zone(*spec.green_zone, (40, 160, 70, 255))
    if spec.red_zone:
        zone(*spec.red_zone, (210, 35, 35, 255))

    # Scale arc line.
    if spec.extra.get("arc_line", True):
        box = [c[0] - R_tick, c[1] - R_tick, c[0] + R_tick, c[1] + R_tick]
        d.arc(box, a0, a0 + spec.sweep_deg, fill=ink, width=max(1, int(0.006 * S)))

    majors = spec.major_values()
    major_len = R_face * spec.extra.get("major_len", 0.12)
    minor_len = major_len * 0.55
    major_w = max(2, int(S * spec.extra.get("major_w", 0.008)))
    minor_w = max(1, int(major_w * 0.5))
    for i, v in enumerate(majors):
        ang = spec.value_to_angle(v)
        d.line([polar(c, R_tick, ang), polar(c, R_tick - major_len, ang)], fill=ink, width=major_w)
        if i < len(majors) - 1:
            for j in range(1, spec.minor_per_major):
                mv = v + spec.major_step * j / spec.minor_per_major
                ma = spec.value_to_angle(mv)
                ln = minor_len * (1.35 if spec.minor_per_major % 2 == 0
                                  and j == spec.minor_per_major // 2 else 1.0)
                d.line([polar(c, R_tick, ma), polar(c, R_tick - ln, ma)], fill=ink, width=minor_w)

    # Major labels.
    fsize = max(8, int(R_face * spec.label_size_frac))
    font = load_font(spec.font_path, fsize)
    r_lab = (R_tick + fsize * 0.75) if spec.labels_outside_ticks \
        else (R_tick - major_len - fsize * 0.85)
    for v in majors:
        ang = spec.value_to_angle(v)
        d.text(polar(c, r_lab, ang), format_label(v, spec.major_step), fill=ink, font=font,
               anchor="mm")

    # Optional inner second scale (e.g. psi under bar). Labels only, plus short ticks.
    if spec.second_scale:
        factor, unit2 = spec.second_scale
        lo2, hi2 = spec.min_value * factor, spec.max_value * factor
        span2 = hi2 - lo2
        steps = [m * 10.0**k for k in range(-3, 6) for m in NICE_STEPS
                 if 4 <= span2 / (m * 10.0**k) <= 10]
        if steps:
            step2 = steps[len(steps) // 2]
            r2 = r_lab - fsize * 1.35
            f2 = load_font(spec.font_path, int(fsize * 0.7))
            col2 = spec.extra.get("second_color", (200, 30, 30))
            v2 = math.ceil(lo2 / step2 - 1e-9) * step2
            while v2 <= hi2 + 1e-9:
                ang = spec.value_to_angle(v2 / factor)
                d.line([polar(c, r2 + fsize * 0.45, ang), polar(c, r2 + fsize * 0.2, ang)],
                       fill=col2 + (255,), width=minor_w)
                d.text(polar(c, r2 - fsize * 0.1, ang), format_label(v2, step2), fill=col2 + (255,),
                       font=f2, anchor="mm")
                v2 += step2
            d.text((c[0], c[1] + R_face * 0.62), unit2, fill=col2 + (255,), font=f2, anchor="mm")

    # Unit and brand.
    ufont = load_font(spec.font_path, int(fsize * rng.uniform(0.9, 1.3)))
    unit_y = c[1] + R_face * (0.42 if spec.sweep_deg > 200 else -0.35)
    d.text((c[0], unit_y), spec.unit, fill=ink, font=ufont, anchor="mm")
    if spec.brand_text:
        bfont = load_font(spec.font_path, int(fsize * 0.6))
        by = c[1] - R_face * 0.35 if spec.sweep_deg > 200 else c[1] + R_face * 0.3
        d.text((c[0], by), spec.brand_text, fill=ink, font=bfont, anchor="mm")

    # Needle, with a soft shadow on its own layer.
    ang = spec.value_to_angle(value)
    r_tip = R_tick * spec.needle_len_frac
    tip = polar(c, r_tip, ang)
    tail = polar(c, R_face * spec.needle_tail_frac, ang + 180)
    w_base = R_face * rng.uniform(0.025, 0.05)

    def needle_poly(offset=(0.0, 0.0)):
        ox, oy = offset
        px, py = math.cos(math.radians(ang + 90)), math.sin(math.radians(ang + 90))
        if spec.needle_style == "line":
            w = w_base * 0.35
            pts = [(tail[0] + px * w, tail[1] + py * w), (tip[0] + px * w, tip[1] + py * w),
                   (tip[0] - px * w, tip[1] - py * w), (tail[0] - px * w, tail[1] - py * w)]
        elif spec.needle_style == "arrow":
            w = w_base * 0.3
            head = polar(c, r_tip - R_face * 0.1, ang)
            pts = [(tail[0] + px * w, tail[1] + py * w), (head[0] + px * w, head[1] + py * w),
                   (head[0] + px * w * 3, head[1] + py * w * 3), tip,
                   (head[0] - px * w * 3, head[1] - py * w * 3), (head[0] - px * w, head[1] - py * w),
                   (tail[0] - px * w, tail[1] - py * w)]
        else:  # taper
            pts = [(tail[0] + px * w_base * 0.7, tail[1] + py * w_base * 0.7),
                   (c[0] + px * w_base, c[1] + py * w_base), tip,
                   (c[0] - px * w_base, c[1] - py * w_base),
                   (tail[0] - px * w_base * 0.7, tail[1] - py * w_base * 0.7)]
        return [(x + ox, y + oy) for x, y in pts]

    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    so = R_face * 0.025
    ImageDraw.Draw(shadow).polygon(needle_poly((so, so)), fill=(0, 0, 0, 90))
    shadow = shadow.filter(ImageFilter.GaussianBlur(S * 0.006))
    face_mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(face_mask).ellipse([c[0] - R_face, c[1] - R_face, c[0] + R_face, c[1] + R_face],
                                      fill=255)
    shadow.putalpha(Image.fromarray(np.minimum(np.array(shadow.getchannel("A")),
                                               np.array(face_mask))))
    img = Image.alpha_composite(img, shadow)
    d = ImageDraw.Draw(img)
    d.polygon(needle_poly(), fill=spec.needle_color + (255,))
    hub = R_face * rng.uniform(0.05, 0.1)
    d.ellipse([c[0] - hub, c[1] - hub, c[0] + hub, c[1] + hub], fill=spec.needle_color + (255,))
    hub2 = hub * 0.45
    hc = tuple(int(x) for x in np.clip(np.array(spec.bezel_color) * 1.1, 0, 255)) + (255,)
    d.ellipse([c[0] - hub2, c[1] - hub2, c[0] + hub2, c[1] + hub2], fill=hc)

    img = img.resize((size, size), Image.LANCZOS)
    k = 1.0 / ss
    sc = lambda p: (p[0] * k, p[1] * k)
    return DialRender(
        image=img,
        center=sc(c),
        tip=sc(tip),
        scale_min=sc(polar(c, R_tick, spec.value_to_angle(spec.min_value))),
        scale_max=sc(polar(c, R_tick, spec.value_to_angle(spec.max_value))),
        face_radius=R_face * k,
        outer_radius=R_out * k,
    )


# ---------------------------------------------------------------- random specs

FACE_PALETTE = [
    ((245, 245, 240), (20, 20, 20)),     # white / black (most common)
    ((250, 250, 250), (15, 15, 15)),
    ((235, 230, 215), (30, 25, 20)),     # aged cream
    ((25, 25, 28), (235, 235, 235)),     # black face / white ink
    ((240, 240, 235), (20, 40, 110)),    # blue ink
]
NEEDLE_COLORS = [(15, 15, 15), (200, 25, 25), (20, 20, 20), (30, 30, 120)]
BEZEL_COLORS = [(175, 175, 180), (150, 150, 155), (40, 40, 40), (190, 160, 90), (120, 120, 125)]
BRANDS = [None, None, "WIKA-TYPE", "GAUGE CO.", "KS B 5305", "CL 1.6", "EN 837-1", "CLASS 1.0",
          "PRESSURE", "TEMP", "EN 13190"]
SECOND_SCALES = {"bar": (14.5038, "psi"), "kgf/cm2": (14.2233, "psi"), "MPa": (145.038, "psi"),
                 "psi": (0.0689476, "bar")}


def sample_spec(rng: np.random.Generator, cfg: dict) -> DialSpec:
    """Random dial design from the ``dial`` section of the generator config."""
    dial = cfg.get("dial", {})
    lo, hi = map(float, dial.get("ranges", [[0, 10]])[rng.integers(len(dial.get("ranges", [[0, 10]])))])
    units = dial.get("units", ["bar"])
    unit = units[rng.integers(len(units))]
    smin, smax = dial.get("sweep_deg", [180, 270])
    sweep = float(rng.choice([smin, smax, 240.0, rng.uniform(smin, smax)], p=[0.15, 0.55, 0.15, 0.15]))
    n_lo, n_hi = dial.get("major_ticks", [5, 11])
    steps = nice_major_steps(lo, hi, n_lo - 1, n_hi - 1) or [(hi - lo) / 5]
    step = float(rng.choice(steps))
    n_major = round((hi - lo) / step)
    minor_opts = [m for m in (2, 4, 5, 10) if n_major * m <= 100]
    minor = int(rng.choice(minor_opts or [2]))

    fonts = find_fonts(cfg.get("fonts_dir") or dial.get("fonts_dir"))
    font = str(rng.choice(fonts)) if fonts else None
    face, ink = FACE_PALETTE[rng.choice(len(FACE_PALETTE), p=[0.4, 0.2, 0.15, 0.15, 0.1])]
    needle = ink if face[0] < 100 else NEEDLE_COLORS[rng.integers(len(NEEDLE_COLORS))]

    red = green = None
    if rng.random() < 0.3:
        red = (lo + (hi - lo) * rng.uniform(0.7, 0.9), hi)
    if rng.random() < 0.15:
        green = (lo + (hi - lo) * rng.uniform(0.2, 0.4), lo + (hi - lo) * rng.uniform(0.5, 0.7))
    second = SECOND_SCALES.get(unit) if rng.random() < dial.get("second_scale_prob", 0.15) else None

    return DialSpec(
        min_value=lo, max_value=hi, unit=unit, sweep_deg=sweep, major_step=step,
        minor_per_major=minor, face_color=face, ink_color=ink, needle_color=needle,
        bezel_color=BEZEL_COLORS[rng.integers(len(BEZEL_COLORS))],
        bezel_frac=float(rng.uniform(0.04, 0.14)),
        tick_radius_frac=float(rng.uniform(0.82, 0.92)),
        label_size_frac=float(rng.uniform(0.08, 0.14)),
        needle_style=str(rng.choice(["taper", "line", "arrow"], p=[0.6, 0.25, 0.15])),
        needle_len_frac=float(rng.uniform(0.85, 1.0)),
        needle_tail_frac=float(rng.uniform(0.0, 0.25)),
        font_path=font, red_zone=red, green_zone=green,
        brand_text=BRANDS[rng.integers(len(BRANDS))], second_scale=second,
        labels_outside_ticks=bool(rng.random() < 0.15 and second is None),
        extra={"arc_line": bool(rng.random() < 0.6), "major_len": float(rng.uniform(0.08, 0.16)),
               "major_w": float(rng.uniform(0.005, 0.011))},
    )


def sample_value(rng: np.random.Generator, spec: DialSpec, rest_prob: float = 0.05) -> float:
    """Uniform reading; sometimes exactly at the minimum (needle on its stop pin)."""
    if rng.random() < rest_prob:
        return spec.min_value
    return float(rng.uniform(spec.min_value, spec.max_value))
