"""The reading as a semicircular meter (SVG): needle, scale ends, alarm band, status badge."""

from __future__ import annotations

import math

STATUS_COLORS = {"NORMAL": "#22c55e", "LOW": "#f59e0b", "HIGH": "#ef4444", "": "#38bdf8",
                 "RETAKE": "#f59e0b"}


def status_of(r) -> str:
    """NORMAL / LOW / HIGH, "" (read, no alarm band) or RETAKE."""
    return "RETAKE" if r.value is None else (r.alarm or "")


def meter_svg(r, band=None, width: int = 300) -> str:
    """``r``: a ``Reading``; ``band``: the instrument's ``AlarmBand`` or None."""
    color = STATUS_COLORS[status_of(r)]
    cx, cy, rad = 150, 150, 110

    def pt(f, rr=rad):
        a = math.pi * (1 - f)
        return cx + rr * math.cos(a), cy - rr * math.sin(a)

    def arc(f0, f1, col):
        (x0, y0), (x1, y1) = pt(f0), pt(f1)
        return (f'<path d="M{x0:.1f},{y0:.1f} A{rad},{rad} 0 0 1 {x1:.1f},{y1:.1f}" '
                f'stroke="{col}" stroke-width="18" fill="none"/>')

    parts = [arc(0, 1, "#334155")]
    if band is not None and r.scale_min is not None and r.scale_max != r.scale_min:
        span = r.scale_max - r.scale_min
        lo = 0.0 if band.low is None else min(max((band.low - r.scale_min) / span, 0), 1)
        hi = 1.0 if band.high is None else min(max((band.high - r.scale_min) / span, 0), 1)
        if lo > 0:
            parts.append(arc(0, lo, "#b45309"))
        parts.append(arc(lo, hi, "#166534"))
        if hi < 1:
            parts.append(arc(hi, 1, "#991b1b"))
    for i in range(11):
        (x0, y0), (x1, y1) = pt(i / 10, rad - 14), pt(i / 10, rad - (24 if i % 5 == 0 else 19))
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                     'stroke="#94a3b8" stroke-width="2"/>')
    if r.fraction is not None:
        x, y = pt(min(max(r.fraction, 0.0), 1.0), rad - 8)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{color}" '
                     'stroke-width="5" stroke-linecap="round"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="9" fill="{color}"/>')
    if r.scale_min is not None:
        for f, v in ((0, r.scale_min), (1, r.scale_max)):
            x = pt(f)[0] + (-4 if f == 0 else 4)
            parts.append(f'<text x="{x:.0f}" y="172" fill="#94a3b8" font-size="13" '
                         f'text-anchor="middle">{v:g}</text>')
    value = "RETAKE" if r.value is None else f"{r.value:.3g}"
    unit = "" if r.value is None else (r.unit or "")
    parts.append(f'<text x="{cx}" y="205" fill="#f8fafc" font-size="34" font-weight="700" '
                 f'text-anchor="middle">{value} <tspan font-size="18" fill="#cbd5e1">{unit}'
                 '</tspan></text>')
    label = status_of(r) or "READ"
    parts.append(f'<rect x="{cx - 48}" y="216" width="96" height="24" rx="12" fill="{color}"/>'
                 f'<text x="{cx}" y="233" fill="#0f172a" font-size="13" font-weight="700" '
                 f'text-anchor="middle">{label}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 250" width="{width}" '
            f'height="{round(width * 250 / 300)}" font-family="sans-serif">{"".join(parts)}</svg>')
