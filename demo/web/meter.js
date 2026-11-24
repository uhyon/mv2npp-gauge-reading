// The reading as a semicircular meter (SVG): a port of desktop/meter.py.
export const STATUS_COLORS = { NORMAL: "#22c55e", LOW: "#f59e0b", HIGH: "#ef4444", "": "#38bdf8",
  RETAKE: "#f59e0b" };

export function statusOf(r) {
  return !r || r.value == null ? "RETAKE" : (r.alarm || "");
}

export function meterSvg(r, band, valueText, width = 300) {
  const color = STATUS_COLORS[statusOf(r)];
  const cx = 150, cy = 150, R = 110;
  const pt = (f, rr = R) => {
    const a = Math.PI * (1 - f);
    return [cx + rr * Math.cos(a), cy - rr * Math.sin(a)];
  };
  const arc = (f0, f1, col) => {
    const [x0, y0] = pt(f0), [x1, y1] = pt(f1);
    return `<path d="M${x0.toFixed(1)},${y0.toFixed(1)} A${R},${R} 0 0 1 ${x1.toFixed(1)},${y1.toFixed(1)}" stroke="${col}" stroke-width="18" fill="none"/>`;
  };
  const parts = [arc(0, 1, "#334155")];
  if (band && r && r.scaleMin != null && r.scaleMax !== r.scaleMin) {
    const span = r.scaleMax - r.scaleMin, clip = (v) => Math.min(Math.max(v, 0), 1);
    const lo = band.low == null ? 0 : clip((band.low - r.scaleMin) / span);
    const hi = band.high == null ? 1 : clip((band.high - r.scaleMin) / span);
    if (lo > 0) parts.push(arc(0, lo, "#b45309"));
    parts.push(arc(lo, hi, "#166534"));
    if (hi < 1) parts.push(arc(hi, 1, "#991b1b"));
  }
  for (let i = 0; i <= 10; i++) {
    const [x0, y0] = pt(i / 10, R - 14), [x1, y1] = pt(i / 10, R - (i % 5 ? 19 : 24));
    parts.push(`<line x1="${x0.toFixed(1)}" y1="${y0.toFixed(1)}" x2="${x1.toFixed(1)}" y2="${y1.toFixed(1)}" stroke="#94a3b8" stroke-width="2"/>`);
  }
  if (r && r.fraction != null) {
    const [x, y] = pt(Math.min(Math.max(r.fraction, 0), 1), R - 8);
    parts.push(`<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${color}" stroke-width="5" stroke-linecap="round"/>`);
  }
  parts.push(`<circle cx="${cx}" cy="${cy}" r="9" fill="${color}"/>`);
  if (r && r.scaleMin != null) {
    const fmt = (v) => Number(v.toPrecision(4)).toString();
    parts.push(`<text x="${(pt(0)[0] - 4).toFixed(0)}" y="172" fill="#94a3b8" font-size="13" text-anchor="middle">${fmt(r.scaleMin)}</text>`);
    parts.push(`<text x="${(pt(1)[0] + 4).toFixed(0)}" y="172" fill="#94a3b8" font-size="13" text-anchor="middle">${fmt(r.scaleMax)}</text>`);
  }
  const ok = r && r.value != null;
  const unit = ok ? (r.unit || "") : "";
  parts.push(`<text x="${cx}" y="205" fill="#f8fafc" font-size="34" font-weight="700" text-anchor="middle">${ok ? valueText : (r ? "RETAKE" : "–")} <tspan font-size="18" fill="#cbd5e1">${unit}</tspan></text>`);
  if (r) {
    parts.push(`<rect x="${cx - 48}" y="216" width="96" height="24" rx="12" fill="${color}"/>`
      + `<text x="${cx}" y="233" fill="#0f172a" font-size="13" font-weight="700" text-anchor="middle">${statusOf(r) || "READ"}</text>`);
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 250" width="${width}" height="${Math.round(width * 250 / 300)}" font-family="system-ui, sans-serif" role="img" aria-label="meter">${parts.join("")}</svg>`;
}
