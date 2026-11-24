// Scale range and unit from the numbers printed on the dial: a port of
// gauge_reader.reading.scale_ocr.read_scale (and units.norm_unit) for the browser demo.
// Each OCR box holding a number is placed on the scale by its angle around the centre (on the
// dial face when there is a dial fit); a RANSAC line value = a + b * fraction through the
// labels gives the range at fraction 0 and 1, snapped to a printed label or the label grid.
import { clockwiseAngle, inv3, project } from "./geometry.js";

const RADIUS_RANGE = [0.3, 1.4];
const FRACTION_RANGE = [-0.15, 1.15];
const INLIER_TOL = 0.04;
const MIN_INLIERS = 3;
const SNAP_TOL = 0.03;
const TRUNCATED_UNITS = { "kg/cm": "kgf/cm2", "kgf/cm": "kgf/cm2", "kg/cm.": "kgf/cm2" };

const UNIT_KIND = new Set([
  "bar", "mbar", "millibar", "kbar", "pa", "pascal", "hpa", "hectopascal", "kpa", "kilopascal",
  "mpa", "megapascal", "psi", "psig", "psia", "lbf/in2", "lb/in2", "poundpersquareinch",
  "poundspersquareinch", "kgf/cm2", "kg/cm2", "kgf/cm^2", "kg/cm^2",
  "kilogram-forcepersquarecentimeter", "kilogramforcepersquarecentimeter", "atm", "atmosphere",
  "torr", "mmhg", "millimeterofmercury", "millimetresofmercury", "millimetersofmercury",
  "millimetreofmercury", "inhg", "inchofmercury", "inchesofmercury", "mmh2o", "mmaq", "mmwc",
  "inh2o", "inchesofwater", "inchofwater", "inwc", "inwg", "inchesh2o", "inchh2o", "\"h2o",
  "inchesofwatercolumn", "cmh2o", "mh2o",
  "°c", "℃", "degc", "c", "celsius", "degreecelsius", "degreescelsius",
  "°f", "℉", "degf", "f", "fahrenheit", "degreefahrenheit", "degreesfahrenheit",
]);
const ALIASES = { magepascals: "megapascal", megapascasl: "megapascal" };

export function normUnit(u) {
  let s = u.trim().toLowerCase().replace(/²/g, "2").replace(/º/g, "°").replace(/˚/g, "°");
  s = s.replace(/[\s_.]+/g, "").replace(/degrees/g, "degree");
  s = ALIASES[s] || s;
  if (!UNIT_KIND.has(s) && s.endsWith("s") && UNIT_KIND.has(s.slice(0, -1))) s = s.slice(0, -1);
  return s;
}

const NUM_RE = /^([-+]?\d+(?:[.,]\d+)?)\s*([^\d\s].*)?$/su;

// Leading number and the rest (a possible unit): "0.1MPa" -> [0.1, "MPa"].
export function parseNumber(text) {
  let t = text.trim().replace(/ /g, "");
  if (/\d/.test(t)) t = t.replace(/(?<=[\d.])[Oo]|[Oo](?=[\d.])/g, "0");
  const m = t.match(NUM_RE);
  if (!m) return [null, ""];
  let num = m[1].replace(",", ".");
  if (/^0\d+$/.test(num)) num = "0." + num.slice(1);
  const v = Number(num);
  return Number.isFinite(v) ? [v, m[2] || ""] : [null, ""];
}

function findUnitStrict(texts) {
  for (const text of texts) {
    for (const tok of [text, ...text.split(/[\s()[\]]+/)]) {
      const [, rest] = parseNumber(tok);
      for (const cand of [tok, rest]) {
        if (!cand) continue;
        let n = normUnit(cand.replaceAll("°℃", "℃").replaceAll("°℉", "℉"));
        n = TRUNCATED_UNITS[n] || n;
        if (n === "pa") continue;
        if (UNIT_KIND.has(n) && (n.length > 1 || /[°℃℉]/.test(cand))) {
          return TRUNCATED_UNITS[normUnit(cand)] || cand.trim().replaceAll("°℃", "℃");
        }
      }
    }
  }
  return "";
}

export function findUnit(texts) {
  const u = findUnitStrict(texts);
  if (u) return u;
  for (const text of texts) {
    const t = text.trim().replace(/[°º]/g, "");
    if (t === "C" || t === "℃") return "°C";
    if (t === "F" || t === "℉") return "°F";
  }
  return "";
}

const centerOf = (poly) => [poly.reduce((a, p) => a + p[0], 0) / poly.length,
  poly.reduce((a, p) => a + p[1], 0) / poly.length];

// [fraction of the scale, distance from the centre in scale radii] of an image point.
export function place(point, kp, fit) {
  let ang, r, sweep;
  if (fit) {
    const q = project(inv3(fit.homography), point[0], point[1]);
    ang = (((Math.atan2(q[1], q[0]) * 180) / Math.PI) % 360 + 360) % 360;
    r = Math.hypot(q[0], q[1]);
    sweep = fit.sweep;
  } else {
    sweep = clockwiseAngle(kp.center, kp.scale_min, kp.scale_max);
    ang = clockwiseAngle(kp.center, kp.scale_min, point);
    const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    const rad = (d(kp.center, kp.scale_min) + d(kp.center, kp.scale_max)) / 2;
    r = rad > 0 ? d(kp.center, point) / rad : Infinity;
  }
  if (sweep <= 0) return [NaN, r];
  if (ang > sweep && ang - sweep > (360 - sweep) / 2) ang -= 360;
  return [ang / sweep, r];
}

// The value and, for a whole number of 2+ digits, the same digits with a dropped decimal point.
export function alternatives(text, value) {
  const out = [value];
  const digits = (text.includes(".") ? text.split(".")[0] : text).replace(/\D/g, "");
  if (!text.includes(".") && !text.includes(",") && digits.length >= 2 && !digits.startsWith("0")) {
    const sign = value < 0 ? -1 : 1;
    for (let k = 1; k < digits.length; k++) out.push(sign * Number(`${digits.slice(0, k)}.${digits.slice(k)}`));
  }
  return [...new Set(out)];
}

export function labelsOnDial(boxes, kp, fit) {
  const out = [];
  boxes.forEach(({ text, poly, conf }, i) => {
    const [v] = parseNumber(text);
    if (v === null) return;
    const [f, r] = place(centerOf(poly), kp, fit);
    if (r >= RADIUS_RANGE[0] && r <= RADIUS_RANGE[1] && f >= FRACTION_RANGE[0] && f <= FRACTION_RANGE[1]) {
      for (const alt of alternatives(text, v)) out.push({ text, value: alt, fraction: f, radius: r, conf, box: i, poly });
    }
  });
  return out;
}

function polyfit1(xs, ys) {
  const n = xs.length, mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  const b = sxy / sxx;
  return [b, my - b * mx];
}

// RANSAC over label pairs: value = a + b * fraction with b > 0. Returns [a, b, inliers] or null.
export function fitLine(labels, tol = INLIER_TOL, minInliers = MIN_INLIERS) {
  let best = null;
  const n = labels.length;
  const better = (s, t) => {
    for (let i = 0; i < s.length; i++) if (s[i] !== t[i]) return s[i] > t[i];
    return false;
  };
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const li = labels[i], lj = labels[j];
      if (li.box === lj.box) continue;
      const df = lj.fraction - li.fraction;
      if (Math.abs(df) < 0.08) continue;
      const b = (lj.value - li.value) / df;
      if (b <= 0) continue;
      const a = li.value - b * li.fraction;
      const perBox = new Map();
      for (const lab of labels) {
        const e = Math.abs(a + b * lab.fraction - lab.value) / b;
        if (e <= tol && (!perBox.has(lab.box) || e < perBox.get(lab.box)[0])) perBox.set(lab.box, [e, lab]);
      }
      const inl = [...perBox.values()].map(([, lab]) => lab);
      const k = new Set(inl.map((lab) => Math.round(lab.value * 1e9) / 1e9)).size;
      const asRead = inl.filter((lab) => lab.value === parseNumber(lab.text)[0]).length;
      const errSum = [...perBox.values()].reduce((s, [e]) => s + e, 0);
      const score = [k + 0.5 * asRead, k, -errSum];
      if (best === null || better(score, best[0])) best = [score, inl];
    }
  }
  if (best === null || best[0][1] < Math.max(minInliers, 2)) return null;
  const inl = best[1];
  const [b, a] = polyfit1(inl.map((l) => l.fraction), inl.map((l) => l.value));
  if (!(b > 0)) return null;
  return [a, b, inl];
}

function snap(x, span, values, step) {
  const near = values.filter((v) => Math.abs(v - x) <= SNAP_TOL * span);
  if (near.length) return near.reduce((m, v) => (Math.abs(v - x) < Math.abs(m - x) ? v : m));
  if (Math.abs(x) <= SNAP_TOL * span) return 0;
  if (step) return Math.round(x / step) * step;
  return x;
}

// boxes: [{text, poly: [[x, y] x4], conf}] in photo coordinates. kp: {center, tip, scale_min,
// scale_max}. fit: the dial fit or null. Returns {scaleMin, scaleMax, unit, labels, inliers}.
export function readScale(boxes, kp, fit) {
  const labels = labelsOnDial(boxes, kp, fit);
  const unit = findUnit(boxes.map((b) => b.text));
  const line = fitLine(labels);
  if (!line) return { scaleMin: null, scaleMax: null, unit, labels, inliers: [] };
  const [a, b, inl] = line;
  const vals = [...new Set(inl.map((l) => l.value))].sort((x, y) => x - y);
  let step = null;
  if (vals.length >= 2) step = Math.min(...vals.slice(1).map((v, i) => v - vals[i]));
  const lo = snap(a, b, vals, step), hi = snap(a + b, b, vals, step);
  if (hi <= lo) return { scaleMin: null, scaleMax: null, unit, labels, inliers: inl };
  return { scaleMin: lo, scaleMax: hi, unit, labels, inliers: inl };
}
