// Text on the dial with PP-OCRv4 (the ONNX models RapidOCR ships, as used by the Python
// reader's RapidOcr) on onnxruntime-web: DB text detection, then CTC recognition. A port of
// rapidocr_onnxruntime's pipeline without the text-angle classifier (the Python reader turns
// it off too: dial numbers are printed upright). Images are {w, h, data} with RGBA bytes; the
// models take BGR like OpenCV. Returns [{text, poly: [[x, y] x4], conf}] in image pixels.

const DET = { limitSide: 736, thresh: 0.3, boxThresh: 0.5, maxCandidates: 1000, unclip: 1.6, minSize: 3 };
const REC = { h: 48, w: 320, batch: 6, textScore: 0.5 };
const MAX_SIDE = 2000, MIN_SIDE = 30;

// ---- image helpers
export function imageFromCanvas(c) {
  const g = c.getContext("2d", { willReadFrequently: true });
  const d = g.getImageData(0, 0, c.width, c.height);
  return { w: c.width, h: c.height, data: d.data };
}

export function cropImage(img, x0, y0, x1, y1) {
  const w = x1 - x0, h = y1 - y0, data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const s = 4 * ((y + y0) * img.w + x0);
    data.set(img.data.subarray(s, s + 4 * w), 4 * y * w);
  }
  return { w, h, data };
}

// cv2.resize INTER_LINEAR (pixel centres at +0.5, no antialiasing)
export function resizeLinear(img, nw, nh) {
  const out = new Uint8ClampedArray(nw * nh * 4), sx = img.w / nw, sy = img.h / nh;
  const xs = new Int32Array(nw * 2), wx = new Float32Array(nw);
  for (let x = 0; x < nw; x++) {
    let fx = (x + 0.5) * sx - 0.5;
    let x0 = Math.floor(fx);
    let a = fx - x0;
    if (x0 < 0) { x0 = 0; a = 0; }
    if (x0 >= img.w - 1) { x0 = img.w - 1; a = 0; }
    xs[2 * x] = x0; xs[2 * x + 1] = Math.min(x0 + 1, img.w - 1); wx[x] = a;
  }
  for (let y = 0; y < nh; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    let y0 = Math.floor(fy);
    let b = fy - y0;
    if (y0 < 0) { y0 = 0; b = 0; }
    if (y0 >= img.h - 1) { y0 = img.h - 1; b = 0; }
    const y1 = Math.min(y0 + 1, img.h - 1), r0 = y0 * img.w, r1 = y1 * img.w;
    for (let x = 0; x < nw; x++) {
      const a = wx[x], i00 = 4 * (r0 + xs[2 * x]), i01 = 4 * (r0 + xs[2 * x + 1]);
      const i10 = 4 * (r1 + xs[2 * x]), i11 = 4 * (r1 + xs[2 * x + 1]), o = 4 * (y * nw + x);
      for (let ch = 0; ch < 3; ch++) {
        out[o + ch] = (img.data[i00 + ch] * (1 - a) + img.data[i01 + ch] * a) * (1 - b)
          + (img.data[i10 + ch] * (1 - a) + img.data[i11 + ch] * a) * b;
      }
      out[o + 3] = 255;
    }
  }
  return { w: nw, h: nh, data: out };
}

const round32 = (v) => Math.round(v / 32) * 32;

// ---- geometry for the DB post-processing
function convexHull(pts) {
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

// cv2.minAreaRect: rotating calipers over the hull. Returns {corners, w, h}.
function minAreaRect(hull) {
  if (hull.length === 1) return { corners: [hull[0], hull[0], hull[0], hull[0]], w: 0, h: 0 };
  let best = null;
  for (let i = 0; i < hull.length; i++) {
    const p = hull[i], q = hull[(i + 1) % hull.length];
    const len = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (len === 0) continue;
    const u = [(q[0] - p[0]) / len, (q[1] - p[1]) / len], v = [-u[1], u[0]];
    let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    for (const s of hull) {
      const a = s[0] * u[0] + s[1] * u[1], b = s[0] * v[0] + s[1] * v[1];
      a0 = Math.min(a0, a); a1 = Math.max(a1, a); b0 = Math.min(b0, b); b1 = Math.max(b1, b);
    }
    const area = (a1 - a0) * (b1 - b0);
    if (!best || area < best.area) best = { area, u, v, a0, a1, b0, b1 };
  }
  const { u, v, a0, a1, b0, b1 } = best;
  const pt = (a, b) => [a * u[0] + b * v[0], a * u[1] + b * v[1]];
  return { corners: [pt(a0, b0), pt(a1, b0), pt(a1, b1), pt(a0, b1)], w: a1 - a0, h: b1 - b0 };
}

// get_mini_boxes: the corners in the order RapidOCR uses, and the short side.
function miniBox(rect) {
  const p = [...rect.corners].sort((a, b) => a[0] - b[0]);
  const [i1, i4] = p[1][1] > p[0][1] ? [0, 1] : [1, 0];
  const [i2, i3] = p[3][1] > p[2][1] ? [2, 3] : [3, 2];
  return [[p[i1], p[i2], p[i3], p[i4]], Math.min(rect.w, rect.h)];
}

function expand(rect, d) {
  const c = rect.corners, cx = (c[0][0] + c[2][0]) / 2, cy = (c[0][1] + c[2][1]) / 2;
  const u = [c[1][0] - c[0][0], c[1][1] - c[0][1]], v = [c[3][0] - c[0][0], c[3][1] - c[0][1]];
  const lu = Math.hypot(...u) || 1, lv = Math.hypot(...v) || 1;
  const w = rect.w + 2 * d, h = rect.h + 2 * d;
  const eu = [(u[0] / lu) * w / 2, (u[1] / lu) * w / 2], ev = [(v[0] / lv) * h / 2, (v[1] / lv) * h / 2];
  return { corners: [[cx - eu[0] - ev[0], cy - eu[1] - ev[1]], [cx + eu[0] - ev[0], cy + eu[1] - ev[1]],
    [cx + eu[0] + ev[0], cy + eu[1] + ev[1]], [cx - eu[0] + ev[0], cy - eu[1] + ev[1]]], w, h };
}

function insideConvex(poly, x, y) {
  let sign = 0;
  for (let i = 0; i < poly.length; i++) {
    const [ax, ay] = poly[i], [bx, by] = poly[(i + 1) % poly.length];
    const c = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
    if (Math.abs(c) < 1e-9) continue;
    const s = c > 0 ? 1 : -1;
    if (sign === 0) sign = s; else if (s !== sign) return false;
  }
  return true;
}

function boxScoreFast(pred, W, H, box) {
  const xs = box.map((p) => p[0]), ys = box.map((p) => p[1]);
  const clip = (v, hi) => Math.min(Math.max(v, 0), hi);
  const xmin = clip(Math.floor(Math.min(...xs)), W - 1), xmax = clip(Math.ceil(Math.max(...xs)), W - 1);
  const ymin = clip(Math.floor(Math.min(...ys)), H - 1), ymax = clip(Math.ceil(Math.max(...ys)), H - 1);
  const poly = box.map(([x, y]) => [Math.trunc(x - xmin), Math.trunc(y - ymin)]);
  let s = 0, n = 0;
  for (let y = ymin; y <= ymax; y++) {
    for (let x = xmin; x <= xmax; x++) {
      if (insideConvex(poly, x - xmin, y - ymin)) { s += pred[y * W + x]; n++; }
    }
  }
  return n ? s / n : 0;
}

// connected regions (8-neighbours) of the dilated text mask; per region the leftmost and
// rightmost pixel of each row (enough for the convex hull)
function regions(mask, W, H) {
  const label = new Int32Array(W * H), out = [];
  const stack = [];
  for (let start = 0; start < W * H; start++) {
    if (!mask[start] || label[start]) continue;
    const id = out.length + 1, rows = new Map();
    label[start] = id; stack.push(start);
    while (stack.length) {
      const i = stack.pop(), x = i % W, y = (i - x) / W;
      const r = rows.get(y);
      if (!r) rows.set(y, [x, x]); else { if (x < r[0]) r[0] = x; if (x > r[1]) r[1] = x; }
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= W) continue;
          const j = yy * W + xx;
          if (mask[j] && !label[j]) { label[j] = id; stack.push(j); }
        }
      }
    }
    const pts = [];
    for (const [y, [x0, x1]] of rows) { pts.push([x0, y]); if (x1 !== x0) pts.push([x1, y]); }
    out.push(pts);
  }
  return out;
}

function orderClockwise(pts) {
  const xs = [...pts].sort((a, b) => a[0] - b[0]);
  const left = xs.slice(0, 2).sort((a, b) => a[1] - b[1]), right = xs.slice(2).sort((a, b) => a[1] - b[1]);
  return [left[0], right[0], right[1], left[1]];
}

// ---- perspective crop of a text box (get_rotate_crop_image)
function solveHomography(src, dst) {
  // h maps src -> dst; 8 unknowns, h33 = 1
  const a = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    a.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  for (let c = 0; c < 8; c++) {
    let p = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    [a[c], a[p]] = [a[p], a[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let r = 0; r < 8; r++) {
      if (r === c || a[c][c] === 0) continue;
      const f = a[r][c] / a[c][c];
      for (let k = c; k < 8; k++) a[r][k] -= f * a[c][k];
      b[r] -= f * b[c];
    }
  }
  const h = b.map((v, i) => v / a[i][i]);
  return [...h, 1];
}

function cropText(img, box) {
  const d = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  const cw = Math.trunc(Math.max(d(box[0], box[1]), d(box[2], box[3])));
  const ch = Math.trunc(Math.max(d(box[0], box[3]), d(box[1], box[2])));
  if (cw < 1 || ch < 1) return null;
  const m = solveHomography([[0, 0], [cw, 0], [cw, ch], [0, ch]], box);   // crop -> image
  const data = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const z = m[6] * x + m[7] * y + m[8];
      let sx = (m[0] * x + m[1] * y + m[2]) / z, sy = (m[3] * x + m[4] * y + m[5]) / z;
      sx = Math.min(Math.max(sx, 0), img.w - 1); sy = Math.min(Math.max(sy, 0), img.h - 1);
      const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
      const x1 = Math.min(x0 + 1, img.w - 1), y1 = Math.min(y0 + 1, img.h - 1);
      const i00 = 4 * (y0 * img.w + x0), i01 = 4 * (y0 * img.w + x1);
      const i10 = 4 * (y1 * img.w + x0), i11 = 4 * (y1 * img.w + x1), o = 4 * (y * cw + x);
      for (let c = 0; c < 3; c++) {
        data[o + c] = (img.data[i00 + c] * (1 - fx) + img.data[i01 + c] * fx) * (1 - fy)
          + (img.data[i10 + c] * (1 - fx) + img.data[i11 + c] * fx) * fy;
      }
      data[o + 3] = 255;
    }
  }
  let out = { w: cw, h: ch, data };
  if (ch / cw >= 1.5) {     // np.rot90: a quarter turn counter-clockwise
    const r = new Uint8ClampedArray(data.length);
    for (let y = 0; y < cw; y++) {
      for (let x = 0; x < ch; x++) {
        const s = 4 * (x * cw + (cw - 1 - y)), o = 4 * (y * ch + x);
        r[o] = data[s]; r[o + 1] = data[s + 1]; r[o + 2] = data[s + 2]; r[o + 3] = 255;
      }
    }
    out = { w: ch, h: cw, data: r };
  }
  return out;
}

// BGR planes, (v / 255 - 0.5) / 0.5, written into t at offset (planes of size W*H, row stride W)
function writeBgr(img, t, off, W, H) {
  const n = W * H;
  for (let y = 0; y < img.h; y++) {
    for (let x = 0; x < img.w; x++) {
      const i = 4 * (y * img.w + x), o = off + y * W + x;
      t[o] = img.data[i + 2] / 127.5 - 1;
      t[o + n] = img.data[i + 1] / 127.5 - 1;
      t[o + 2 * n] = img.data[i] / 127.5 - 1;
    }
  }
}

export class DialOcr {
  constructor(ort, det, rec, chars) {
    Object.assign(this, { ort, det, rec, chars });
  }

  // base: URL of a folder with det.onnx, rec.onnx and keys.json
  static async load(ort, base) {
    const opts = { executionProviders: ["wasm"] };
    const [det, rec, keys] = await Promise.all([
      ort.InferenceSession.create(`${base}/det.onnx`, opts),
      ort.InferenceSession.create(`${base}/rec.onnx`, opts),
      fetch(`${base}/keys.json`).then((r) => r.json()),
    ]);
    return new DialOcr(ort, det, rec, ["blank", ...keys, " "]);
  }

  async detect(img) {
    const { w, h } = img;
    const ratio = Math.min(h, w) < DET.limitSide ? DET.limitSide / Math.min(h, w) : 1;
    const rh = round32(Math.trunc(h * ratio)), rw = round32(Math.trunc(w * ratio));
    if (rh <= 0 || rw <= 0) return [];
    const r = resizeLinear(img, rw, rh);
    const t = new Float32Array(3 * rw * rh);
    writeBgr(r, t, 0, rw, rh);
    const out = await this.det.run({ [this.det.inputNames[0]]: new this.ort.Tensor("float32", t, [1, 3, rh, rw]) });
    const pred = out[this.det.outputNames[0]].data;
    const mask = new Uint8Array(rw * rh);
    for (let y = 0; y < rh; y++) {        // threshold, then dilate with a 2x2 kernel
      for (let x = 0; x < rw; x++) {
        const i = y * rw + x;
        mask[i] = pred[i] > DET.thresh || (x > 0 && pred[i - 1] > DET.thresh)
          || (y > 0 && pred[i - rw] > DET.thresh) || (x > 0 && y > 0 && pred[i - rw - 1] > DET.thresh) ? 1 : 0;
      }
    }
    const boxes = [];
    for (const pts of regions(mask, rw, rh).slice(0, DET.maxCandidates)) {
      const rect = minAreaRect(convexHull(pts));
      const [pbox, sside] = miniBox(rect);
      if (sside < DET.minSize) continue;
      if (boxScoreFast(pred, rw, rh, pbox) < DET.boxThresh) continue;
      const dist = (rect.w * rect.h * DET.unclip) / (2 * (rect.w + rect.h));
      const [box, s2] = miniBox(expand(rect, dist));
      if (s2 < DET.minSize + 2) continue;
      const scaled = box.map(([x, y]) => [Math.min(Math.max(Math.round((x / rw) * w), 0), w),
        Math.min(Math.max(Math.round((y / rh) * h), 0), h)]);
      const ordered = orderClockwise(scaled).map(([x, y]) => [Math.trunc(Math.min(Math.max(x, 0), w - 1)),
        Math.trunc(Math.min(Math.max(y, 0), h - 1))]);
      const d = (p, q) => Math.trunc(Math.hypot(p[0] - q[0], p[1] - q[1]));
      if (d(ordered[0], ordered[1]) <= 3 || d(ordered[0], ordered[3]) <= 3) continue;
      boxes.push(ordered);
    }
    return boxes;
  }

  async recognize(crops) {
    const ratios = crops.map((c) => c.w / c.h);
    const order = ratios.map((_, i) => i).sort((a, b) => ratios[a] - ratios[b]);
    const res = new Array(crops.length);
    for (let s = 0; s < order.length; s += REC.batch) {
      const idx = order.slice(s, s + REC.batch);
      const maxRatio = Math.max(REC.w / REC.h, ...idx.map((i) => ratios[i]));
      const W = Math.trunc(REC.h * maxRatio), H = REC.h, n = W * H;
      const t = new Float32Array(idx.length * 3 * n);
      idx.forEach((i, k) => {
        const rw = Math.min(Math.ceil(H * ratios[i]), W);
        writeBgr(resizeLinear(crops[i], rw, H), t, k * 3 * n, W, H);
      });
      const out = await this.rec.run({ [this.rec.inputNames[0]]: new this.ort.Tensor("float32", t, [idx.length, 3, H, W]) });
      const y = out[this.rec.outputNames[0]], [, T, C] = y.dims;
      idx.forEach((i, k) => {
        let text = "", confs = [], prev = -1;
        for (let tt = 0; tt < T; tt++) {
          const o = (k * T + tt) * C;
          let best = 0, bp = y.data[o];
          for (let c = 1; c < C; c++) if (y.data[o + c] > bp) { bp = y.data[o + c]; best = c; }
          if (best !== prev && best !== 0) { text += this.chars[best]; confs.push(bp); }
          prev = best;
        }
        res[i] = [text, confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : 0];
      });
    }
    return res;
  }

  // Text boxes of an image, as RapidOCR returns them (score >= 0.5).
  async read(img) {
    let work = img, rx = 1, ry = 1;
    if (Math.max(img.w, img.h) > MAX_SIDE || Math.min(img.w, img.h) < MIN_SIDE) {
      const r = Math.max(img.w, img.h) > MAX_SIDE ? MAX_SIDE / Math.max(img.w, img.h) : MIN_SIDE / Math.min(img.w, img.h);
      const nh = round32(Math.trunc(img.h * r)), nw = round32(Math.trunc(img.w * r));
      if (nh <= 0 || nw <= 0) return [];
      work = resizeLinear(img, nw, nh);
      rx = img.w / nw; ry = img.h / nh;
    }
    const boxes = await this.detect(work);
    if (!boxes.length) return [];
    const keep = [], crops = [];
    for (const b of boxes) {
      const c = cropText(work, b);
      if (c) { keep.push(b); crops.push(c); }
    }
    const rec = await this.recognize(crops);
    const out = [];
    keep.forEach((b, i) => {
      const [text, conf] = rec[i];
      if (conf >= REC.textScore) {
        out.push({ text, conf, poly: b.map(([x, y]) => [Math.min(Math.max(x * rx, 0), img.w), Math.min(Math.max(y * ry, 0), img.h)]) });
      }
    });
    return out;
  }
}
