// The gauge reader in the browser: a port of gauge_reader.reader.GaugeReader.read. The range is
// given (registry or visitor) or read from the dial numbers (ocr.js + scale.js). Canvas image
// operations + onnxruntime-web; geometry in geometry.js.
import { decodeTop, fitDial, letterbox, median, clockwiseAngle } from "./geometry.js";
import { cropImage } from "./ocr.js";
import { readScale } from "./scale.js";

export const CROP_MARGIN = 0.5;      // real context around a small gauge's box, per side
export const CROP_PAD = 0.15;        // grey margin around a close-up read whole
export const WHOLE_IMAGE_SHARE = 0.4; // a box this large (share of the photo) is read whole
export const TTA_ANGLE = 25;
export const RETAKE_SPREAD = 5;      // % of scale
export const MIN_SWEEP_DEG = 90;
export const CLIP = 0.1;
export const CONF = 0.1;
export const IMGSZ = 640;
export const OCR_CROP = 1.6;         // OCR window, in scale radii around the centre
const GREY = "rgb(114,114,114)";

function canvas(w, h) {
  const c = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h)
    : Object.assign(document.createElement("canvas"), { width: w, height: h });
  return c;
}

function grey(w, h) {
  const c = canvas(w, h), g = c.getContext("2d", { willReadFrequently: true });
  g.fillStyle = GREY;
  g.fillRect(0, 0, w, h);
  return [c, g];
}

// pad_image: frac of the size on every side, grey. Returns [canvas, dx, dy].
export function padImage(src, frac) {
  if (frac <= 0) return [src, 0, 0];
  const dx = Math.round(src.width * frac), dy = Math.round(src.height * frac);
  const [c, g] = grey(src.width + 2 * dx, src.height + 2 * dy);
  g.drawImage(src, dx, dy);
  return [c, dx, dy];
}

// cv2 rotation by angle degrees counter-clockwise about the centre, same size, grey corners.
export function rotateImage(src, angle) {
  const [c, g] = grey(src.width, src.height);
  g.translate(src.width / 2, src.height / 2);
  g.rotate((-angle * Math.PI) / 180);
  g.translate(-src.width / 2, -src.height / 2);
  g.drawImage(src, 0, 0);
  return c;
}

// crop_box(square=True): a square window around the box centre plus margin per side.
export function cropBox(src, box, margin = CROP_MARGIN) {
  let [x0, y0, x1, y1] = box;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, half = Math.max(x1 - x0, y1 - y0) / 2;
  [x0, y0, x1, y1] = [cx - half, cy - half, cx + half, cy + half];
  const mx = (x1 - x0) * margin, my = (y1 - y0) * margin;
  const X0 = Math.floor(x0 - mx), Y0 = Math.floor(y0 - my);
  const X1 = Math.ceil(x1 + mx), Y1 = Math.ceil(y1 + my);
  const [c, g] = grey(X1 - X0, Y1 - Y0);
  g.drawImage(src, -X0, -Y0);
  return [c, X0, Y0];
}

function tensorOf(ort, src) {
  const lb = letterbox(src.width, src.height, IMGSZ);
  const [c, g] = grey(IMGSZ, IMGSZ);
  g.imageSmoothingQuality = "high";
  g.drawImage(src, lb.left, lb.top, lb.nw, lb.nh);
  const px = g.getImageData(0, 0, IMGSZ, IMGSZ).data;
  const n = IMGSZ * IMGSZ, data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    data[i] = px[4 * i] / 255;
    data[n + i] = px[4 * i + 1] / 255;
    data[2 * n + i] = px[4 * i + 2] / 255;
  }
  return [new ort.Tensor("float32", data, [1, 3, IMGSZ, IMGSZ]), lb];
}

function toDetection(d, dx = 0, dy = 0) {
  const p = d.kpts.map(([x, y]) => [x + dx, y + dy]);
  const [x0, y0, x1, y1] = d.box;
  return {
    box: [x0 + dx, y0 + dy, x1 + dx, y1 + dy], conf: d.conf,
    keypoints: { center: p[0], tip: p[1], scale_min: p[2], scale_max: p[3] },
    arc: p.slice(4), kptConf: d.kpts.map((k) => k[2]),
  };
}

function dialFit(det) {
  if (det._fit === undefined) {
    try { det._fit = fitDial(det.keypoints, det.arc); } catch { det._fit = null; }
  }
  return det._fit;
}

function sweepOf(det) {
  const f = dialFit(det);
  if (f) return f.sweep;
  const k = det.keypoints;
  return clockwiseAngle(k.center, k.scale_min, k.scale_max);
}

// Needle fraction on the dial face; in the image plane when no dial fits (geometry.py).
function fractionOf(det) {
  const f = dialFit(det);
  if (f) return f.fraction;
  const k = det.keypoints, sweep = clockwiseAngle(k.center, k.scale_min, k.scale_max);
  let needle = clockwiseAngle(k.center, k.scale_min, k.tip);
  if (!sweep) return NaN;
  if (needle > sweep && needle - sweep > (360 - sweep) / 2) needle -= 360;
  return needle / sweep;
}

export class GaugeReader {
  constructor(ort, session) {
    this.ort = ort;
    this.session = session;
  }

  static async load(ort, modelUrl, onProgress) {
    const res = await fetch(modelUrl);
    if (!res.ok) throw new Error(`model download failed: ${res.status}`);
    const total = Number(res.headers.get("content-length")) || 0;
    const reader = res.body.getReader(), parts = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      got += value.length;
      if (onProgress) onProgress(got, total);
    }
    const buf = new Uint8Array(got);
    let o = 0;
    for (const p of parts) { buf.set(p, o); o += p.length; }
    const session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    return new GaugeReader(ort, session);
  }

  async detect(src) {
    const [tensor, lb] = tensorOf(this.ort, src);
    const out = await this.session.run({ [this.session.inputNames[0]]: tensor });
    const y = out[this.session.outputNames[0]];
    const d = decodeTop(y.data, y.dims[2], lb, src.width, src.height, CONF, (y.dims[1] - 5) / 3);
    return d ? toDetection(d) : null;
  }

  // img: a canvas or ImageBitmap of the photo. scale: {min, max, unit} when known; else the
  // range is read from the dial numbers with ocr (a DialOcr, ocr.js), or the answer is no_scale.
  // band: {low, high} (either may be null). fast: upright pass only (no RETAKE spread check).
  async read(img, scale, band = null, fast = false, ocr = null) {
    const ms = {};
    let t = performance.now();
    const [padded, px, py] = padImage(img, CROP_PAD);
    const top0 = await this.detect(padded);
    ms.detect = performance.now() - t;
    if (!top0) return { status: "no_gauge", alarm: "RETAKE", ms };
    const top = shiftDet(top0, -px, -py);
    const out = { status: "ok", alarm: "RETAKE", box: top.box, boxConf: top.conf, ms };

    const [x0, y0, x1, y1] = top.box;
    out.boxShare = ((x1 - x0) * (y1 - y0)) / (img.width * img.height);
    const closeup = out.boxShare >= WHOLE_IMAGE_SHARE;
    let crop, ox, oy, pad;
    if (closeup) { crop = img; ox = 0; oy = 0; pad = CROP_PAD; } else {
      [crop, ox, oy] = cropBox(img, top.box);
      pad = 0;
    }
    const [, cpx, cpy] = padImage(crop, pad);
    ox -= cpx; oy -= cpy;
    out.framing = closeup ? "whole photo" : "square crop";
    const angles = fast ? [0] : [0, TTA_ANGLE, -TTA_ANGLE];
    const fracs = [];
    let upright = null;
    t = performance.now();
    for (const ang of angles) {
      let d;
      if (ang === 0 && closeup) d = top0;
      else d = await this.detect(padImage(ang ? rotateImage(crop, ang) : crop, pad)[0]);
      if (!d) continue;
      if (ang === 0) upright = shiftDet(d, ox, oy);
      if (sweepOf(d) >= MIN_SWEEP_DEG) {
        const f = fractionOf(d);
        if (Number.isFinite(f)) fracs.push(Math.min(Math.max(f, -CLIP), 1 + CLIP));
      }
    }
    ms.keypoints = performance.now() - t;
    if (!upright) upright = top;
    out.keypoints = upright.keypoints;
    out.arc = upright.arc;
    out.fit = dialFit(upright);
    out.fractions = fracs;
    if (sweepOf(upright) < MIN_SWEEP_DEG || !fracs.length) {
      out.status = "bad_geometry";
      return out;
    }
    out.fraction = median(fracs);
    if (angles.length > 1) {
      out.spread = fracs.length >= 2 ? (Math.max(...fracs) - Math.min(...fracs)) * 100 : 99;
      if (out.spread > RETAKE_SPREAD) { out.status = "retake"; return out; }
    }
    if (scale) out.scaleSource = "given";
    else if (ocr) {
      t = performance.now();
      const sr = await readDialScale(img, out.keypoints, out.fit, ocr);
      ms.ocr = performance.now() - t;
      out.labels = sr.inliers;
      out.unit = sr.unit;
      if (sr.scaleMin !== null) {
        scale = { min: sr.scaleMin, max: sr.scaleMax, unit: sr.unit };
        out.scaleSource = "ocr";
      }
    }
    if (!scale) { out.status = "no_scale"; return out; }
    out.scaleMin = scale.min; out.scaleMax = scale.max; out.unit = scale.unit;
    out.value = scale.min + out.fraction * (scale.max - scale.min);
    out.alarm = "";
    if (band) {
      if (band.low != null && out.value < band.low) out.alarm = "LOW";
      else if (band.high != null && out.value > band.high) out.alarm = "HIGH";
      else out.alarm = "NORMAL";
    }
    return out;
  }
}

// The dial around the predicted centre, OCR_CROP scale radii on each side (scale_ocr.dial_crop),
// read with OCR; the text boxes are placed on the scale (scale.js).
export async function readDialScale(img, kp, fit, ocr) {
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(img, 0, 0);
  const full = { w: c.width, h: c.height, data: g.getImageData(0, 0, c.width, c.height).data };
  const [cx, cy] = kp.center;
  const r = Math.max(Math.hypot(kp.scale_min[0] - cx, kp.scale_min[1] - cy),
    Math.hypot(kp.scale_max[0] - cx, kp.scale_max[1] - cy)) * OCR_CROP;
  let x0 = Math.max(Math.trunc(cx - r), 0), y0 = Math.max(Math.trunc(cy - r), 0);
  let x1 = Math.min(Math.trunc(cx + r) + 1, full.w), y1 = Math.min(Math.trunc(cy + r) + 1, full.h);
  if (x1 - x0 < 16 || y1 - y0 < 16) { x0 = 0; y0 = 0; x1 = full.w; y1 = full.h; }
  const boxes = await ocr.read(cropImage(full, x0, y0, x1, y1));
  const placed = boxes.map((b) => ({ ...b, poly: b.poly.map(([x, y]) => [x + x0, y + y0]) }));
  return readScale(placed, kp, fit);
}

function shiftDet(d, dx, dy) {
  return toDetection({ box: d.box, conf: d.conf, kpts: [d.keypoints.center, d.keypoints.tip,
    d.keypoints.scale_min, d.keypoints.scale_max, ...d.arc].map((p, i) => [...p, d.kptConf[i]]) },
  dx, dy);
}
