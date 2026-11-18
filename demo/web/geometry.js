// Dial-face geometry for the browser demo: a port of gauge_reader.reading.perspective
// (fit_dial, frontal_matrix) and the YOLO pose output decoding. Pure functions, no DOM,
// so tests/test_web_geometry.py can check them against the Python code with Node.

export const ARC_FRACTIONS = [0.25, 0.5, 0.75];
export const SWEEP_RANGE = [20.0, 359.0];

// 3x3 matrices are flat arrays of 9 numbers, row-major.
export function project(h, x, y) {
  const w = h[6] * x + h[7] * y + h[8];
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
}

export function mul3(a, b) {
  const o = new Array(9).fill(0);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) o[3 * i + j] += a[3 * i + k] * b[3 * k + j];
  return o;
}

export function det3(m) {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
    + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

export function inv3(m) {
  const d = det3(m);
  const c = [
    m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
    m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
    m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3]];
  return c.map((v) => v / d);
}

// Eigenvector of the smallest eigenvalue of a symmetric n x n matrix (cyclic Jacobi).
function smallestEigvec(s, n) {
  const a = s.slice();
  const v = new Array(n * n).fill(0);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < n; k++) {      // rotate columns p, q
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - sn * akq;
          a[k * n + q] = sn * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {      // and rows p, q
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - sn * aqk;
          a[q * n + k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p], vkq = v[k * n + q];
          v[k * n + p] = c * vkp - sn * vkq;
          v[k * n + q] = sn * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < n; i++) if (a[i * n + i] < a[best * n + best]) best = i;
  const out = [];
  for (let k = 0; k < n; k++) out.push(v[k * n + best]);
  return out;
}

function facePoints(sweep, fractions) {
  const pts = fractions.map((f) => {
    const a = (sweep * f * Math.PI) / 180;
    return [Math.cos(a), Math.sin(a)];
  });
  pts.push([0, 0]);
  return pts;
}

// Least-squares (DLT) homography from the face model to the image for one sweep, as
// perspective._fit_batch. Returns [h, summed squared reprojection error in px].
function fitOne(sweep, img, fractions, norm) {
  const { cx, cy, s, tinv } = norm;
  const face = facePoints(sweep, fractions);
  const rows = [];
  for (let i = 0; i < face.length; i++) {
    const [x, y] = face[i];
    const u = (img[i][0] - cx) * s, v = (img[i][1] - cy) * s;
    rows.push([x, y, 1, 0, 0, 0, -u * x, -u * y, -u]);
    rows.push([0, 0, 0, x, y, 1, -v * x, -v * y, -v]);
  }
  const ata = new Array(81).fill(0);
  for (const r of rows)
    for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) ata[9 * i + j] += r[i] * r[j];
  const h = mul3(tinv, smallestEigvec(ata, 9));
  let err = 0;
  for (let i = 0; i < face.length; i++) {
    const [px, py] = project(h, face[i][0], face[i][1]);
    err += (px - img[i][0]) ** 2 + (py - img[i][1]) ** 2;
  }
  return [h, Number.isFinite(err) ? err : Infinity];
}

// Needle fraction read on the dial face (perspective.fit_dial). kp: {center, tip,
// scale_min, scale_max} as [x, y]; arc: the three scale-arc points. Throws when no fit.
export function fitDial(kp, arc, fractions = ARC_FRACTIONS) {
  if (arc.length !== fractions.length) throw new Error("arc points do not match fractions");
  const fr = [0, ...fractions, 1];
  const img = [kp.scale_min, ...arc, kp.scale_max, kp.center];
  if (!img.every((p) => Number.isFinite(p[0]) && Number.isFinite(p[1])))
    throw new Error("keypoints are not finite");
  const cx = img.reduce((a, p) => a + p[0], 0) / img.length;
  const cy = img.reduce((a, p) => a + p[1], 0) / img.length;
  const md = img.reduce((a, p) => a + Math.hypot(p[0] - cx, p[1] - cy), 0) / img.length;
  const s = Math.SQRT2 / Math.max(md, 1e-9);
  const norm = { cx, cy, s, tinv: [1 / s, 0, cx, 0, 1 / s, cy, 0, 0, 1] };

  let best = null, bestErr = Infinity;
  for (let sw = SWEEP_RANGE[0]; sw <= SWEEP_RANGE[1] + 1e-9; sw += 1.0) {
    const [, err] = fitOne(sw, img, fr, norm);
    if (err < bestErr) { bestErr = err; best = sw; }
  }
  if (best === null) throw new Error("no homography fits these keypoints");
  let sweep = 0, h = null, sq = Infinity;
  for (let i = 0; i <= 100; i++) {
    const sw = Math.min(Math.max(best - 1.0 + 0.02 * i, SWEEP_RANGE[0]), SWEEP_RANGE[1]);
    const [hh, err] = fitOne(sw, img, fr, norm);
    if (err < sq) { sq = err; h = hh; sweep = sw; }
  }
  if (h === null || Math.abs(det3(h)) < 1e-12) throw new Error("degenerate homography");

  const q = project(inv3(h), kp.tip[0], kp.tip[1]);
  let t = (((Math.atan2(q[1], q[0]) * 180) / Math.PI) % 360 + 360) % 360;
  if (t > sweep && t - sweep > (360 - sweep) / 2) t -= 360;
  const c = img[img.length - 1];
  let radius = 0;
  for (let i = 0; i < img.length - 1; i++) radius += Math.hypot(img[i][0] - c[0], img[i][1] - c[1]);
  radius /= img.length - 1;
  const rms = radius > 0 ? Math.sqrt(sq / img.length) / radius : Infinity;
  return { fraction: t / sweep, sweep, rms, homography: h };
}

// Map from the photo to the frontal view (perspective.frontal_matrix).
export function frontalMatrix(fit, size = 320, radius = 0.34) {
  const h = fit.homography;
  const eps = 1e-3;
  const o = project(h, 0, 0), ex = project(h, eps, 0), ey = project(h, 0, eps);
  const d1 = [ex[0] - o[0], ex[1] - o[1]], d2 = [ey[0] - o[0], ey[1] - o[1]];
  const sign = d1[0] * d2[1] - d1[1] * d2[0] > 0 ? 1 : -1;
  const gap = 360 - fit.sweep;
  const phi0 = ((90 + (sign * gap) / 2) * Math.PI) / 180;
  const r = radius * size, c = size / 2;
  const a = [r * Math.cos(phi0), -sign * r * Math.sin(phi0), c,
    r * Math.sin(phi0), sign * r * Math.cos(phi0), c, 0, 0, 1];
  return mul3(a, inv3(h));
}

// Ultralytics letterbox to a square input (LetterBox(auto=False)): scale, then centred grey pad.
export function letterbox(w, h, size = 640) {
  const r = Math.min(size / h, size / w);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const left = Math.round((size - nw) / 2 - 0.1), top = Math.round((size - nh) / 2 - 0.1);
  return { r, nw, nh, left, top };
}

// The most confident gauge in a pose output (1, 5 + 3K, N): what NMS keeps first.
// Returns null below conf, else {box, conf, kpts: [[x, y, v] x K]} in input-image px
// (clipped to the image, as Ultralytics does).
export function decodeTop(out, n, lb, w, h, conf = 0.1, k = 7) {
  let bi = -1, bc = conf;
  for (let i = 0; i < n; i++) {
    const c = out[4 * n + i];
    if (c >= bc && (bi < 0 || c > bc)) { bc = c; bi = i; }
  }
  if (bi < 0) return null;
  const at = (row) => out[row * n + bi];
  const clip = (v, hi) => Math.min(Math.max(v, 0), hi);
  const ux = (x) => clip((x - lb.left) / lb.r, w), uy = (y) => clip((y - lb.top) / lb.r, h);
  const cx = at(0), cy = at(1), bw = at(2), bh = at(3);
  const box = [ux(cx - bw / 2), uy(cy - bh / 2), ux(cx + bw / 2), uy(cy + bh / 2)];
  const kpts = [];
  for (let j = 0; j < k; j++) kpts.push([ux(at(5 + 3 * j)), uy(at(6 + 3 * j)), at(7 + 3 * j)]);
  return { box, conf: bc, kpts };
}

// Clockwise angle (deg, [0, 360)) from centre->start to centre->end, y down (geometry.py).
export function clockwiseAngle(c, s, e) {
  const a0 = Math.atan2(s[1] - c[1], s[0] - c[0]), a1 = Math.atan2(e[1] - c[1], e[0] - c[0]);
  return ((((a1 - a0) * 180) / Math.PI) % 360 + 360) % 360;
}

export function median(xs) {
  const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
