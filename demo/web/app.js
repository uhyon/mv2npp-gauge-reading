// Page logic of the browser demo: model loading, inputs, drawing the reading.
import { GaugeReader } from "./reader.js";
import { frontalMatrix, inv3, project } from "./geometry.js";

const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const MODEL_URL = new URLSearchParams(location.search).get("model")
  || "https://huggingface.co/uhyon/mv2npp-gauge-pose/resolve/main/best.onnx";
const CUSTOM = "__custom__";
const $ = (id) => document.getElementById(id);

let reader = null, photo = null, registry = {};

// gauges.csv: tag,scale_min,scale_max,unit,alarm_low,alarm_high,description ('#' comments)
function parseRegistry(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() && !l.startsWith("#"));
  const head = lines.shift().split(",");
  const out = {};
  for (const line of lines) {
    const cells = line.split(","), row = {};
    head.forEach((h, i) => { row[h.trim()] = (cells[i] || "").trim(); });
    const num = (v) => (v === "" ? null : Number(v));
    out[row.tag] = { tag: row.tag, min: num(row.scale_min), max: num(row.scale_max), unit: row.unit,
      low: num(row.alarm_low), high: num(row.alarm_high), description: row.description };
  }
  return out;
}

function fmt(v) {
  return Number.isFinite(v) ? Number(v.toPrecision(4)).toString() : "–";
}

// a reading to 1/1000 of the span (finer would claim more than the model reads)
function fmtValue(v, span) {
  const d = Math.max(0, Math.min(6, 3 - Math.floor(Math.log10(Math.abs(span) || 1))));
  return v.toFixed(d);
}

function setupTags() {
  const sel = $("tag");
  for (const g of Object.values(registry)) {
    sel.append(new Option(`${g.tag}: ${fmt(g.min)}–${fmt(g.max)} ${g.unit}`, g.tag));
  }
  sel.append(new Option("Other gauge: enter the range", CUSTOM));
  sel.addEventListener("change", showTag);
  showTag();
}

function showTag() {
  const g = registry[$("tag").value];
  $("custom").hidden = !!g;
  if (!g) { $("taginfo").textContent = "Range and unit as printed on the dial."; return; }
  let band = "";
  if (g.low != null && g.high != null) band = ` Normal band ${fmt(g.low)} to ${fmt(g.high)} ${g.unit}.`;
  else if (g.high != null) band = ` HIGH above ${fmt(g.high)} ${g.unit}.`;
  else if (g.low != null) band = ` LOW below ${fmt(g.low)} ${g.unit}.`;
  $("taginfo").textContent = `${g.description}.${band}`;
}

function currentScale() {
  const g = registry[$("tag").value];
  if (g) return [{ min: g.min, max: g.max, unit: g.unit },
    g.low != null || g.high != null ? { low: g.low, high: g.high } : null];
  const min = Number($("cmin").value), max = Number($("cmax").value);
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [null, null];
  return [{ min, max, unit: $("cunit").value.trim() }, null];
}

async function setPhoto(blob, tag) {
  photo = await createImageBitmap(blob, { imageOrientation: "from-image" });
  if (tag && registry[tag]) { $("tag").value = tag; showTag(); }
  const c = $("drawn");
  c.width = photo.width; c.height = photo.height;
  c.getContext("2d").drawImage(photo, 0, 0);
  const f = $("frontal").getContext("2d");
  f.clearRect(0, 0, 320, 320);
  $("summary").innerHTML = '<p class="hint">Ready to read.</p>';
  $("read").disabled = !reader;
  if (reader) await run();
}

async function setupExamples() {
  let list = [];
  try { list = await (await fetch("examples/index.json")).json(); } catch { return; }
  const box = $("examples");
  for (const ex of list) {
    const b = document.createElement("button");
    b.title = ex.tag ? `${ex.file} (${ex.tag})` : ex.file;
    b.innerHTML = `<img src="examples/${ex.file}" alt="${ex.file}" loading="lazy">`;
    b.addEventListener("click", async () => {
      box.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      await setPhoto(await (await fetch(`examples/${ex.file}`)).blob(), ex.tag);
    });
    box.append(b);
  }
}

const COLORS = { NORMAL: "#3caa3c", "": "#3caa3c", LOW: "#f08c28", HIGH: "#dc2828", RETAKE: "#e6a000" };

function drawReading(r) {
  const c = $("drawn"), g = c.getContext("2d");
  c.width = photo.width; c.height = photo.height;
  g.drawImage(photo, 0, 0);
  const s = Math.max(1, Math.max(c.width, c.height) / 640);
  const lw = Math.max(1, Math.round(2 * s)), rad = Math.max(2, Math.round(4 * s));
  const color = COLORS[r.value != null ? r.alarm : "RETAKE"] || COLORS.NORMAL;
  g.lineWidth = lw;
  if (r.box) {
    g.strokeStyle = color;
    const [x0, y0, x1, y1] = r.box;
    g.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
  const k = r.keypoints;
  if (k) {
    g.strokeStyle = "#78c8e6";
    g.beginPath();
    if (r.fit) {
      for (let i = 0; i <= 90; i++) {
        const a = (r.fit.sweep * i / 90) * Math.PI / 180;
        const [x, y] = project(r.fit.homography, Math.cos(a), Math.sin(a));
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      }
    } else {
      [k.scale_min, ...r.arc, k.scale_max].forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    }
    g.stroke();
    g.strokeStyle = "#dc2828";
    g.lineWidth = lw + 1;
    g.beginPath(); g.moveTo(...k.center); g.lineTo(...k.tip); g.stroke();
    g.lineWidth = 1;
    const dot = ([x, y], fill, r0) => {
      g.fillStyle = fill; g.beginPath(); g.arc(x, y, r0, 0, 2 * Math.PI); g.fill();
    };
    r.arc.forEach((p) => dot(p, "#78c8e6", rad));
    for (const [name, fill] of [["scale_min", "#00a0e6"], ["scale_max", "#00a0e6"],
      ["center", "#ffffff"], ["tip", "#dc2828"]]) {
      dot(k[name], fill, rad + 1);
      g.strokeStyle = "#000"; g.beginPath(); g.arc(...k[name], rad + 1, 0, 2 * Math.PI); g.stroke();
    }
  }
  const text = r.value != null ? `${fmtValue(r.value, r.scaleMax - r.scaleMin)} ${r.unit}${r.alarm ? "  " + r.alarm : ""}`.trim()
    : `RETAKE (${r.status})`;
  const fs = Math.round(16 * s);
  g.font = `600 ${fs}px system-ui, sans-serif`;
  const tw = g.measureText(text).width;
  const x = r.box ? Math.max(0, Math.min(r.box[0], c.width - tw - 12)) : 0;
  const y = r.box ? Math.max(fs + 8, r.box[1] - 4) : fs + 8;
  g.fillStyle = color;
  g.fillRect(x, y - fs - 6, tw + 12, fs + 10);
  g.fillStyle = "#fff";
  g.fillText(text, x + 6, y - 2);
}

function drawFrontal(r) {
  const size = 320, c = $("frontal"), g = c.getContext("2d");
  g.fillStyle = "rgb(114,114,114)";
  g.fillRect(0, 0, size, size);
  if (!r.fit) return;
  const m = inv3(frontalMatrix(r.fit, size, 0.34));
  const src = document.createElement("canvas");
  src.width = photo.width; src.height = photo.height;
  const sg = src.getContext("2d", { willReadFrequently: true });
  sg.drawImage(photo, 0, 0);
  const sp = sg.getImageData(0, 0, src.width, src.height).data, W = src.width, H = src.height;
  const out = g.getImageData(0, 0, size, size), op = out.data;
  for (let v = 0; v < size; v++) {
    for (let u = 0; u < size; u++) {
      const [x, y] = project(m, u, v);
      if (!(x >= 0 && y >= 0 && x < W - 1 && y < H - 1)) continue;
      const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
      const i00 = 4 * (y0 * W + x0), i10 = i00 + 4, i01 = i00 + 4 * W, i11 = i01 + 4;
      const o = 4 * (v * size + u);
      for (let ch = 0; ch < 3; ch++) {
        op[o + ch] = (sp[i00 + ch] * (1 - fx) + sp[i10 + ch] * fx) * (1 - fy)
          + (sp[i01 + ch] * (1 - fx) + sp[i11 + ch] * fx) * fy;
      }
      op[o + 3] = 255;
    }
  }
  g.putImageData(out, 0, 0);
}

const STATUS_TEXT = {
  ok: "read", no_gauge: "no gauge found", bad_geometry: "keypoints do not form a usable scale arc",
  retake: "the three readings disagree by more than 5% of scale",
};

function showSummary(r, fast) {
  const ok = r.value != null;
  const badge = ok ? (r.alarm ? `<span class="badge ${r.alarm}">${r.alarm}</span>` : "")
    : '<span class="badge RETAKE">RETAKE</span>';
  const rows = [
    ["Status", STATUS_TEXT[r.status] || r.status],
    ["Scale", ok ? `${fmt(r.scaleMin)} to ${fmt(r.scaleMax)} ${r.unit}` : "–"],
    ["Needle position", r.fraction != null ? `${(r.fraction * 100).toFixed(1)}% of scale` : "–"],
    ["Spread of 3 readings", fast ? "not checked (fast mode)" : r.spread != null ? `${r.spread.toFixed(1)}% of scale` : "–"],
    ["Fitted scale arc", r.fit ? `${r.fit.sweep.toFixed(0)}°, fit error ${(r.fit.rms * 100).toFixed(1)}% of radius` : "–"],
    ["Framing", r.framing ? `${r.framing} (gauge box ${(r.boxShare * 100).toFixed(0)}% of photo)` : "–"],
    ["Detection confidence", r.boxConf != null ? r.boxConf.toFixed(2) : "–"],
    ["Time", Object.entries(r.ms).map(([k, v]) => `${k} ${Math.round(v)} ms`).join(", ")],
  ];
  $("summary").innerHTML = `<div class="value">${ok ? `${fmtValue(r.value, r.scaleMax - r.scaleMin)} ${r.unit}` : "No reading"}${badge}</div>`
    + `<table>${rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join("")}</table>`;
}

async function run() {
  if (!photo || !reader) return;
  const [scale, band] = currentScale();
  if (!scale) { $("status").textContent = "Enter a scale range with min ≠ max."; return; }
  $("read").disabled = true;
  $("status").textContent = "Reading…";
  await new Promise((r) => setTimeout(r, 20));
  try {
    const fast = $("fast").checked;
    const r = await reader.read(photo, scale, band, fast);
    drawReading(r);
    drawFrontal(r);
    showSummary(r, fast);
    window.lastReading = r;
    $("status").textContent = "Done.";
  } catch (e) {
    console.error(e);
    $("status").textContent = `Error: ${e.message}`;
  } finally {
    $("read").disabled = false;
  }
}

function setupInputs() {
  $("file").addEventListener("change", (e) => e.target.files[0] && setPhoto(e.target.files[0]));
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    if (e.dataTransfer.files[0]) setPhoto(e.dataTransfer.files[0]);
  });
  $("read").addEventListener("click", run);
}

async function main() {
  setupInputs();
  try { registry = parseRegistry(await (await fetch("gauges.csv")).text()); } catch { registry = {}; }
  setupTags();
  setupExamples();
  const ort = window.ort;
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  try {
    reader = await GaugeReader.load(ort, MODEL_URL, (got, total) => {
      $("status").textContent = `Loading the model… ${(got / 1e6).toFixed(0)}${total ? ` / ${(total / 1e6).toFixed(0)}` : ""} MB`;
    });
    $("status").textContent = "Model ready.";
    window.gaugeReader = reader;
    if (photo) { $("read").disabled = false; await run(); }
  } catch (e) {
    console.error(e);
    $("status").textContent = `Could not load the model: ${e.message}`;
  }
}

main();
