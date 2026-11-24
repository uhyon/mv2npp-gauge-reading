// Page logic of the browser demo: a round-check layout (photo list, step tabs, meter, details,
// results table, CSV) like the desktop app. Reading in reader.js, geometry in geometry.js.
import { GaugeReader } from "./reader.js";
import { frontalMatrix, inv3, project } from "./geometry.js";
import { meterSvg, statusOf, STATUS_COLORS } from "./meter.js";
import { DialOcr } from "./ocr.js";

const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const PARAMS = new URLSearchParams(location.search);
const MODEL_URL = PARAMS.get("model")
  || "https://huggingface.co/uhyon/mv2npp-gauge-pose/resolve/main/best.onnx";
const CUSTOM = "__custom__", AUTO = "__ocr__";
const OCR_BASE = PARAMS.get("ocr") || "ocr";
const $ = (id) => document.getElementById(id);

const T = {
  ko: {
    title: "mv2npp 계기 순시 점검", open: "📷 사진 열기", examples: "🖼 예시 사진",
    readall: "▶ 전체 다시 판독", csv: "💾 보고서 저장 (CSV)", about: "ⓘ 정보", apply: "적용",
    listEmpty: "사진을 열거나 여기로 끌어다 놓으세요. 예시 사진도 있습니다.",
    tabDrawn: "검출 · 키포인트", tabFront: "정면 보정", tabPhoto: "원본",
    viewEmpty: "왼쪽에서 사진을 고르세요.", tag: "계기 태그", fast: "빠른 모드 (회전 검증 없이 1회 판독)",
    modelLoading: "모델 불러오는 중…", modelReady: "모델 v5b (ONNX, 브라우저 CPU)",
    modelFail: "모델을 불러오지 못했습니다: ",
    limits: "합성 데이터로만 학습 · 실제 원전 사진 미검증 · 사진은 업로드되지 않음",
    custom: "미등록 계기: 범위 직접 입력", auto: "미등록 계기: OCR로 범위 자동 판독",
    ocrLoading: "OCR 모델 불러오는 중… (처음 한 번, 약 16 MB)", ocrFail: "OCR 모델을 불러오지 못했습니다: ",
    src: { given: "등록부", ocr: "OCR (문자판 숫자)", custom: "직접 입력" },
    cols: ["사진", "태그", "판독값", "단위", "눈금", "범위 출처", "상태", "회전 차이 %", "시간 ms"],
    keys: ["상태", "눈금 범위", "범위 출처", "경보 범위", "바늘 위치", "회전 판독 차이", "문자판 맞춤", "처리 시간"],
    read: "판독", pending: "판독 중…", waiting: "대기", needScale: "범위 필요",
    needScaleHint: "눈금 범위를 입력하고 적용을 누르세요.",
    st: { ok: "판독", retake: "재촬영: 회전 판독 차이 5% 초과", bad_geometry: "재촬영: 눈금 키포인트 불량",
      no_gauge: "재촬영: 계기 없음", no_scale: "눈금 숫자를 못 읽음: 오른쪽에서 범위를 직접 입력하세요" },
    summary: (n, ok, al, rt) => `점검 ${n}건 · 판독 ${ok} · 경보 ${al} · 재촬영 ${rt}`,
    fastNo: "확인 안 함 (빠른 모드)",
    aboutHtml: `<h3>mv2npp · 아날로그 계기 자동 판독</h3>
<p>원전 현장 순시용 머신비전 연구. 합성 렌더로만 학습한 YOLO11s-pose 모델(v5b)이 계기와 7개 키포인트를 찾고,
문자판을 정면으로 보정해 바늘을 읽습니다. 세 번(정자세, ±25° 회전) 읽어 5% 넘게 다르면 재촬영(RETAKE)을 요청합니다.
모든 계산은 브라우저에서 하며 사진은 업로드되지 않습니다.</p>
<p>등록된 계기(예시 등록부, 가상의 태그)는 등록부의 범위·단위·경보 범위를 쓰고, 미등록 계기는 문자판 숫자를
OCR(PP-OCRv4, RapidOCR의 ONNX 모델)로 읽어 첫 눈금과 끝 눈금을 정합니다. 숫자를 못 읽으면 범위를 직접 입력할 수 있습니다.
실제 원전 사진으로는 검증하지 않았습니다.</p>
<p>코드·결과: <a href="https://github.com/uhyon/mv2npp-gauge-reading" target="_blank">github.com/uhyon/mv2npp-gauge-reading</a> (MIT) ·
모델: <a href="https://huggingface.co/uhyon/mv2npp-gauge-pose" target="_blank">uhyon/mv2npp-gauge-pose</a><br>
예시 사진: 자체 합성 렌더(두 장은 CC BY 2.0 배경 사진 위), Roboflow Universe 사진(CC BY 4.0) ·
<a href="examples/ATTRIBUTION.md" target="_blank">출처</a></p>`,
  },
  en: {
    title: "mv2npp gauge round", open: "📷 Open photos", examples: "🖼 Examples",
    readall: "▶ Read all again", csv: "💾 Save report (CSV)", about: "ⓘ About", apply: "Apply",
    listEmpty: "Open photos or drop them here. Example photos are available too.",
    tabDrawn: "Detection · keypoints", tabFront: "Frontal view", tabPhoto: "Original",
    viewEmpty: "Pick a photo on the left.", tag: "Gauge tag", fast: "Fast mode (one pass, no rotation check)",
    modelLoading: "Loading the model…", modelReady: "Model v5b (ONNX, browser CPU)",
    modelFail: "Could not load the model: ",
    limits: "Trained on synthetic data only · not validated on real plant photos · photos are not uploaded",
    custom: "Unregistered gauge: enter the range", auto: "Unregistered gauge: read the range with OCR",
    ocrLoading: "Loading the OCR models… (once, about 16 MB)", ocrFail: "Could not load the OCR models: ",
    src: { given: "registry", ocr: "OCR (dial numbers)", custom: "entered" },
    cols: ["Photo", "Tag", "Value", "Unit", "Scale", "Range from", "Status", "Spread %", "Time ms"],
    keys: ["Status", "Scale range", "Range from", "Alarm band", "Needle position", "Spread of 3 reads", "Dial fit", "Time"],
    read: "read", pending: "reading…", waiting: "waiting", needScale: "range needed",
    needScaleHint: "Enter the scale range and press Apply.",
    st: { ok: "read", retake: "RETAKE: the three reads differ by more than 5%",
      bad_geometry: "RETAKE: scale keypoints unusable", no_gauge: "RETAKE: no gauge found",
      no_scale: "Dial numbers not read: enter the range on the right" },
    summary: (n, ok, al, rt) => `${n} photos · ${ok} read · ${al} alarms · ${rt} retakes`,
    fastNo: "not checked (fast mode)",
    aboutHtml: `<h3>mv2npp · analog gauge reader</h3>
<p>A machine-vision study for nuclear power plant rounds. A YOLO11s-pose model (version 5b) trained only on
synthetic renders finds the gauge and seven keypoints, the dial is straightened, and the needle is read on its
face. Each photo is read three times (upright and turned ±25°); if the reads differ by more than 5% of the
scale the answer is RETAKE. Everything runs in your browser; photos are not uploaded.</p>
<p>A registered gauge (example registry, made-up tags) takes its range, unit and alarm band from the registry; an
unregistered gauge has its dial numbers read with OCR (PP-OCRv4, the ONNX models of RapidOCR) to find the first and
last scale values. When no numbers are read, the range can be entered by hand. Not validated on real plant photos.</p>
<p>Code and results: <a href="https://github.com/uhyon/mv2npp-gauge-reading" target="_blank">github.com/uhyon/mv2npp-gauge-reading</a> (MIT) ·
model: <a href="https://huggingface.co/uhyon/mv2npp-gauge-pose" target="_blank">uhyon/mv2npp-gauge-pose</a><br>
Example photos: own synthetic renders (two on CC BY 2.0 background photos) and Roboflow Universe photos
(CC BY 4.0) · <a href="examples/ATTRIBUTION.md" target="_blank">credits</a></p>`,
  },
};
let lang = PARAMS.get("lang") === "en" ? "en" : "ko";
const t = () => T[lang];

let reader = null, registry = {}, exampleList = [], ocr = null, ocrLoading = null;
const items = [];
let current = -1, view = "drawn", busy = false;

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

const fmt = (v) => (Number.isFinite(v) ? Number(v.toPrecision(4)).toString() : "–");
// a reading to 1/1000 of the span (finer would claim more than the model reads)
function fmtValue(v, span) {
  const d = Math.max(0, Math.min(6, 3 - Math.floor(Math.log10(Math.abs(span) || 1))));
  return v.toFixed(d);
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// [scale or null (read it with OCR), band, ready]: ready is false for a hand-entered range
// that is not entered yet
function scaleOf(it) {
  const g = registry[it.tag];
  if (g) return [{ min: g.min, max: g.max, unit: g.unit }, g.low != null || g.high != null ? { low: g.low, high: g.high } : null, true];
  if (it.mode === "custom") return [it.custom || null, null, !!it.custom];
  return [null, null, true];
}

function ensureOcr() {
  if (!ocrLoading) {
    $("model").textContent = t().ocrLoading;
    ocrLoading = DialOcr.load(window.ort, OCR_BASE).then((o) => {
      ocr = o; window.dialOcr = o;
      $("model").textContent = t().modelReady + " · OCR";
      return o;
    }).catch((e) => {
      console.error(e);
      $("model").textContent = t().ocrFail + e.message;
      ocrLoading = null;
      return null;
    });
  }
  return ocrLoading;
}

function stateOf(it) {
  if (it.error) return "ERROR";
  if (it.reading) return statusOf(it.reading) || t().read;
  if (!scaleOf(it)[2]) return t().needScale;
  return it.queued ? t().pending : t().waiting;
}

// ---- adding photos
async function addBlob(blob, name, tag = null) {
  const bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  items.push({ name, tag: registry[tag] ? tag : null, mode: "ocr", bitmap, thumb: URL.createObjectURL(blob),
    reading: null, custom: null, queued: false, error: "" });
  return items.length - 1;
}

async function addFiles(files) {
  const first = items.length;
  for (const f of files) if (f.type.startsWith("image/")) {
    const m = f.name.match(/^([^_]+)__/);
    await addBlob(f, f.name, m ? m[1] : null);
  }
  afterAdd(first);
}

async function addExamples() {
  const first = items.length;
  const have = new Set(items.map((it) => it.name));
  for (const ex of exampleList) {
    if (have.has(ex.file)) continue;
    await addBlob(await (await fetch(`examples/${ex.file}`)).blob(), ex.file, ex.tag);
  }
  afterAdd(first);
}

function afterAdd(first) {
  if (items.length === first) return;
  for (let i = first; i < items.length; i++) items[i].queued = scaleOf(items[i])[2];
  renderList(); renderTable();
  if (current < 0) select(first);
  pump();
}

// ---- reading, one photo at a time
async function pump() {
  if (busy || !reader) return;
  const i = items.findIndex((it) => it.queued);
  if (i < 0) return;
  busy = true;
  const it = items[i];
  const [scale, band] = scaleOf(it);
  try {
    await new Promise((r) => setTimeout(r, 10));
    const o = scale ? null : await ensureOcr();
    it.reading = await reader.read(it.bitmap, scale, band, $("fast").checked, o);
    if (it.reading.scaleSource === "given" && it.mode === "custom" && !registry[it.tag]) it.reading.scaleSource = "custom";
    it.reading.fast = $("fast").checked;
    it.error = "";
  } catch (e) {
    console.error(e);
    it.error = e.message;
  }
  it.queued = false;
  busy = false;
  renderList(); renderTable();
  if (i === current) showCurrent();
  pump();
}

function requeue(i) {
  const it = items[i];
  it.reading = null; it.error = "";
  it.queued = scaleOf(it)[2];
  renderList(); renderTable();
  if (i === current) showCurrent();
  pump();
}

// ---- list and table
function valueText(it) {
  const r = it.reading;
  return r && r.value != null ? `${fmtValue(r.value, r.scaleMax - r.scaleMin)} ${r.unit}`.trim() : "–";
}

function colorOf(it) {
  return it.reading ? STATUS_COLORS[statusOf(it.reading)] : "#64748b";
}

function renderList() {
  const box = $("list");
  if (!items.length) { box.innerHTML = `<p class="empty">${t().listEmpty}</p>`; return; }
  box.innerHTML = items.map((it, i) => `<div class="item${i === current ? " active" : ""}" data-i="${i}">
    <img src="${it.thumb}" alt="">
    <div class="meta"><div class="name">${esc(it.tag || "–")} · ${esc(it.name)}</div>
    <div class="val" style="color:${colorOf(it)}"><span class="dot" style="background:${colorOf(it)}"></span>${esc(valueText(it))}</div>
    <div class="name">[${esc(stateOf(it))}]</div></div></div>`).join("");
  box.querySelectorAll(".item").forEach((el) => el.addEventListener("click", () => select(+el.dataset.i)));
}

function renderTable() {
  $("table").tHead.innerHTML = `<tr>${t().cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  $("table").tBodies[0].innerHTML = items.map((it, i) => {
    const r = it.reading, st = stateOf(it);
    const cells = [it.name, it.tag || "–", r && r.value != null ? fmtValue(r.value, r.scaleMax - r.scaleMin) : "–",
      r && r.unit ? r.unit : "–", r && r.scaleMin != null ? `${fmt(r.scaleMin)}–${fmt(r.scaleMax)}` : "–",
      r && r.scaleSource ? t().src[r.scaleSource] : "–", st,
      r && r.spread != null ? r.spread.toFixed(1) : "–",
      r ? Math.round(Object.values(r.ms).reduce((a, b) => a + b, 0)) : "–"];
    return `<tr data-i="${i}">${cells.map((c, j) => `<td class="${j === 6 ? esc(st) : ""}">${esc(c)}</td>`).join("")}</tr>`;
  }).join("");
  $("table").querySelectorAll("tbody tr").forEach((el) => el.addEventListener("click", () => select(+el.dataset.i)));
  const sts = items.map((it) => (it.reading ? statusOf(it.reading) : null));
  $("summary").textContent = items.length ? t().summary(items.length,
    sts.filter((s) => s !== null && s !== "RETAKE").length,
    sts.filter((s) => s === "HIGH" || s === "LOW").length, sts.filter((s) => s === "RETAKE").length) : "";
}

// ---- the selected photo
function select(i) {
  current = i;
  renderList();
  showCurrent();
}

function setTagSelect(it) {
  $("tag").value = it.tag || (it.mode === "custom" ? CUSTOM : AUTO);
  $("custom").hidden = $("tag").value !== CUSTOM;
  if (!it.tag && it.custom) {
    $("cmin").value = it.custom.min; $("cmax").value = it.custom.max; $("cunit").value = it.custom.unit;
  }
}

function showCurrent() {
  const it = items[current];
  if (!it) return;
  setTagSelect(it);
  const r = it.reading;
  drawPhoto($("photo"), it.bitmap);
  if (r) { drawReading($("drawn"), it.bitmap, r); drawFrontal($("frontal"), it.bitmap, r); } else {
    drawPhoto($("drawn"), it.bitmap);
    const f = $("frontal").getContext("2d"); f.fillStyle = "#0f172a"; f.fillRect(0, 0, 420, 420);
  }
  showView();
  const g = registry[it.tag];
  const band = g && (g.low != null || g.high != null) ? { low: g.low, high: g.high } : null;
  $("meter").innerHTML = meterSvg(r, band, r && r.value != null ? fmtValue(r.value, r.scaleMax - r.scaleMin) : "");
  const k = t().keys;
  const src = r && r.scaleSource ? t().src[r.scaleSource]
    + (r.scaleSource === "ocr" && r.labels ? ` · ${r.labels.map((l) => l.text).join(", ")}` : "") : "–";
  const rows = !r ? [[k[0], it.error || (scaleOf(it)[2] ? t().pending : t().needScaleHint)]] : [
    [k[0], r.value != null ? (statusOf(r) || t().read) : (t().st[r.status] || r.status), STATUS_COLORS[statusOf(r)]],
    [k[1], r.scaleMin != null ? `${fmt(r.scaleMin)}–${fmt(r.scaleMax)} ${r.unit}` : "–"],
    [k[2], src],
    [k[3], band ? `${band.low != null ? fmt(band.low) : ""}–${band.high != null ? fmt(band.high) : ""} ${g.unit}` : "–"],
    [k[4], r.fraction != null ? `${(r.fraction * 100).toFixed(1)}%` : "–"],
    [k[5], r.fast ? t().fastNo : r.spread != null ? `${r.spread.toFixed(1)}% FS` : "–"],
    [k[6], r.fit ? `${r.fit.sweep.toFixed(0)}°, rms ${(r.fit.rms * 100).toFixed(1)}%` : "–"],
    [k[7], `${Math.round(Object.values(r.ms).reduce((a, b) => a + b, 0))} ms`
      + (r.ms.ocr ? ` (OCR ${Math.round(r.ms.ocr)})` : "")],
  ];
  $("kv").innerHTML = rows.map(([a, b, c]) => `<div class="card"${rows.length === 1 ? ' style="grid-column:1/-1"' : ""}><b>${a}</b><span${c ? ` style="color:${c}"` : ""}>${esc(b)}</span></div>`).join("");
}

function showView() {
  const has = current >= 0;
  for (const id of ["drawn", "frontal", "photo"]) $(id).hidden = !has || id !== view;
  $("viewhint").hidden = has;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
}

// ---- drawing
function drawPhoto(c, bm) {
  c.width = bm.width; c.height = bm.height;
  c.getContext("2d").drawImage(bm, 0, 0);
}

const COLORS = { NORMAL: "#3caa3c", "": "#3caa3c", LOW: "#f08c28", HIGH: "#dc2828", RETAKE: "#e6a000" };

function drawReading(c, bm, r) {
  drawPhoto(c, bm);
  const g = c.getContext("2d");
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
  if (r.labels && r.labels.length) {       // the dial numbers the range was read from
    g.strokeStyle = "#facc15"; g.lineWidth = lw;
    for (const l of r.labels) {
      g.beginPath();
      l.poly.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.closePath(); g.stroke();
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

function drawFrontal(c, bm, r) {
  const size = 420, g = c.getContext("2d");
  g.fillStyle = "rgb(114,114,114)";
  g.fillRect(0, 0, size, size);
  if (!r.fit) return;
  const m = inv3(frontalMatrix(r.fit, size, 0.34));
  const src = document.createElement("canvas");
  src.width = bm.width; src.height = bm.height;
  const sg = src.getContext("2d", { willReadFrequently: true });
  sg.drawImage(bm, 0, 0);
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

// ---- report, language, inputs
function saveCsv() {
  if (!items.length) return;
  const head = ["photo", "tag", "status", "alarm", "value", "unit", "scale_min", "scale_max", "scale_source",
    "needle_fraction", "spread_pct", "ms"];
  const q = (v) => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  const rows = items.map((it) => {
    const r = it.reading || {};
    return [it.name, it.tag || "", r.status || it.error || "", r.alarm || "",
      r.value != null ? Number(r.value.toPrecision(6)) : "", r.unit || "", r.scaleMin, r.scaleMax, r.scaleSource || "",
      r.fraction != null ? r.fraction.toFixed(4) : "", r.spread != null ? r.spread.toFixed(2) : "",
      r.ms ? Math.round(Object.values(r.ms).reduce((a, b) => a + b, 0)) : ""].map(q).join(",");
  });
  const blob = new Blob(["﻿" + [head.join(","), ...rows].join("\n")], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "gauge_round.csv" });
  a.click();
  URL.revokeObjectURL(a.href);
}

function applyLang() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    if (key === "modelLoading" && reader) { el.textContent = t().modelReady; return; }
    if (t()[key]) el.textContent = t()[key];
  });
  $("lang").textContent = lang === "ko" ? "EN" : "한국어";
  const sel = $("tag"), keep = sel.value;
  sel.innerHTML = "";
  for (const g of Object.values(registry)) sel.append(new Option(`${g.tag}: ${fmt(g.min)}–${fmt(g.max)} ${g.unit}`, g.tag));
  sel.append(new Option(t().auto, AUTO));
  sel.append(new Option(t().custom, CUSTOM));
  if (keep) sel.value = keep;
  renderList(); renderTable(); showCurrent();
}

function setupInputs() {
  $("file").addEventListener("change", (e) => { addFiles([...e.target.files]); e.target.value = ""; });
  $("examples").addEventListener("click", addExamples);
  $("readall").addEventListener("click", () => items.forEach((_, i) => requeue(i)));
  $("csv").addEventListener("click", saveCsv);
  $("about").addEventListener("click", () => { $("dlgbody").innerHTML = t().aboutHtml; $("dlg").showModal(); });
  $("lang").addEventListener("click", () => { lang = lang === "ko" ? "en" : "ko"; applyLang(); });
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => { view = b.dataset.view; showView(); }));
  $("tag").addEventListener("change", () => {
    const it = items[current];
    const v = $("tag").value;
    $("custom").hidden = v !== CUSTOM;
    if (!it) return;
    it.tag = v === CUSTOM || v === AUTO ? null : v;
    it.mode = v === CUSTOM ? "custom" : "ocr";
    requeue(current);
  });
  $("capply").addEventListener("click", () => {
    const it = items[current];
    const min = Number($("cmin").value), max = Number($("cmax").value);
    if (!it || !Number.isFinite(min) || !Number.isFinite(max) || min === max) return;
    it.tag = null; it.mode = "custom";
    it.custom = { min, max, unit: $("cunit").value.trim() };
    requeue(current);
  });
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    addFiles([...e.dataTransfer.files]);
  });
}

async function main() {
  setupInputs();
  try { registry = parseRegistry(await (await fetch("gauges.csv")).text()); } catch { registry = {}; }
  try { exampleList = await (await fetch("examples/index.json")).json(); } catch { exampleList = []; }
  applyLang();
  renderTable();
  showView();
  $("meter").innerHTML = meterSvg(null, null, "");
  const ort = window.ort;
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  addExamples();      // the examples are listed at once and read when the model is ready
  try {
    reader = await GaugeReader.load(ort, MODEL_URL, (got, total) => {
      $("model").textContent = `${t().modelLoading} ${(got / 1e6).toFixed(0)}${total ? ` / ${(total / 1e6).toFixed(0)}` : ""} MB`;
    });
    $("model").textContent = t().modelReady;
    window.gaugeReader = reader;
    window.gaugeItems = items;
    pump();
  } catch (e) {
    console.error(e);
    $("model").textContent = t().modelFail + e.message;
  }
}

main();
