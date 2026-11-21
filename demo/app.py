"""Gradio demo: read an analog gauge from a photo.

    uv run python demo/app.py                      # http://127.0.0.1:7860
    GAUGE_WEIGHTS=runs/pose/arc_v5b/weights/best.onnx uv run python demo/app.py

Pick the gauge's tag when it is a registered instrument (configs/gauges.csv): the range,
unit and alarm band come from the registry, and OCR only checks the dial numbers. "Unknown
gauge" reads the range with OCR. Example images: own synthetic renders and CC BY 4.0 Roboflow
photos only (demo/examples/ATTRIBUTION.md).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(os.environ.get("GAUGE_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
from gauge_reader.draw import draw_reading  # noqa: E402
from gauge_reader.reader import GaugeReader  # noqa: E402
from gauge_reader.reading.perspective import frontal_view  # noqa: E402
from gauge_reader.registry import load_registry  # noqa: E402

def _weights() -> Path:
    """GAUGE_WEIGHTS (a local file), else GAUGE_WEIGHTS_REPO (a Hugging Face model repo, file
    GAUGE_WEIGHTS_FILE, default best.onnx), else weights/best.onnx."""
    if os.environ.get("GAUGE_WEIGHTS"):
        return Path(os.environ["GAUGE_WEIGHTS"])
    if os.environ.get("GAUGE_WEIGHTS_REPO"):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(os.environ["GAUGE_WEIGHTS_REPO"],
                                    os.environ.get("GAUGE_WEIGHTS_FILE", "best.onnx")))
    return ROOT / "weights" / "best.onnx"


WEIGHTS = _weights()
REGISTRY = load_registry(ROOT / "configs" / "gauges.csv")
UNKNOWN = "Unknown gauge (read the range with OCR)"
EXAMPLES = ROOT / "demo" / "examples"

_readers: dict[tuple[bool, bool], GaugeReader] = {}


def reader(fast: bool, check: bool) -> GaugeReader:
    """One reader per mode, sharing the loaded model."""
    key = (fast, check)
    if key not in _readers:
        base = next(iter(_readers.values()), None)
        _readers[key] = GaugeReader(
            WEIGHTS if base is None else None, model=None if base is None else base.model,
            ocr=True if base is None else base.ocr, tta_angle=None if fast else 25.0,
            check_scale=check)
    return _readers[key]


def tag_choices() -> list[str]:
    return [UNKNOWN] + [f"{t}: {i.scale_min:g}–{i.scale_max:g} {i.unit} ({i.description})"
                        for t, i in REGISTRY.items()]


def run(image: np.ndarray | None, tag_choice: str, fast: bool, check: bool):
    if image is None:
        return None, None, "Upload a photo or pick an example."
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    tag = tag_choice.split(":")[0] if tag_choice and tag_choice != UNKNOWN else None
    inst = REGISTRY.get(tag) if tag else None
    r = reader(fast, check and inst is not None).read(
        bgr, scale=inst.scale if inst else None, band=inst.band if inst else None)
    drawn = cv2.cvtColor(draw_reading(bgr, r), cv2.COLOR_BGR2RGB)
    front = None
    if r.fit is not None:
        front = cv2.cvtColor(frontal_view(bgr, r.fit, size=360), cv2.COLOR_BGR2RGB)
    return drawn, front, summary(r, inst)


STATUS_TEXT = {
    "ok": "read",
    "retake": "RETAKE: the upright and turned readings disagree by more than 5% of the scale",
    "bad_geometry": "RETAKE: the scale keypoints do not form a usable dial",
    "no_gauge": "RETAKE: no gauge found",
    "no_scale": "RETAKE: no scale range (register the gauge or take a sharper photo)",
}


def summary(r, inst) -> str:
    lines = []
    if r.value is not None:
        unit = r.unit or "(unit not read)"
        lines.append(f"## {r.value:.4g} {unit}" + (f"  ·  **{r.alarm}**" if r.alarm else ""))
    else:
        lines.append("## RETAKE")
    lines.append(f"- Status: {STATUS_TEXT.get(r.status, r.status)}")
    if r.scale_min is not None:
        src = f"registry ({inst.tag})" if r.scale_source == "given" and inst else "OCR"
        lines.append(f"- Scale: {r.scale_min:g}–{r.scale_max:g} {r.unit}, from {src}")
    if inst is not None and inst.band is not None:
        lines.append(f"- Alarm band: {inst.band.low:g}–{inst.band.high:g} {inst.unit}")
    if r.scale_check:
        lines.append(f"- OCR check of the registered scale: **{r.scale_check}**"
                     + (" (is this the right gauge?)" if r.scale_check == "mismatch" else ""))
    if r.fraction is not None:
        lines.append(f"- Needle at {r.fraction * 100:.1f}% of the scale"
                     + (f", spread of the turned readings {r.spread:.1f}%" if r.spread is not None else ""))
    if r.ms:
        lines.append("- Time: " + ", ".join(f"{k} {v:.0f} ms" for k, v in r.ms.items())
                     + f" (total {sum(r.ms.values()):.0f} ms, CPU)")
    return "\n".join(lines)


def examples() -> list[list]:
    if not EXAMPLES.is_dir():
        return []
    rows = []
    for p in sorted(EXAMPLES.glob("*.jpg")) + sorted(EXAMPLES.glob("*.png")):
        tag = p.stem.split("__")[0] if "__" in p.stem else None
        choice = next((c for c in tag_choices() if tag and c.startswith(tag + ":")), UNKNOWN)
        rows.append([str(p), choice, False, False])
    return rows


with gr.Blocks(title="Analog gauge reader") as app:
    gr.Markdown(
        "# Analog gauge reader\n"
        "Machine vision for reading analog gauges in nuclear power plant rounds, trained on "
        "synthetic renders only. Detection and keypoints (YOLO pose), reading on the dial face "
        "(frontal correction), rotation check with RETAKE, scale range from a gauge registry "
        "or OCR. Not validated on real plant photos.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Photo")
            tag = gr.Dropdown(tag_choices(), value=UNKNOWN, label="Gauge")
            fast = gr.Checkbox(False, label="Fast mode (one pass, no rotation check)")
            check = gr.Checkbox(True, label="Check the registered scale with OCR")
            btn = gr.Button("Read", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Detection and keypoints")
            out_txt = gr.Markdown()
            out_front = gr.Image(label="Dial seen from the front")
    btn.click(run, [inp, tag, fast, check], [out_img, out_front, out_txt])
    ex = examples()
    if ex:
        gr.Examples(ex, [inp, tag, fast, check], [out_img, out_front, out_txt], fn=run,
                    cache_examples=False)
    gr.Markdown("Example images: own synthetic renders and CC BY 4.0 Roboflow Universe photos; "
                "see demo/examples/ATTRIBUTION.md.")

if __name__ == "__main__":
    app.launch()
