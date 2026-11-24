"""Assemble the browser version of the demo (demo/web) as a static Hugging Face Space.

    uv run python scripts/make_static_space.py --out build/static_space
    hf upload uhyon/mv2npp-gauge-reader build/static_space . --repo-type space

A static Space needs no paid hardware: the page runs the ONNX model in the visitor's browser
(onnxruntime-web) and downloads it from the model repo (uhyon/mv2npp-gauge-pose). The scale range
of an unregistered gauge is read from the dial numbers with OCR in the browser (demo/web/ocr.js):
the PP-OCRv4 detection and recognition models that rapidocr-onnxruntime ships are copied from the
installed package into ocr/ (install the ``ocr`` extra first). Examples named TAG__*.jpg use the
registry (configs/gauges.csv); the others read their range with OCR.
Test locally with ``python -m http.server -d build/static_space`` and open
``http://localhost:8000/?model=<url of best.onnx>`` to use another model file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "demo" / "web"
EXAMPLES = ROOT / "demo" / "examples"

# PP-OCRv4 models of rapidocr-onnxruntime (Apache-2.0, from PaddleOCR) -> ocr/
OCR_MODELS = {"det.onnx": "ch_PP-OCRv4_det_infer.onnx", "rec.onnx": "ch_PP-OCRv4_rec_infer.onnx"}

CARD = """\
---
title: mv2npp gauge reader
emoji: 🧭
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Read analog gauges from photos, in the browser
models:
- uhyon/mv2npp-gauge-pose
---

# Analog gauge reader (mv2npp), browser version

Reads an analog pressure or temperature gauge from a photo, entirely in the browser
(onnxruntime-web): detection and seven keypoints (YOLO11s-pose, trained on synthetic renders
only), reading on the dial face (frontal correction), and a rotation check that answers RETAKE
when unsure. The scale range comes from a small example gauge registry or, for an unregistered
gauge, from the dial numbers read with OCR in the browser (PP-OCRv4 models of
[RapidOCR](https://github.com/RapidAI/RapidOCR), Apache-2.0, from
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)); it can also be entered by hand. Built as
a machine-vision study for nuclear power plant rounds; it has **not** been evaluated on real plant
photos.

Code, results and limitations: https://github.com/uhyon/mv2npp-gauge-reading (MIT).
Example images: own synthetic renders (two on CC BY 2.0 background photos) and CC BY 4.0
Roboflow Universe photos (credits in `examples/ATTRIBUTION.md`).
"""


def examples() -> list[dict]:
    out = []
    for p in sorted(EXAMPLES.glob("*.jpg")):
        tag = p.name.split("__")[0] if "__" in p.name else None
        out.append({"file": p.name, "tag": tag})
    return out


def copy_ocr(out: Path) -> None:
    """The OCR models and their character list (stored in the recognition model's metadata)."""
    import onnxruntime as ort
    import rapidocr_onnxruntime
    models = Path(rapidocr_onnxruntime.__file__).parent / "models"
    out.mkdir(parents=True, exist_ok=True)
    for dst, src in OCR_MODELS.items():
        shutil.copy(models / src, out / dst)
    meta = ort.InferenceSession(str(out / "rec.onnx")).get_modelmeta().custom_metadata_map
    keys = meta["character"].splitlines()
    (out / "keys.json").write_text(json.dumps(keys, ensure_ascii=False), encoding="utf-8")
    (out / "NOTICE.txt").write_text(
        "PP-OCRv4 text detection and recognition models (ch_PP-OCRv4_det_infer, "
        "ch_PP-OCRv4_rec_infer) from PaddleOCR, converted to ONNX and shipped by RapidOCR "
        "(rapidocr-onnxruntime). Apache License 2.0: https://github.com/PaddlePaddle/PaddleOCR, "
        "https://github.com/RapidAI/RapidOCR\n", encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("build/static_space"))
    a = ap.parse_args(argv)
    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    for p in WEB.iterdir():
        if p.suffix in {".html", ".js", ".css"}:
            shutil.copy(p, a.out / p.name)
    shutil.copy(ROOT / "configs" / "gauges.csv", a.out / "gauges.csv")
    shutil.copytree(EXAMPLES, a.out / "examples", ignore=shutil.ignore_patterns("*.py*"))
    (a.out / "examples" / "index.json").write_text(json.dumps(examples(), indent=1) + "\n")
    shutil.copy(ROOT / "LICENSE", a.out / "LICENSE")
    copy_ocr(a.out / "ocr")
    (a.out / "README.md").write_text(CARD, encoding="utf-8")
    print(f"static Space in {a.out}: {sorted(p.name for p in a.out.iterdir())}")


if __name__ == "__main__":
    main()
