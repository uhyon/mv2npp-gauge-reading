"""Assemble the browser version of the demo (demo/web) as a static Hugging Face Space.

    uv run python scripts/make_static_space.py --out build/static_space
    hf upload uhyon/mv2npp-gauge-reader build/static_space . --repo-type space

A static Space needs no paid hardware: the page runs the ONNX model in the visitor's browser
(onnxruntime-web) and downloads it from the model repo (uhyon/mv2npp-gauge-pose). There is no
OCR in this version, so every example comes with a registry tag (configs/gauges.csv).
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

# examples without a TAG__ prefix (the Gradio demo reads their range with OCR)
EXAMPLE_TAGS = {"roboflow_small.jpg": "PI-401", "roboflow_dim.jpg": "PI-402",
                "roboflow_tilt.jpg": "PI-402", "roboflow_closeup.jpg": "PI-403"}

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
when unsure. The scale range comes from a small example gauge registry or from the visitor; the
full Python version also reads the dial numbers with OCR. Built as a machine-vision study for
nuclear power plant rounds; it has **not** been evaluated on real plant photos.

Code, results and limitations: https://github.com/uhyon/mv2npp-gauge-reading (MIT).
Example images: own synthetic renders (two on CC BY 2.0 background photos) and CC BY 4.0
Roboflow Universe photos (credits in `examples/ATTRIBUTION.md`).
"""


def examples() -> list[dict]:
    out = []
    for p in sorted(EXAMPLES.glob("*.jpg")):
        tag = p.name.split("__")[0] if "__" in p.name else EXAMPLE_TAGS.get(p.name)
        if tag is None:
            raise SystemExit(f"{p.name}: no registry tag (add it to EXAMPLE_TAGS)")
        out.append({"file": p.name, "tag": tag})
    return out


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
    (a.out / "README.md").write_text(CARD, encoding="utf-8")
    print(f"static Space in {a.out}: {sorted(p.name for p in a.out.iterdir())}")


if __name__ == "__main__":
    main()
