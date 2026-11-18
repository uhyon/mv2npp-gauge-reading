"""Assemble the Hugging Face Space for the demo into a folder, ready for `hf upload`.

    uv run python scripts/make_space.py --out build/space --weights-repo uhyon/mv2npp-gauge-pose
    hf upload uhyon/mv2npp-gauge-reader build/space . --repo-type space

The Space runs demo/app.py with the src/gauge_reader package, configs/gauges.csv and the
demo examples copied next to it (the same tree as the repository), and downloads the model
(ONNX) from the model repo at start-up, so a new model only needs a new upload there.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cpu
torch
ultralytics>=8.3
onnxruntime>=1.17
rapidocr-onnxruntime>=1.2
opencv-python-headless>=4.8
numpy
pyyaml
pillow
huggingface_hub>=0.23
"""

CARD = """\
---
title: mv2npp gauge reader
emoji: 🧭
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Read analog gauges from photos, trained on synthetic data
---

# Analog gauge reader (mv2npp)

Reads an analog pressure or temperature gauge from a photo: detection and keypoints (YOLO
pose, trained on synthetic renders only), reading on the dial face (frontal correction), a
rotation check that answers RETAKE when unsure, and the scale range from a gauge registry or
OCR. Built as a machine-vision study for nuclear power plant rounds; it has **not** been
evaluated on real plant photos.

Code, results and limitations: https://github.com/uhyon/mv2npp-gauge-reading (MIT).
Example images: own synthetic renders and CC BY 4.0 Roboflow Universe photos
(`demo/examples/ATTRIBUTION.md`).
"""

APP = """\
import os
import runpy

os.environ.setdefault("GAUGE_WEIGHTS_REPO", "{repo}")
os.environ.setdefault("GAUGE_ROOT", os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "app.py"),
               run_name="__main__")
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("build/space"))
    ap.add_argument("--weights-repo", default="uhyon/mv2npp-gauge-pose")
    a = ap.parse_args(argv)
    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    shutil.copytree(ROOT / "src" / "gauge_reader", a.out / "src" / "gauge_reader", ignore=ignore)
    (a.out / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "gauges.csv", a.out / "configs" / "gauges.csv")
    (a.out / "demo").mkdir()
    shutil.copy(ROOT / "demo" / "app.py", a.out / "demo" / "app.py")
    shutil.copytree(ROOT / "demo" / "examples", a.out / "demo" / "examples", ignore=ignore)
    shutil.copy(ROOT / "LICENSE", a.out / "LICENSE")
    (a.out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (a.out / "README.md").write_text(CARD, encoding="utf-8")
    (a.out / "app.py").write_text(APP.format(repo=a.weights_repo), encoding="utf-8")
    n = sum(1 for p in a.out.rglob("*") if p.is_file())
    print(f"{n} files -> {a.out} (weights from {a.weights_repo})")


if __name__ == "__main__":
    main()
