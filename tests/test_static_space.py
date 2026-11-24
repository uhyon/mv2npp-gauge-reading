import importlib.util
import json
from pathlib import Path

import pytest

from gauge_reader.registry import load_registry

pytest.importorskip("rapidocr_onnxruntime")

ROOT = Path(__file__).resolve().parents[1]


def test_static_space_build(tmp_path):
    spec = importlib.util.spec_from_file_location("mss", ROOT / "scripts" / "make_static_space.py")
    mss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mss)
    out = tmp_path / "space"
    mss.main(["--out", str(out)])
    for name in ("index.html", "app.js", "reader.js", "geometry.js", "meter.js", "style.css",
                 "gauges.csv",
                 "README.md", "examples/ATTRIBUTION.md"):
        assert (out / name).is_file(), name
    assert "sdk: static" in (out / "README.md").read_text()
    reg = load_registry(out / "gauges.csv")
    examples = json.loads((out / "examples" / "index.json").read_text())
    assert examples and all((e["tag"] is None or e["tag"] in reg)
                            and (out / "examples" / e["file"]).is_file() for e in examples)
    assert any(e["tag"] is None for e in examples)          # read with OCR
    keys = json.loads((out / "ocr" / "keys.json").read_text(encoding="utf-8"))
    assert len(keys) > 6000 and "0" in keys and (out / "ocr" / "det.onnx").is_file()
