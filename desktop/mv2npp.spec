# PyInstaller spec of the Windows desktop app: uv run pyinstaller desktop/mv2npp.spec
# Bundles ONNX Runtime, OpenCV, PySide6 (Essentials), RapidOCR and weights/best.onnx;
# no PyTorch or Ultralytics (gauge_reader.keypoints.onnx_pose). See desktop/README.md.
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
ocr_datas, ocr_bins, ocr_hidden = collect_all("rapidocr_onnxruntime")
datas = [
    (str(ROOT / "configs" / "gauges.csv"), "configs"),
    (str(ROOT / "weights" / "best.onnx"), "weights"),
    (str(ROOT / "LICENSE"), "."),
] + [(str(p), "demo/examples") for p in (ROOT / "demo" / "examples").iterdir()] + ocr_datas

a = Analysis(
    [str(ROOT / "desktop" / "app.py")],
    pathex=[str(ROOT / "src"), str(ROOT / "desktop")],
    datas=datas, binaries=ocr_bins,
    hiddenimports=ocr_hidden + ["gauge_reader.keypoints.onnx_pose", "gauge_reader.reader"],
    excludes=["torch", "torchvision", "ultralytics", "matplotlib", "pandas", "scipy",
              "tkinter", "IPython", "gradio", "polars"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="mv2npp-gauge-reader",
          console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="mv2npp-gauge-reader", upx=False)
