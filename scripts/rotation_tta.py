"""Read every image upright and turned by +/- --angle degrees; keep the spread as a RETAKE signal.

    uv run python scripts/rotation_tta.py --weights runs/pose/arc/weights/best.pt \\
        --manifest data/processed/meterfl_manifest.csv --out runs/pose/arc/meterfl_tta.csv
    uv run python scripts/evaluate.py score meterfl --pred runs/pose/arc/meterfl_tta.csv \\
        --out runs/pose/arc/meterfl_tta

A reading does not change when the photo is turned, so the three readings should agree. When
the model is unsure of the dial (a dial turned far from upright, an unusual design) they
drift apart. Output columns: ``value`` = the median of the three readings (what
evaluate.py scores), ``value_0`` the upright reading, ``frac_0 / frac_p / frac_m`` the needle
fractions (clipped like the readings), ``spread`` = max - min fraction in % of scale (99 when
fewer than two readings exist), ``status`` of the upright pass. Other arguments as in
predict_readings.py; synthetic images need --pad 0.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gauge_reader.predict import read_rows


def rotate(angle: float):
    def f(img, row):
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, m, (w, h), borderValue=(114, 114, 114))
    return f


def combine(rows: list[dict], passes: list[list[dict]], clip: float = 0.1) -> list[dict]:
    """Merge the upright, +angle and -angle passes (records in manifest order)."""
    out = []
    for row, recs in zip(rows, zip(*passes)):
        fr = []
        for r in recs:
            try:
                fr.append(min(max(float(r["fraction"]), -clip), 1 + clip))
            except (KeyError, ValueError, TypeError):
                fr.append(None)
        got = [f for f in fr if f is not None]
        rec = {"id": row["id"], "status": recs[0]["status"], "pred_unit": "", "value": "",
               "value_0": recs[0].get("value", "")}
        for k, f in zip(("frac_0", "frac_p", "frac_m"), fr):
            rec[k] = "" if f is None else f"{f:.4f}"
        rec["spread"] = f"{(max(got) - min(got)) * 100:.2f}" if len(got) >= 2 else "99"
        try:
            lo, hi = float(row["scale_min"]), float(row["scale_max"])
        except (KeyError, ValueError, TypeError):
            lo = hi = None
        if got and lo is not None:
            rec["value"] = f"{lo + float(np.median(got)) * (hi - lo):.6g}"
            rec["pred_unit"] = row.get("unit", "")
        out.append(rec)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--angle", type=float, default=25.0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(str(a.weights))
    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    passes = []
    for ang in (0.0, a.angle, -a.angle):
        recs, _ = read_rows(model, rows, imgsz=a.imgsz, conf=a.conf, pad=a.pad, device=a.device,
                            transform=rotate(ang) if ang else None)
        passes.append(recs)
        print(f"  pass {ang:+.0f} deg done")
    out = combine(rows, passes)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} rows -> {a.out}")


if __name__ == "__main__":
    main()
