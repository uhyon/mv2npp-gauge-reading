"""Read the scale range and unit with OCR, using the keypoints of a predictions CSV.

    uv run python scripts/predict_readings.py --weights runs/pose/arc/weights/best.pt \\
        --manifest data/processed/meterfl_manifest.csv --out runs/pose/arc/meterfl_pred.csv
    uv run python scripts/ocr_scale.py --manifest data/processed/meterfl_manifest.csv \\
        --pred runs/pose/arc/meterfl_pred.csv --out runs/pose/arc/meterfl_ocr.csv
    uv run python scripts/evaluate.py score meterfl --pred runs/pose/arc/meterfl_ocr.csv \\
        --out runs/pose/arc/meterfl_ocr

Writes a predictions CSV whose ``value`` uses the OCR range (min + fraction x span) and whose
``pred_unit`` is the OCR unit (blank when none was read; evaluate.py then assumes the label
unit), so evaluate.py scores the full reading. Columns ``ocr_min``, ``ocr_max``, ``ocr_unit``,
``n_labels`` (numbers found on the dial), ``n_inliers`` (on the fitted line), ``ocr_rms``,
``ocr_status`` (ok / no_keypoints / no_scale), ``ocr_ms`` and, when the manifest has a range,
``range_ok`` (both ends within 1% of the true span) and ``unit_ok`` are added. A summary of the
range and unit accuracy is printed. See gauge_reader.reading.scale_ocr. With predictions made
by ``predict_readings.py --tta`` the needle fraction is the median of the three turned readings
and rows whose spread is over --retake-spread (or with status retake) are declined, also those
without a range in the manifest (predict_readings keeps their keypoints, status no_range).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gauge_reader.keypoints.yolo import ARC_NAMES, KP_NAMES, detections_from_arrays
from gauge_reader.reading.scale_ocr import RapidOcr, dial_crop, read_scale
from gauge_reader.units import convert, norm_unit


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def detection_from_pred(p: dict):
    names = list(KP_NAMES) + [n for n in ARC_NAMES if _f(p.get(f"{n}_x")) is not None]
    pts = [(_f(p.get(f"{n}_x")), _f(p.get(f"{n}_y"))) for n in names]
    if any(v is None for xy in pts for v in xy):
        return None
    (d,) = detections_from_arrays(np.zeros((1, 4)), np.ones(1), np.array([pts]))
    return d


def same_unit(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if norm_unit(a) == norm_unit(b):
        return True
    c = convert(1.0, a, b)
    return c is not None and abs(c - 1.0) < 1e-6


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True, help="predictions CSV with keypoint columns")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--retake-spread", type=float, default=5.0,
                    help="with --tta predictions: decline above this spread (%% of scale)")
    ap.add_argument("--crop", type=float, default=1.6,
                    help="OCR only the dial: this many scale radii around the centre (0 = whole image)")
    a = ap.parse_args(argv)

    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with open(a.pred, newline="", encoding="utf-8-sig") as f:
        preds = {r["id"]: r for r in csv.DictReader(f)}
    ocr = RapidOcr()
    out = []
    for i, row in enumerate(rows):
        p = preds.get(row["id"], {})
        rec = {"id": row["id"], "value": "", "pred_unit": "", "ocr_status": "no_keypoints"}
        det = detection_from_pred(p)
        frac = _f(p.get("fraction"))
        tta = [v for v in (_f(p.get(k)) for k in ("frac_0", "frac_p", "frac_m")) if v is not None]
        if tta:                      # predictions made with --tta: median of the turned readings
            frac = float(np.median(tta))
        spread = _f(p.get("spread"))
        retake = p.get("status") == "retake" or (
            spread is not None and a.retake_spread >= 0 and spread > a.retake_spread)
        if retake:
            frac = None
            rec["retake"] = 1
        if det is not None:
            img = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
            t0 = time.perf_counter()
            boxes = []
            if img is not None:
                crop, (dx, dy) = dial_crop(img, det.keypoints, a.crop)
                boxes = [(t, [(x + dx, y + dy) for x, y in poly], c) for t, poly, c in ocr(crop)]
            rec["ocr_ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
            s = read_scale(boxes, det.keypoints, det.dial_fit())
            rec.update(n_labels=len(s.labels), n_inliers=len(s.inliers), ocr_unit=s.unit,
                       ocr_status="ok" if s.ok else "no_scale",
                       ocr_texts=" | ".join(t for t, _, _ in boxes))
            if s.ok:
                rec.update(ocr_min=f"{s.scale_min:.6g}", ocr_max=f"{s.scale_max:.6g}",
                           ocr_rms=f"{s.rms:.4f}")
                if frac is not None:
                    fc = min(max(frac, -a.clip), 1 + a.clip)
                    rec["value"] = f"{s.scale_min + fc * (s.scale_max - s.scale_min):.6g}"
                    rec["pred_unit"] = s.unit
        lo, hi = _f(row.get("scale_min")), _f(row.get("scale_max"))
        if lo is not None and hi is not None:
            span = abs(hi - lo)
            mn, mx = _f(rec.get("ocr_min")), _f(rec.get("ocr_max"))
            rec["range_ok"] = int(mn is not None and abs(mn - lo) <= 0.01 * span
                                  and abs(mx - hi) <= 0.01 * span)
        rec["unit_ok"] = int(same_unit(rec.get("ocr_unit", ""), row.get("unit", "")))
        out.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}")
    keys = list(dict.fromkeys(k for r in out for k in r))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    n = len(out)
    st = {k: sum(r["ocr_status"] == k for r in out) for k in ("ok", "no_scale", "no_keypoints")}
    rng = [r["range_ok"] for r in out if "range_ok" in r]
    ms = sorted(float(r["ocr_ms"]) for r in out if r.get("ocr_ms"))
    print(f"{n} rows -> {a.out}; status {st}; range correct {sum(rng)}/{len(rng)} "
          f"({100 * sum(rng) / max(len(rng), 1):.0f}%); unit correct {sum(r['unit_ok'] for r in out)}/{n}; "
          f"unit read {sum(bool(r.get('ocr_unit')) for r in out)}/{n}"
          + (f"; median OCR time {ms[len(ms) // 2]:.0f} ms" if ms else ""))


if __name__ == "__main__":
    main()
