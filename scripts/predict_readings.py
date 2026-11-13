"""Read gauges in a manifest with a trained pose model and write a predictions CSV.

    uv run python scripts/evaluate.py manifest meterfl --out data/processed/meterfl_manifest.csv
    uv run python scripts/predict_readings.py --weights runs/pose/base/weights/best.pt \\
        --manifest data/processed/meterfl_manifest.csv --out runs/pose/base/meterfl_pred.csv
    uv run python scripts/evaluate.py score meterfl --pred runs/pose/base/meterfl_pred.csv \\
        --out runs/pose/base/meterfl

The scale range comes from the manifest (ground truth): this measures the needle and
keypoint stage only. Rows without a range, or with no gauge found, get a blank value
(declined). The most confident detection is used. Extra columns (keypoints, confidences,
needle fraction) are kept for error analysis; evaluate.py reads only id, value, pred_unit.

--pad adds a grey margin around each test image, because the synthetic training images
show the gauge with background around it while MeterFL crops fill the frame.
--overlay DIR saves each image with the predicted keypoints drawn on it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gauge_reader.keypoints.yolo import KP_NAMES, detections_from_result, pad_image, shift

COLORS = {"center": (255, 255, 0), "tip": (0, 0, 255), "scale_min": (255, 0, 255),
          "scale_max": (0, 165, 255)}


def _float(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def draw(img: np.ndarray, det, text: str) -> np.ndarray:
    out = img.copy()
    s = max(1, round(max(img.shape[:2]) / 300))
    x0, y0, x1, y1 = map(int, det.box)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), s)
    c = tuple(map(int, det.keypoints.center))
    cv2.line(out, c, tuple(map(int, det.keypoints.tip)), (0, 0, 255), s)
    for name in KP_NAMES:
        p = tuple(map(int, getattr(det.keypoints, name)))
        cv2.circle(out, p, 3 * s, COLORS[name], -1)
    cv2.putText(out, text, (4, 14 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (0, 255, 0), s)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1, help="minimum box confidence")
    ap.add_argument("--pad", type=float, default=0.15, help="margin per side, share of image size")
    ap.add_argument("--device", default=None)
    ap.add_argument("--overlay", type=Path, help="folder for images with predictions drawn")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(str(a.weights))
    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if a.overlay:
        a.overlay.mkdir(parents=True, exist_ok=True)

    fields = ["id", "value", "pred_unit", "box_conf", "min_kpt_conf", "fraction", "n_det",
              "status"] + [f"{k}_{c}" for k in KP_NAMES for c in "xy"]
    out = []
    counts = {"ok": 0, "no_range": 0, "no_detection": 0, "unreadable": 0, "bad_geometry": 0}
    for i, row in enumerate(rows):
        rec = {"id": row["id"], "value": "", "pred_unit": "", "status": "ok"}
        lo, hi = _float(row.get("scale_min")), _float(row.get("scale_max"))
        img = cv2.imread(row["image"], cv2.IMREAD_COLOR)
        if img is None:
            rec["status"] = "unreadable"
        elif lo is None or hi is None:
            rec["status"] = "no_range"
        else:
            padded, dx, dy = pad_image(img, a.pad)
            res = model.predict(padded, imgsz=a.imgsz, conf=a.conf, device=a.device,
                                verbose=False)[0]
            dets = [shift(d, dx, dy) for d in detections_from_result(res)]
            rec["n_det"] = len(dets)
            if not dets:
                rec["status"] = "no_detection"
            else:
                d = dets[0]
                rec.update(box_conf=f"{d.box_conf:.4f}", min_kpt_conf=f"{d.min_kpt_conf:.4f}")
                for k in KP_NAMES:
                    x, y = getattr(d.keypoints, k)
                    rec[f"{k}_x"], rec[f"{k}_y"] = f"{x:.1f}", f"{y:.1f}"
                try:
                    frac = d.fraction()
                    value = lo + frac * (hi - lo)
                    rec.update(value=f"{value:.6g}", pred_unit=row.get("unit", ""),
                               fraction=f"{frac:.4f}")
                except ValueError:  # min and max keypoints in the same direction
                    rec["status"] = "bad_geometry"
                if a.overlay:
                    text = f"{rec['value'] or '?'} {row.get('unit', '')} conf {d.box_conf:.2f}"
                    cv2.imwrite(str(a.overlay / f"{row['id']}.jpg"), draw(img, d, text))
        counts[rec["status"]] += 1
        out.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} rows -> {a.out}  " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))


if __name__ == "__main__":
    main()
