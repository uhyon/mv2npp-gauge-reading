"""Read gauges in a manifest with a trained pose model and write a predictions CSV.

    uv run python scripts/evaluate.py manifest meterfl --out data/processed/meterfl_manifest.csv
    uv run python scripts/predict_readings.py --weights runs/pose/base/weights/best.pt \\
        --manifest data/processed/meterfl_manifest.csv --out runs/pose/base/meterfl_pred.csv
    uv run python scripts/evaluate.py score meterfl --pred runs/pose/base/meterfl_pred.csv \\
        --out runs/pose/base/meterfl

The scale range comes from the manifest (ground truth): this measures the needle and
keypoint stage only. Rows without a range, or with no gauge found, get a blank value
(declined). The most confident detection is used. Extra columns (keypoints, confidences,
needle fraction, scale arc, model time in ms) are kept for error analysis; evaluate.py reads
only id, value, pred_unit. A scale arc under --min-sweep degrees is declined (bad_geometry)
and the needle fraction is clipped to [-clip, 1+clip]; see gauge_reader.predict.

--pad adds a grey margin around each test image, because the synthetic training images
show the gauge with background around it while MeterFL crops fill the frame (use --pad 0
for synthetic images). --overlay DIR saves each image with the predicted keypoints drawn.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gauge_reader.predict import median_ms, read_rows, write_records


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1, help="minimum box confidence")
    ap.add_argument("--pad", type=float, default=0.15, help="margin per side, share of image size")
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-sweep", type=float, default=90.0,
                    help="decline when the min-to-max scale arc is smaller (degrees)")
    ap.add_argument("--clip", type=float, default=0.1,
                    help="clip the needle fraction to [-clip, 1+clip]; negative = no clipping")
    ap.add_argument("--overlay", type=Path, help="folder for images with predictions drawn")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(str(a.weights))
    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    records, counts = read_rows(model, rows, imgsz=a.imgsz, conf=a.conf, pad=a.pad,
                                device=a.device, overlay=a.overlay, min_sweep=a.min_sweep,
                                clip=None if a.clip < 0 else a.clip, progress=50)
    write_records(records, a.out)
    ms = median_ms(records)
    print(f"{len(records)} rows -> {a.out}  " + ", ".join(f"{k} {v}" for k, v in counts.items() if v)
          + (f"; median model time {ms:.0f} ms/image" if ms is not None else ""))


if __name__ == "__main__":
    main()
