"""Scan the background photos for gauges that slipped through the manual review.

An unlabelled gauge in a background teaches the detector that gauges are background. Two
detectors are run, because each misses what the other catches:

- a COCO-pretrained YOLO (never saw these photos): round dials often fire as "clock";
- optionally our gauge model (--weights): it knows gauges, but it was trained on these very
  photos as "no gauge", so it may have learned to ignore exactly the hidden ones.

    uv run python scripts/scan_backgrounds.py                         # COCO "clock" only
    uv run python scripts/scan_backgrounds.py --weights runs/pose/base/weights/best.pt

Writes <out>/hits.csv (one row per detection) and <out>/sheet_*.jpg: each hit cropped with
some context and labelled with file name, model and confidence. Look at the sheets, open the
listed photos at full size, and delete the ones with a real gauge. Then, if any were deleted,
find the affected synthetic images through the `background` field of labels.jsonl
(scripts/generate_synth.py --backfill-backgrounds adds it to older sets).

Runs on CPU in a few minutes for about 300 photos. Needs the train extra.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gauge_reader.synth.scene import list_backgrounds

COCO_CLOCK = 74


def crop_tile(img: np.ndarray, box, label: str, size: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * 2.0 + 16           # the hit plus the same again as context
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a0, b0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    a1, b1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    tile = img[b0:b1, a0:a1].copy()
    cv2.rectangle(tile, (int(x0) - a0, int(y0) - b0), (int(x1) - a0, int(y1) - b0), (0, 0, 255), 2)
    s = size / max(tile.shape[:2])
    tile = cv2.resize(tile, (max(1, round(tile.shape[1] * s)), max(1, round(tile.shape[0] * s))))
    out = np.full((size + 34, size, 3), 30, np.uint8)
    out[:tile.shape[0], :tile.shape[1]] = tile
    for k, line in enumerate(label.split("\n")[:2]):
        cv2.putText(out, line, (3, size + 13 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
    return out


def write_sheets(tiles: list[np.ndarray], out: Path, cols: int = 6, rows: int = 5) -> int:
    per = cols * rows
    n = 0
    for start in range(0, len(tiles), per):
        chunk = tiles[start:start + per]
        th, tw = chunk[0].shape[:2]
        sheet = np.zeros((th * ((len(chunk) + cols - 1) // cols), tw * cols, 3), np.uint8)
        for j, t in enumerate(chunk):
            r, c = divmod(j, cols)
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
        n += 1
        cv2.imwrite(str(out / f"sheet_{n:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backgrounds", default="assets/backgrounds")
    ap.add_argument("--coco", default="yolo11m.pt", help="COCO-pretrained detector")
    ap.add_argument("--coco-conf", type=float, default=0.1)
    ap.add_argument("--weights", type=Path, help="our gauge model (optional second detector)")
    ap.add_argument("--conf", type=float, default=0.1, help="confidence for --weights")
    ap.add_argument("--imgsz", type=int, default=1280, help="large, so small far gauges show")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("assets/backgrounds_scan"))
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    paths = list_backgrounds(a.backgrounds)
    if not paths:
        raise SystemExit(f"no photos in {a.backgrounds}")
    models = [("coco_clock", YOLO(a.coco), a.coco_conf, [COCO_CLOCK])]
    if a.weights:
        models.append(("gauge", YOLO(str(a.weights)), a.conf, None))

    a.out.mkdir(parents=True, exist_ok=True)
    hits, tiles = [], []
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"unreadable: {p}")
            continue
        for name, model, conf, classes in models:
            res = model.predict(img, imgsz=a.imgsz, conf=conf, classes=classes, device=a.device,
                                verbose=False)[0]
            for box, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                x0, y0, x1, y1 = map(float, box)
                hits.append({"file": Path(p).name, "model": name, "conf": f"{c:.3f}",
                             "x0": round(x0), "y0": round(y0), "x1": round(x1), "y1": round(y1),
                             "size_px": round(max(x1 - x0, y1 - y0))})
                tiles.append((float(c), crop_tile(img, (x0, y0, x1, y1),
                                                  f"{Path(p).name[:34]}\n{name} {c:.2f}")))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(paths)} photos, {len(hits)} hits")

    hits.sort(key=lambda h: -float(h["conf"]))
    with open(a.out / "hits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "model", "conf", "x0", "y0", "x1", "y1", "size_px"])
        w.writeheader()
        w.writerows(hits)
    tiles.sort(key=lambda t: -t[0])
    n = write_sheets([t for _, t in tiles], a.out) if tiles else 0
    files = len({h["file"] for h in hits})
    print(f"{len(paths)} photos scanned: {len(hits)} hits in {files} photos -> {a.out}/hits.csv, "
          f"{n} sheets")


if __name__ == "__main__":
    main()
