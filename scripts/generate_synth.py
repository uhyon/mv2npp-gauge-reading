"""Generate the 2D synthetic gauge dataset.

    python scripts/generate_synth.py --config configs/synth_2d.yaml --out data/synthetic/synth_2d
    python scripts/generate_synth.py --num 200 --out data/synthetic/try    # quick look
    python scripts/generate_synth.py --num 24 --preview preview.jpg        # contact sheet only

Output layout (one folder per split, YOLO-pose compatible):

    <out>/images/{train,val}/000000.jpg
    <out>/labels/{train,val}/000000.txt   class cx cy w h  x y v  (x4: center, tip, min, max)
    <out>/labels.jsonl                    full ground truth per image (value, range, unit,
                                          keypoints, homography, conditions, camera)
    <out>/data.yaml                       Ultralytics dataset file (kpt_shape [4, 3])

Image i is generated from seed (config seed + i), so any image can be regenerated alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gauge_reader.synth.scene import list_backgrounds, make_sample

KP_NAMES = ["center", "tip", "scale_min", "scale_max"]

_CFG: dict = {}
_BGS: list[str] = []


def _init(cfg, bgs):
    global _CFG, _BGS
    _CFG, _BGS = cfg, bgs


def sample_for(i: int, cfg: dict, bgs: list[str]):
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + i)
    return make_sample(rng, cfg, bgs)


def label_record(i: int, split: str, s) -> dict:
    sp = s.spec
    return {
        "id": f"{i:06d}", "split": split, "file": f"images/{split}/{i:06d}.jpg",
        "value": s.value, "min_value": sp.min_value, "max_value": sp.max_value, "unit": sp.unit,
        "sweep_deg": sp.sweep_deg, "major_step": sp.major_step, "minor_per_major": sp.minor_per_major,
        "second_scale": list(sp.second_scale) if sp.second_scale else None,
        "keypoints": {k: list(s.keypoints[k]) for k in KP_NAMES},
        "bbox_xyxy": list(s.bbox),
        "homography": np.asarray(s.homography).round(6).tolist(),
        "dial_keypoints": {k: list(s.dial_keypoints[k]) for k in KP_NAMES},
        "dial_size": s.dial_size,
        "conditions": s.conditions,
        "camera": {k: round(float(v), 3) for k, v in s.camera.items()},
        "style": {"needle": sp.needle_style, "face": list(sp.face_color),
                  "font": os.path.basename(sp.font_path) if sp.font_path else None},
    }


def yolo_line(s, size: int) -> str:
    x0, y0, x1, y1 = s.bbox
    parts = [0, (x0 + x1) / 2 / size, (y0 + y1) / 2 / size, (x1 - x0) / size, (y1 - y0) / size]
    for k in KP_NAMES:
        x, y = s.keypoints[k]
        v = 2 if 0 <= x < size and 0 <= y < size else 0
        parts += [min(max(x / size, 0), 1), min(max(y / size, 0), 1), v]
    return " ".join(f"{p:.6f}" if isinstance(p, float) else str(p) for p in parts)


def _work(args):
    i, split, out = args
    s = sample_for(i, _CFG, _BGS)
    size = s.image.shape[0]
    cv2.imwrite(os.path.join(out, "images", split, f"{i:06d}.jpg"), s.image[..., ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(os.path.join(out, "labels", split, f"{i:06d}.txt"), "w") as f:
        f.write(yolo_line(s, size) + "\n")
    return label_record(i, split, s)


def draw_labels(img: np.ndarray, s) -> np.ndarray:
    out = img[..., ::-1].copy()
    x0, y0, x1, y1 = map(int, s.bbox)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    colors = {"center": (255, 255, 0), "tip": (0, 0, 255), "scale_min": (255, 0, 255),
              "scale_max": (0, 165, 255)}
    for k, c in colors.items():
        x, y = map(int, s.keypoints[k])
        cv2.circle(out, (x, y), 6, c, 2)
    cv2.putText(out, f"{s.value:.3g} {s.spec.unit} [{s.spec.min_value:g},{s.spec.max_value:g}]",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(out, ",".join(s.conditions) or "clean", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)
    return out


def preview(cfg, bgs, n: int, path: str, annotate: bool, start: int = 0, thumb: int = 320):
    tiles = []
    for i in range(start, start + n):
        s = sample_for(i, cfg, bgs)
        t = draw_labels(s.image, s) if annotate else s.image[..., ::-1]
        tiles.append(cv2.resize(t, (thumb, thumb), interpolation=cv2.INTER_AREA))
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * thumb, cols * thumb, 3), np.uint8)
    for j, t in enumerate(tiles):
        r, c = divmod(j, cols)
        sheet[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = t
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {path} ({n} images)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/synth_2d.yaml")
    ap.add_argument("--out", default="data/synthetic/synth_2d")
    ap.add_argument("--num", type=int, help="override num_images")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--preview", help="write a contact sheet to this path and exit")
    ap.add_argument("--no-annotate", action="store_true", help="preview without label overlay")
    ap.add_argument("--start", type=int, default=0, help="first image index (preview)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bgs = list_backgrounds(cfg.get("backgrounds_dir"))
    n = args.num or int(cfg.get("num_images", 1000))

    if args.preview:
        preview(cfg, bgs, n if args.num else 24, args.preview, not args.no_annotate, args.start)
        return

    n_val = round(n * args.val_frac)
    jobs = [(i, "val" if i >= n - n_val else "train", args.out) for i in range(n)]
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)
    print(f"generating {n} images ({n_val} val) with {args.workers} workers, "
          f"{len(bgs)} background photos")

    with open(os.path.join(args.out, "labels.jsonl"), "w", encoding="utf-8") as f:
        if args.workers > 1:
            with Pool(args.workers, initializer=_init, initargs=(cfg, bgs)) as pool:
                for k, rec in enumerate(pool.imap(_work, jobs, chunksize=16)):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if (k + 1) % 500 == 0:
                        print(f"  {k + 1}/{n}")
        else:
            _init(cfg, bgs)
            f.writelines(json.dumps(_work(job), ensure_ascii=False) + "\n" for job in jobs)

    data_yaml = {
        "path": os.path.abspath(args.out), "train": "images/train", "val": "images/val",
        "names": {0: "gauge"}, "kpt_shape": [4, 3],
        # A mirrored dial reads counter-clockwise, which real gauges never do:
        # train with fliplr=0. flip_idx is given only because Ultralytics requires it.
        "flip_idx": [0, 1, 3, 2],
    }
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# keypoints: 0 center, 1 needle tip, 2 scale min, 3 scale max\n")
        f.write("# train with fliplr=0 (mirrored dials read counter-clockwise)\n")
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
