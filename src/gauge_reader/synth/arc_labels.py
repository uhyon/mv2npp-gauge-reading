"""Add the scale-arc keypoints to a generated dataset without rendering it again.

A generated set (``scripts/generate_synth.py``) stores, for every image, the homography from
the flat dial to the photo and the dial geometry (labels.jsonl). That is enough to place
keypoints anywhere on the scale arc, so the 7-keypoint labels (center, tip, scale min,
scale max, then the arc at 25, 50 and 75% of the scale) are computed from it.

The copy holds new labels and one symbolic link per image: Ultralytics resolves a linked
*folder* back to its source and would then read the source's 4-keypoint labels. An earlier copy
in the same folder is emptied first, so no stale image is trained on; a folder that holds real
images (the dataset itself, say) is refused.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from PIL import Image

from gauge_reader.reading.perspective import ARC_FRACTIONS, arc_points

KP_NAMES = ("center", "tip", "scale_min", "scale_max")
KPT_SHAPE = [len(KP_NAMES) + len(ARC_FRACTIONS), 3]
FLIP_IDX = [0, 1, 3, 2, 6, 5, 4]   # a mirror swaps min/max and 25%/75% (flips stay off anyway)


def record_keypoints(rec: dict) -> list[tuple[float, float]]:
    """The 7 keypoints of one labels.jsonl record, in image px."""
    dk = rec["dial_keypoints"]
    arc = arc_points(rec["homography"], dk["center"], dk["scale_min"], rec["sweep_deg"])
    return [tuple(rec["keypoints"][k]) for k in KP_NAMES] + arc


def label_line(rec: dict, width: int, height: int) -> str:
    """YOLO pose line: class, box, then x y visibility per keypoint, all normalised.

    Same rules as ``generate_synth.yolo_line``: points outside the image are clipped to it
    and marked invisible (0)."""
    x0, y0, x1, y1 = rec["bbox_xyxy"]
    parts: list = [0, (x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width,
                   (y1 - y0) / height]
    for x, y in record_keypoints(rec):
        v = 2 if 0 <= x < width and 0 <= y < height else 0
        parts += [min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0), v]
    return " ".join(f"{p:.6f}" if isinstance(p, float) else str(p) for p in parts)


def _clear(out: Path) -> None:
    """Remove an earlier copy's links and label files from ``out``; refuse real images."""
    imgs, lbls = (sorted(d.rglob("*")) if d.is_dir() else [] for d in (out / "images", out / "labels"))
    imgs = [p for p in imgs if p.is_symlink() or p.is_file()]
    real = [p for p in imgs if not p.is_symlink()]
    if real:
        raise SystemExit(f"{out} holds real images ({real[0]}), not a linked copy: "
                         "choose an empty or new folder")
    for p in imgs:
        p.unlink()
    for p in lbls:
        if p.is_file() and not p.is_symlink() and p.suffix in (".txt", ".cache"):
            p.unlink()


def build(data: Path, out: Path) -> dict[str, int]:
    """Write the 7-keypoint copy of the set in ``data`` to ``out``. Returns images per split."""
    data, out = Path(data).resolve(), Path(out)
    with open(data / "data.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _clear(out)
    counts: dict[str, int] = {}
    with open(data / "labels.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            src = data / rec["file"]
            split = rec["split"]
            img_dst = out / rec["file"]
            lbl_dst = out / "labels" / split / (Path(rec["file"]).stem + ".txt")
            if not counts.get(split):
                img_dst.parent.mkdir(parents=True, exist_ok=True)
                lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                w, h = im.size
            lbl_dst.write_text(label_line(rec, w, h) + "\n", encoding="utf-8")
            os.symlink(src, img_dst)
            counts[split] = counts.get(split, 0) + 1
    for split in ("train", "val"):
        n_img = sum(1 for _ in (data / cfg[split]).iterdir())
        if counts.get(split, 0) != n_img:
            raise SystemExit(f"{split}: {counts.get(split, 0)} records in labels.jsonl but "
                             f"{n_img} files in {data / cfg[split]}")
    cfg.update(path=str(out.resolve()), kpt_shape=KPT_SHAPE, flip_idx=FLIP_IDX)
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        f.write("# 7 keypoints: 0 center, 1 needle tip, 2 scale min, 3 scale max, "
                "4-6 scale arc at 25/50/75%\n")
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return counts
