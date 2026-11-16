"""Train the gauge keypoint model (Ultralytics YOLO pose) on the synthetic dataset.

    uv run python scripts/train_pose.py --data data/synthetic/synth_2d --epochs 50
    uv run python scripts/train_pose.py --data /kaggle/input/synth-2d/synth_2d \\
        --project /kaggle/working/runs --epochs 50 --batch 32
    uv run python scripts/train_pose.py --resume runs/pose/arc/weights/last.pt \\
        --data data/synthetic/synth_2d --name arc          # same --data/--project/--name
    uv run python scripts/train_pose.py --data data/synthetic/synth_2d --fraction 0.05 \\
        --epochs 1 --name smoke                              # quick check that it runs

--data is the dataset folder (images/, labels/, data.yaml) or a data.yaml. The folder's
data.yaml stores an absolute path from the machine that generated it, so a copy with the
right path is written to the run folder: the dataset can be moved (Kaggle, Colab) as is.

By default the model learns 7 keypoints: the dataset's 4 plus three on the scale arc (25, 50
and 75% of the scale), computed from labels.jsonl into a linked copy under --work
(scripts/arc_labels.py). With them a reading is taken on the dial face, which undoes the tilt
of the view (gauge_reader.reading.perspective). --no-arc trains on the dataset's own 4 points,
as Versions 1 and 2 did.

Horizontal flips are always off: a mirrored dial reads counter-clockwise, which real
gauges never do. Rotation is the opposite case: a turned dial still reads clockwise, and real
gauges are mounted and photographed at any angle, so images are rotated at random over the
full circle (--degrees 180). Version 1 (2026-11-13) was trained with --degrees 0 on renders
rolled by at most 15 degrees, and misread every MeterFL dial turned far from upright.
Needs the train extra: uv pip install -e ".[train]"
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

KPT_SHAPES = ([4, 3], [7, 3])


def dataset_yaml(data: Path, out_dir: Path) -> Path:
    """Copy of the dataset's data.yaml with ``path`` set to where the dataset is now."""
    src = data / "data.yaml" if data.is_dir() else data
    root = src.parent
    with open(src, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("kpt_shape") not in KPT_SHAPES:
        raise SystemExit(f"{src}: kpt_shape {cfg.get('kpt_shape')}, expected one of {KPT_SHAPES}")
    for split in ("train", "val"):
        if not (root / cfg[split]).is_dir():
            raise SystemExit(f"{src}: {split} folder {root / cfg[split]} not found")
    cfg["path"] = str(root.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "data.yaml"
    with open(dst, "w", encoding="utf-8") as f:
        f.write("# copy of the dataset's data.yaml with path set for this machine\n")
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return dst


def prepare_data(data: Path, *, arc: bool, work: Path, out_dir: Path) -> Path:
    """data.yaml to train on: the dataset's own (4 keypoints) or its 7-keypoint copy in ``work``."""
    if arc:
        from gauge_reader.synth.arc_labels import build

        counts = build(data if data.is_dir() else data.parent, work)
        print("arc labels: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f" images in {work}")
        data = work
    return dataset_yaml(data, out_dir)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/synthetic/synth_2d"))
    ap.add_argument("--model", default="yolo11s-pose.pt",
                    help="pretrained pose weights (yolo11n/s/m-pose.pt) or a .yaml to train from scratch")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=float, default=16,
                    help="batch size; -1 = auto (60%% of GPU memory), 0.7 = 70%% of GPU memory")
    ap.add_argument("--device", default=None, help="0, 0,1, cpu (default: Ultralytics picks)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="runs/pose")
    ap.add_argument("--name", default="base")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the train set to use")
    ap.add_argument("--degrees", type=float, default=180.0,
                    help="random rotation range, +/- degrees (0 = off, as in Version 1)")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-arc", dest="arc", action="store_false",
                    help="train on the dataset's 4 keypoints only (Versions 1 and 2)")
    ap.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "gauge_arc",
                    help="where the 7-keypoint copy of the dataset is built (links + labels)")
    ap.add_argument("--resume", type=Path,
                    help="continue an interrupted run from its last.pt (give the same --data, "
                         "--project and --name; the keypoint count comes from the checkpoint)")
    a = ap.parse_args(argv)

    from ultralytics import YOLO  # imported late so --help works without the train extra

    work, out_dir = a.work / a.name, Path(a.project) / f"{a.name}_data"
    if a.resume:
        # A new session (Kaggle) has lost the run's data.yaml copy and the arc copy: rebuild
        # them for the keypoints the checkpoint was trained on, then pass the data along.
        model = YOLO(str(a.resume))
        arc = list(model.model.kpt_shape) == KPT_SHAPES[1]
        data_yaml = prepare_data(a.data, arc=arc, work=work, out_dir=out_dir)
        model.train(resume=True, data=str(data_yaml))
        return

    data_yaml = prepare_data(a.data, arc=a.arc, work=work, out_dir=out_dir)
    batch = int(a.batch) if a.batch >= 1 or a.batch == -1 else a.batch
    model = YOLO(a.model)
    model.train(
        data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=batch, device=a.device,
        workers=a.workers, project=a.project, name=a.name, exist_ok=False, seed=a.seed,
        fraction=a.fraction, patience=a.patience, plots=True,
        fliplr=0.0,   # mirrored dials read counter-clockwise: never flip
        degrees=a.degrees,  # turned dials still read clockwise: rotate freely
        # The renderer already varies view angle, scale, light and blur; keep the remaining
        # Ultralytics augmentation at its defaults (mosaic, HSV, translate, scale).
    )
    print(f"best weights: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
