"""Train the gauge keypoint model (Ultralytics YOLO pose) on the synthetic dataset.

    uv run python scripts/train_pose.py --data data/synthetic/synth_2d --epochs 50
    uv run python scripts/train_pose.py --data /kaggle/input/synth-2d/synth_2d \\
        --project /kaggle/working/runs --epochs 50 --batch 32
    uv run python scripts/train_pose.py --resume runs/pose/base/weights/last.pt
    uv run python scripts/train_pose.py --data data/synthetic/synth_2d --fraction 0.05 \\
        --epochs 1 --name smoke                              # quick check that it runs

--data is the dataset folder (images/, labels/, data.yaml) or a data.yaml. The folder's
data.yaml stores an absolute path from the machine that generated it, so a copy with the
right path is written to the run folder: the dataset can be moved (Kaggle, Colab) as is.

Horizontal flips are always off: a mirrored dial reads counter-clockwise, which real
gauges never do. Needs the train extra: uv pip install -e ".[train]"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

KPT_SHAPE = [4, 3]


def dataset_yaml(data: Path, out_dir: Path) -> Path:
    """Copy of the dataset's data.yaml with ``path`` set to where the dataset is now."""
    src = data / "data.yaml" if data.is_dir() else data
    root = src.parent
    with open(src, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("kpt_shape") != KPT_SHAPE:
        raise SystemExit(f"{src}: kpt_shape {cfg.get('kpt_shape')}, expected {KPT_SHAPE}")
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
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, help="continue an interrupted run from its last.pt")
    a = ap.parse_args(argv)

    from ultralytics import YOLO  # imported late so --help works without the train extra

    if a.resume:
        YOLO(str(a.resume)).train(resume=True)
        return

    data_yaml = dataset_yaml(a.data, Path(a.project) / f"{a.name}_data")
    batch = int(a.batch) if a.batch >= 1 or a.batch == -1 else a.batch
    model = YOLO(a.model)
    model.train(
        data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=batch, device=a.device,
        workers=a.workers, project=a.project, name=a.name, exist_ok=False, seed=a.seed,
        fraction=a.fraction, patience=a.patience, plots=True,
        fliplr=0.0,   # mirrored dials read counter-clockwise: never flip
        # The renderer already varies view angle, roll, scale, light and blur; keep the
        # remaining Ultralytics augmentation at its defaults (mosaic, HSV, translate, scale).
    )
    print(f"best weights: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
