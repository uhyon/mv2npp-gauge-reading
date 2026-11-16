"""Copy a generated dataset with three more keypoints on the scale arc (7 in all).

    uv run python scripts/arc_labels.py --data data/synthetic/synth_2d --out /tmp/synth_2d_arc

The new points sit at 25, 50 and 75% of the scale and are computed from labels.jsonl (each
image's homography and dial geometry), so nothing is rendered again. Images are linked, not
copied. scripts/train_pose.py does this by itself unless --no-arc is given; models trained on
these labels are read on the dial face (gauge_reader.reading.perspective), which undoes the
tilt of the view.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gauge_reader.synth.arc_labels import build


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/synthetic/synth_2d"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    counts = build(a.data, a.out)
    print(", ".join(f"{k} {v}" for k, v in counts.items()) + f" images -> {a.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
