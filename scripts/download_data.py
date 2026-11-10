"""Download the public evaluation datasets into data/raw/.

Run this on your own machine (not committed: data/ is git-ignored).

    pip install -e ".[data]"
    python scripts/download_data.py --all
    python scripts/download_data.py --synanthropic
    ROBOFLOW_API_KEY=... python scripts/download_data.py --roboflow

Licenses differ per dataset. Read docs/DATASETS.md before using or redistributing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

ROBOFLOW_PROJECTS = [
    # (workspace, project, version, export format)
    ("hustlegrad", "analog-gauge-reading-v4tii", 7, "yolov8"),
    ("analog-6kh04", "analog-gauge-obaw9", 1, "yolov8"),
]


def download_synanthropic() -> None:
    from huggingface_hub import snapshot_download

    out = RAW / "synanthropic"
    snapshot_download(
        repo_id="Synanthropic/reading-analog-gauge",
        repo_type="dataset",
        local_dir=out,
    )
    print(f"[ok] Synanthropic (MIT) -> {out}. Unzip corners.zip / keypoint.zip inside it.")


def download_roboflow() -> None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY (free account at roboflow.com) first.")
    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    for ws, proj, ver, fmt in ROBOFLOW_PROJECTS:
        out = RAW / "roboflow" / proj
        rf.workspace(ws).project(proj).version(ver).download(fmt, location=str(out))
        print(f"[ok] Roboflow {ws}/{proj} v{ver} -> {out}. Record its license in docs/DATASETS.md.")


def cambridge_instructions() -> None:
    out = RAW / "cambridge"
    out.mkdir(parents=True, exist_ok=True)
    print(
        "[manual] Cambridge SyntheticGauges / RealGauges (CC BY-NC 4.0):\n"
        "  1. Open http://www.jjcvision.com/projects/gauge_reading.html\n"
        "  2. Click 'Download' (accepting the license) and extract into\n"
        f"     {out}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true")
    p.add_argument("--synanthropic", action="store_true")
    p.add_argument("--roboflow", action="store_true")
    p.add_argument("--cambridge", action="store_true")
    a = p.parse_args()
    if not any(vars(a).values()):
        p.print_help()
        return
    RAW.mkdir(parents=True, exist_ok=True)
    if a.all or a.synanthropic:
        download_synanthropic()
    if a.all or a.roboflow:
        download_roboflow()
    if a.all or a.cambridge:
        cambridge_instructions()


if __name__ == "__main__":
    main()
