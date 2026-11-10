"""Build the MeasureBench subset used as a secondary test set.

    pip install datasets
    python scripts/filter_measurebench.py --out data/processed/measurebench_dial

Prints counts per image_type before and after filtering, and writes the kept
images plus a labels.csv (question_id, image_type, unit, interval_low,
interval_high, source). Scale ranges are not in MeasureBench: add
scale_min/scale_max columns by hand for the % FS subset.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

KEEP_TYPES = {"pressure_gauge", "thermometer", "barometer", "hygrometer", "manometer"}
DROP_HINTS = ("clock", "watch", "ammeter", "voltmeter", "multimeter", "meter_", "energy", "utility",
              "speedometer", "tachometer", "compass", "odometer")
UNITS = {"bar", "kpa", "mpa", "psi", "kgf/cm2", "kgf/cm²", "mmh2o", "inhg", "mbar", "hpa",
         "°c", "°f", "c", "f", "celsius", "fahrenheit", "degree celsius", "degree fahrenheit"}


def unit_ok(units: list[str]) -> bool:
    return any(u.strip().lower() in UNITS for u in units)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/processed/measurebench_dial")
    p.add_argument("--keep-type", action="append", default=[], help="extra image_type to keep")
    a = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset("FlagEval/MeasureBench", split="real_world")
    dial = [r for r in ds if r["design"] == "dial"]
    print(f"real_world: {len(ds)}, dial: {len(dial)}")
    print("dial by image_type:", Counter(r["image_type"] for r in dial).most_common())

    keep_types = KEEP_TYPES | set(a.keep_type)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    kept = []
    for r in dial:
        t = r["image_type"]
        kw = json.loads(r["evaluator_kwargs"])
        if t not in keep_types or any(h in t for h in DROP_HINTS) or not unit_ok(kw.get("units", [])):
            continue
        intervals = kw.get("interval")
        lo, hi = (intervals if intervals and not isinstance(intervals[0], list) else intervals[0])
        path = out / "images" / f"{r['question_id']}.png"
        r["image"].save(path)
        kept.append({"question_id": r["question_id"], "image_type": t, "unit": kw["units"][0],
                     "interval_low": lo, "interval_high": hi,
                     "source": (r.get("meta_info") or {}).get("source", ""),
                     "scale_min": "", "scale_max": ""})
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0].keys()) if kept else ["question_id"])
        w.writeheader()
        w.writerows(kept)
    print(f"kept: {len(kept)}", Counter(k["image_type"] for k in kept).most_common())
    print(f"-> {out}/labels.csv (CC BY-SA 4.0: keep attribution, share derived sets alike)")


if __name__ == "__main__":
    main()
