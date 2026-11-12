"""Evaluate gauge readings on MeterFL (main, % FS) or MeasureBench (interval hit rate).

1. Write the inference manifest (one row per test image with its known range):

    uv run python scripts/evaluate.py manifest meterfl --out data/processed/meterfl_manifest.csv
    uv run python scripts/evaluate.py manifest measurebench --out data/processed/mb_manifest.csv

2. Run the reader on the manifest and fill ``value`` (and ``pred_unit`` if it predicts one;
   leave value blank when the reader declines). Then score:

    uv run python scripts/evaluate.py score meterfl --pred runs/x/meterfl_pred.csv --out runs/x/meterfl
    uv run python scripts/evaluate.py score measurebench --pred runs/x/mb_pred.csv --out runs/x/mb

   Writes report.md, report.json and rows.csv (per image, worst first) to --out.

Reference baselines, no model needed:

    uv run python scripts/evaluate.py score meterfl --baseline midscale   # always reads mid-scale
    uv run python scripts/evaluate.py score meterfl --baseline oracle     # ground truth (sanity check)

Default paths: MeterFL in data/raw/Meter-FL (corrections from configs/meterfl_corrections.csv),
MeasureBench subset in data/processed/measurebench_dial/labels.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gauge_reader.eval.datasets import (
    METERFL_CORRECTIONS,
    load_measurebench,
    load_meterfl,
    write_manifest,
)
from gauge_reader.eval.report import (
    Prediction,
    format_report,
    load_predictions,
    report,
    score,
    write_rows,
)


def load(a) -> list:
    if a.dataset == "meterfl":
        corr = None if a.no_corrections else METERFL_CORRECTIONS
        return load_meterfl(a.meterfl, corrections=corr, pressure_only=not a.all_types)
    flags = tuple(x for x in (a.exclude_flag or []))
    return load_measurebench(a.measurebench, exclude_barometers=a.no_barometers, exclude_flags=flags)


def baseline(samples, kind: str) -> dict[str, Prediction]:
    out = {}
    for s in samples:
        if kind == "oracle":
            out[s.id] = Prediction(s.reading, s.unit)
        elif s.has_range:  # midscale
            out[s.id] = Prediction((s.scale_min + s.scale_max) / 2, s.unit)
        else:
            out[s.id] = Prediction(None)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["manifest", "score"])
    ap.add_argument("dataset", choices=["meterfl", "measurebench"])
    ap.add_argument("--meterfl", type=Path, default=Path("data/raw/Meter-FL"))
    ap.add_argument("--measurebench", type=Path,
                    default=Path("data/processed/measurebench_dial/labels.csv"))
    ap.add_argument("--all-types", action="store_true", help="MeterFL: keep ammeters, voltmeters etc.")
    ap.add_argument("--no-corrections", action="store_true", help="MeterFL: raw labels")
    ap.add_argument("--no-barometers", action="store_true", help="MeasureBench: drop barometers")
    ap.add_argument("--exclude-flag", action="append",
                    help="MeasureBench: drop samples with this flag (repeatable)")
    ap.add_argument("--pred", type=Path, help="predictions CSV (id, value, pred_unit)")
    ap.add_argument("--baseline", choices=["midscale", "oracle"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    samples = load(a)
    if a.command == "manifest":
        out = a.out or Path(f"data/processed/{a.dataset}_manifest.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(samples, out)
        print(f"{len(samples)} images -> {out}")
        return

    if bool(a.pred) == bool(a.baseline):
        ap.error("score needs exactly one of --pred or --baseline")
    preds = load_predictions(a.pred) if a.pred else baseline(samples, a.baseline)
    unknown = set(preds) - {s.id for s in samples}
    if unknown:
        print(f"warning: {len(unknown)} predictions for ids not in the test set, ignored")
    rows = score(samples, preds)
    rep = report(rows)
    text = format_report(rep)
    corrected = sum(bool(s.corrected) for s in samples)
    head = f"# {a.dataset}: {a.pred or 'baseline ' + a.baseline}\n\n{len(samples)} images"
    if a.dataset == "meterfl":
        head += f", {corrected} with hand corrections" if corrected else ", raw labels"
    if a.baseline == "oracle":  # the oracle must score perfectly; list anything that does not
        bad = [r for r in rows if r.hit is False or (r.fs_error or 0) > 1e-9]
        text += f"\n\noracle misses: {len(bad)}"
        for r in bad:
            text += (f"\n- {r.sample.id} ({r.sample.subset}): pred {r.pred.value:g} {r.pred.unit}, "
                     f"fs {r.fs_error}, alternatives {json.dumps(r.sample.alternatives, ensure_ascii=False)}")
    print(head + "\n\n" + text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "report.md").write_text(head + "\n\n" + text + "\n", encoding="utf-8")
        (a.out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
        write_rows(rows, a.out / "rows.csv")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
