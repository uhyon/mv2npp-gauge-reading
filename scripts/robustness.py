"""Robustness to NPP field conditions (ER-05): the same test set, clean and degraded.

    uv run python scripts/robustness.py meterfl --weights runs/pose/base/weights/best.pt \\
        --out runs/pose/base/robust_meterfl
    uv run python scripts/robustness.py synth --weights runs/pose/base/weights/best.pt --pad 0 \\
        --out runs/pose/base/robust_synth --limit 500

Reads the clean images, then each condition at severities 1–3 (low light, radiation
speckle, motion blur; gauge_reader.eval.degrade), one condition at a time, with a seed fixed
per image so every model sees the same degraded images. Degraded images are made in memory;
nothing is written except the results:

    <out>/report.md, report.json   one row per setting, with the change against clean
    <out>/preds/<setting>.csv      the predictions, for error and RETAKE analysis
    <out>/examples.jpg             the first images under every setting (--examples N; for
                                   checking only: MeterFL images must not go into demos)

The scale range comes from the ground truth, as in predict_readings.py. Each setting runs
the whole set once, so MeterFL (262 images) x 10 settings takes a few minutes on a CPU.

``--reader known`` reads with the integrated ``GaugeReader`` instead (small-gauge crop,
rotation median and the spread RETAKE), the range given as for a registered gauge;
``--reader ocr`` lets OCR read the range too (the full pipeline for an unknown gauge).
The report then also gives the RETAKE share (no value because the turned readings disagree).

    uv run python scripts/robustness.py meterfl --weights runs/pose/arc/weights/best.pt \
        --reader known --out runs/pose/arc/robust_meterfl_reader
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gauge_reader.eval.datasets import load_dataset, manifest_rows
from gauge_reader.eval.degrade import CONDITIONS, SEVERITIES, degrade, rng_for
from gauge_reader.eval.report import Prediction, score, summarize
from gauge_reader.predict import median_ms, read_rows, write_records


def read_with_reader(reader, rows, transform, given: bool) -> tuple[list[dict], dict[str, int]]:
    """GaugeReader on every row (degraded by ``transform``). Range from the row when ``given``."""
    import time

    records, counts = [], {}
    for row in rows:
        img = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
        rec = {"id": row["id"], "value": "", "pred_unit": "", "status": "unreadable"}
        if img is not None:
            if transform is not None:
                img = transform(img, row)
            scale = None
            if given:
                try:
                    scale = (float(row["scale_min"]), float(row["scale_max"]), row.get("unit", ""))
                except (TypeError, ValueError, KeyError):
                    scale = None
            t = time.perf_counter()
            r = reader.read(img, scale=scale)
            ms = (time.perf_counter() - t) * 1000
            status = "no_range" if given and scale is None and r.status == "no_scale" else r.status
            rec = {"id": row["id"], "value": "" if r.value is None else f"{r.value:.6g}",
                   "pred_unit": r.unit if r.value is not None else "", "status": status,
                   "fraction": "" if r.fraction is None else f"{r.fraction:.4f}",
                   "spread": "" if r.spread is None else f"{r.spread:.2f}",
                   "box_share": "" if r.box_share is None else f"{r.box_share:.4f}",
                   "ms": f"{ms:.1f}"}
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        records.append(rec)
    return records, counts


def write_csv(records: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in records for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)


def settings(conditions, severities) -> list[tuple[str, str | None, int]]:
    out = [("clean", None, 0)]
    out += [(f"{c}_s{s}", c, s) for c in conditions for s in severities]
    return out


def _fmt(v, fmt: str) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def table(results: dict[str, dict]) -> str:
    clean = results["clean"]
    has_fs = "fs_mean" in clean
    has_hit = "hit_rate_all" in clean
    head = ["setting", "n", "cover"]
    if has_fs:
        head += ["FS mean", "FS med", "≤2%FS all", "Δ FS mean", "Δ ≤2%FS all"]
    if has_hit:
        head += ["hit all", "Δ hit all"]
    has_retake = "retake_share" in clean and any(m.get("retake_share") for m in results.values())
    if has_retake:
        head += ["RETAKE"]
    head += ["ms/img"]
    lines = ["| " + " | ".join(head) + " |", "| --- |" + " ---: |" * (len(head) - 1)]
    for name, m in results.items():
        cells = [name, str(m["n"]), _fmt(m["coverage"], "{:.0%}")]
        if has_fs:
            cells += [_fmt(m.get("fs_mean"), "{:.2f}"), _fmt(m.get("fs_median"), "{:.2f}"),
                      _fmt(m.get("within_2fs_all"), "{:.0%}"),
                      "" if name == "clean" else _fmt(m["fs_mean"] - clean["fs_mean"], "{:+.2f}"),
                      "" if name == "clean" else
                      _fmt((m["within_2fs_all"] - clean["within_2fs_all"]) * 100, "{:+.0f} pp")]
        if has_hit:
            cells += [_fmt(m.get("hit_rate_all"), "{:.0%}"),
                      "" if name == "clean" else
                      _fmt((m["hit_rate_all"] - clean["hit_rate_all"]) * 100, "{:+.0f} pp")]
        if has_retake:
            cells += [_fmt(m.get("retake_share"), "{:.0%}")]
        cells += [_fmt(m.get("median_ms"), "{:.0f}")]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def examples_sheet(rows: list[dict], sets, seed: int, n: int, path: Path, thumb: int = 160) -> None:
    picked = []
    for row in rows:
        img = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
        if img is not None:
            picked.append((row, img))
        if len(picked) == n:
            break
    if not picked:
        return
    sheet = np.full((len(picked) * thumb + 20, len(sets) * thumb, 3), 255, np.uint8)
    for c, (name, cond, sev) in enumerate(sets):
        cv2.putText(sheet, name.replace("radiation_", ""), (c * thumb + 3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        for r, (row, img) in enumerate(picked):
            out = img if cond is None else degrade(img, cond, sev, rng_for(row["id"], cond, sev, seed))
            s = thumb / max(out.shape[:2])
            t = cv2.resize(out, (max(1, round(out.shape[1] * s)), max(1, round(out.shape[0] * s))),
                           interpolation=cv2.INTER_AREA)
            y, x = 20 + r * thumb, c * thumb
            sheet[y:y + t.shape[0], x:x + t.shape[1]] = t
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["meterfl", "measurebench", "synth"])
    ap.add_argument("--data", type=Path, help="dataset path (default: the usual location)")
    ap.add_argument("--split", default="val", help="synth: split of labels.jsonl")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--severities", default=",".join(map(str, SEVERITIES)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="use only the first N images (quick runs)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--pad", type=float, default=0.15, help="0 for synthetic (whole-scene) images")
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-sweep", type=float, default=90.0,
                    help="decline when the min-to-max scale arc is smaller (degrees)")
    ap.add_argument("--clip", type=float, default=0.1,
                    help="clip the needle fraction to [-clip, 1+clip]; negative = no clipping")
    ap.add_argument("--examples", type=int, default=4, help="images in examples.jpg (0 = none)")
    ap.add_argument("--reader", choices=["off", "known", "ocr"], default="off",
                    help="read with GaugeReader: range given (known) or read by OCR (ocr)")
    ap.add_argument("--retake-spread", type=float, default=5.0,
                    help="--reader: RETAKE above this spread (%% of scale); negative = off")
    a = ap.parse_args(argv)

    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    sevs = [int(s) for s in a.severities.split(",") if s.strip()]
    bad = [c for c in conds if c not in CONDITIONS] + [s for s in sevs if s not in SEVERITIES]
    if bad:
        ap.error(f"unknown condition or severity: {bad}")

    from ultralytics import YOLO

    samples = load_dataset(a.dataset, a.data, split=a.split)
    if a.limit:
        samples = samples[:a.limit]
    rows = manifest_rows(samples)
    model = YOLO(str(a.weights))
    reader = None
    if a.reader != "off":
        from gauge_reader.reader import GaugeReader
        reader = GaugeReader(model=model, ocr=a.reader == "ocr", conf=a.conf, imgsz=a.imgsz,
                             device=a.device, min_sweep=a.min_sweep,
                             clip=0.0 if a.clip < 0 else a.clip,
                             retake_spread=None if a.retake_spread < 0 else a.retake_spread)
    sets = settings(conds, sevs)
    a.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name, cond, sev in sets:
        transform = None
        if cond is not None:
            def transform(img, row, c=cond, s=sev):
                return degrade(img, c, s, rng_for(row["id"], c, s, a.seed))
        if reader is None:
            records, counts = read_rows(model, rows, imgsz=a.imgsz, conf=a.conf, pad=a.pad,
                                        device=a.device, transform=transform,
                                        min_sweep=a.min_sweep, clip=None if a.clip < 0 else a.clip)
            write_records(records, a.out / "preds" / f"{name}.csv")
        else:
            records, counts = read_with_reader(reader, rows, transform, given=a.reader == "known")
            write_csv(records, a.out / "preds" / f"{name}.csv")
        preds = {r["id"]: Prediction(float(r["value"]) if r["value"] else None, r["pred_unit"] or None)
                 for r in records}
        m = summarize(score(samples, preds))
        m["median_ms"] = median_ms(records)
        m["status"] = {k: v for k, v in counts.items() if v}
        m["retake_share"] = counts.get("retake", 0) / max(len(rows), 1)
        results[name] = m
        fs = m.get("fs_mean", float("nan"))
        print(f"{name:22s} cover {m['coverage']:.0%}  FS mean {fs:.2f}  "
              + ", ".join(f"{k} {v}" for k, v in m["status"].items()))

    head = (f"# Robustness: {a.dataset} ({len(samples)} images), {a.weights}\n\n"
            f"Conditions applied one at a time at severities {sevs}, seed {a.seed}, pad {a.pad}, "
            f"min sweep {a.min_sweep}°, clip {a.clip}. "
            + ("" if reader is None else
               f"GaugeReader, range {'given' if a.reader == 'known' else 'by OCR'}, rotation "
               f"median, RETAKE spread > {a.retake_spread}% (pad does not apply). ")
            + "Δ = change against clean (FS in % of full scale, shares in percentage points). "
            + ("Scale range from ground truth.\n" if a.reader != "ocr" else "\n"))
    text = head + "\n" + table(results) + "\n"
    (a.out / "report.md").write_text(text, encoding="utf-8")
    (a.out / "report.json").write_text(json.dumps(results, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    if a.examples:
        examples_sheet(rows, sets, a.seed, a.examples, a.out / "examples.jpg")
    print("\n" + text + f"\n-> {a.out}")


if __name__ == "__main__":
    main()
