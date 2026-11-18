"""Read gauges end to end: photo -> detection -> keypoints -> RETAKE checks -> OCR -> value.

    # one or more photos: prints one JSON line each
    uv run python scripts/read_gauge.py --weights runs/pose/arc/weights/best.pt photo.jpg
    uv run python scripts/read_gauge.py --weights ... photo.jpg --band 0.2 0.8   # LOW/HIGH/NORMAL

    # a test set manifest: writes a predictions CSV that evaluate.py scores
    uv run python scripts/read_gauge.py --weights runs/pose/arc/weights/best.pt \\
        --manifest data/processed/meterfl_manifest.csv --out runs/pose/arc/meterfl_e2e.csv
    uv run python scripts/evaluate.py score meterfl --pred runs/pose/arc/meterfl_e2e.csv \\
        --out runs/pose/arc/meterfl_e2e

    # known instruments: range, unit and alarm band from a registry by tag; OCR only as backup
    uv run python scripts/read_gauge.py --weights ... --registry configs/gauges.csv --tag PI-101 photo.jpg
    uv run python scripts/read_gauge.py --weights ... --registry configs/gauges.csv --tag PI-101 \
        --check-scale photo.jpg     # also OCR the dial and report match / mismatch with PI-101

The range and unit come from the registry (--tag, or a ``tag`` column in the manifest), else
from OCR, unless --gt-range takes them from the manifest. --no-tta
reads upright only (no spread check). See gauge_reader.reader. The CSV keeps status, alarm,
the OCR range, the spread and the time per stage (ms_detect, ms_keypoints, ms_ocr, ms_total).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gauge_reader.reader import GaugeReader
from gauge_reader.reading.geometry import AlarmBand
from gauge_reader.registry import load_registry


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path, help="predictions CSV (with --manifest)")
    ap.add_argument("--gt-range", action="store_true", help="take the range and unit from the manifest")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--retake-spread", type=float, default=5.0)
    ap.add_argument("--band", type=float, nargs=2, metavar=("LOW", "HIGH"))
    ap.add_argument("--registry", type=Path, help="CSV of known gauges (gauge_reader.registry)")
    ap.add_argument("--tag", help="the photographed gauge's tag in --registry (single photos)")
    ap.add_argument("--check-scale", action="store_true",
                    help="with a known range, still OCR the dial and report whether it agrees")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--crop-margin", type=float, default=0.5,
                    help="real context around a small gauge's box, share of its size per side")
    ap.add_argument("--crop-pad", type=float, default=0.0, help="grey margin around that crop")
    ap.add_argument("--whole-share", type=float, default=0.4,
                    help="boxes at least this share of the photo are read whole (0: always whole, >1: always cropped)")
    ap.add_argument("--no-square", dest="square", action="store_false",
                    help="crop the box's own shape instead of a square window")
    a = ap.parse_args(argv)

    reader = GaugeReader(a.weights, ocr=not a.no_ocr and (a.check_scale or not a.gt_range), conf=a.conf, device=a.device,
                         tta_angle=None if a.no_tta else 25.0,
                         retake_spread=None if a.retake_spread < 0 else a.retake_spread,
                         crop_margin=a.crop_margin, crop_pad=a.crop_pad,
                         whole_share=a.whole_share, square=a.square, check_scale=a.check_scale)
    band = AlarmBand(*a.band) if a.band else None
    registry = load_registry(a.registry) if a.registry else {}
    if a.tag and a.tag not in registry:
        ap.error(f"tag {a.tag} is not in --registry")

    def known(tag):
        """(scale, band, tag) for a registered tag; band from --band before the registry's."""
        inst = registry.get((tag or "").strip())
        if inst is None:
            return None, band, ""
        return inst.scale, band or inst.band, inst.tag

    if not a.manifest:
        scale, bnd, tag = known(a.tag)
        for p in a.images:
            r = reader.read(p, scale=scale, band=bnd)
            print(json.dumps({"image": str(p), **({"tag": tag} if tag else {}), **r.as_dict()},
                             ensure_ascii=False))
        return

    with open(a.manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, row in enumerate(rows):
        scale, bnd, tag = known(row.get("tag") or a.tag)
        if a.gt_range:
            lo, hi = _f(row.get("scale_min")), _f(row.get("scale_max"))
            scale = (lo, hi, row.get("unit", "")) if lo is not None and hi is not None else None
        try:
            r = reader.read(row["image"], scale=scale, band=bnd)
        except ValueError:
            out.append({"id": row["id"], "value": "", "pred_unit": "", "status": "unreadable"})
            continue
        if a.gt_range and scale is None and r.status == "no_scale":
            r.status = "no_range"
        rec = {"id": row["id"], "value": "" if r.value is None else f"{r.value:.6g}",
               "pred_unit": r.unit if r.value is not None else "", "status": r.status,
               "alarm": r.alarm, "ocr_min": "" if r.scale_min is None else f"{r.scale_min:.6g}",
               "ocr_max": "" if r.scale_max is None else f"{r.scale_max:.6g}",
               "scale_source": r.scale_source, "scale_check": r.scale_check,
               "ocr_read_min": "" if r.ocr_scale is None else f"{r.ocr_scale[0]:.6g}",
               "ocr_read_max": "" if r.ocr_scale is None else f"{r.ocr_scale[1]:.6g}",
               "ocr_read_unit": "" if r.ocr_scale is None else r.ocr_scale[2],
               "fraction": "" if r.fraction is None else f"{r.fraction:.4f}",
               "spread": "" if r.spread is None else f"{r.spread:.2f}",
               "box_conf": "" if r.box_conf is None else f"{r.box_conf:.4f}",
               "box_share": "" if r.box_share is None else f"{r.box_share:.4f}"}
        for k, v in r.ms.items():
            rec[f"ms_{k}"] = f"{v:.1f}"
        rec["ms_total"] = f"{sum(r.ms.values()):.1f}"
        out.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}")
    keys = list(dict.fromkeys(k for r in out for k in r))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    counts: dict[str, int] = {}
    for r in out:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    tot = sorted(float(r["ms_total"]) for r in out if r.get("ms_total"))
    print(f"{len(out)} rows -> {a.out}; " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + (f"; median time {tot[len(tot) // 2]:.0f} ms/image" if tot else ""))


if __name__ == "__main__":
    main()
