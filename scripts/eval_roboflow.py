"""Evaluation 3 (keypoints, Roboflow v4tii) and 4 (detection, Roboflow obaw9).

    uv run python scripts/eval_roboflow.py keypoints --weights runs/pose/arc/weights/best.pt \\
        --out runs/pose/arc/v4tii
    uv run python scripts/eval_roboflow.py detection --weights runs/pose/arc/weights/best.pt \\
        --out runs/pose/arc/obaw9

keypoints: the 326 originals of hustlegrad v4tii (train deduped to its 245 originals, valid 49,
test 32; ``data/raw/roboflow/gauge-detection-2-2ynzm``). Each labelled gauge is matched to a
detection by box IoU (>= 0.3, detections at conf >= --conf). Point errors are distances in % of
the gauge size (sqrt of the box area) after undoing the export's stretch resize (the Gauge box
is made square); PCK@5% / @10% count points within 5% / 10% of the gauge size. The needle
fraction (0 at the min tick, 1 at the max tick, x100 = % of the scale) of the labelled points
is compared with the model's image-plane fraction (``frac_err_img``, also unstretched) and, for
arc models, its dial-face fraction (``frac_err_face``). No reading GT exists in this set.

detection: box AP of the gauge class. On obaw9 (1,471 images, ``data/raw/roboflow/
analog-gauge-obaw9``, never used for training) the gauge box is the box of the ``meter``
polygon; v4tii Gauge boxes are scored too. AP50 and AP50-95 use all detections down to
conf 0.001; precision and recall are also given at conf 0.25.

Writes <out>/rows.csv (keypoints) and <out>/report.md + report.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gauge_reader.eval.roboflow import (
    DetRecord,
    ap_range,
    average_precision,
    label_path,
    list_images,
    match_gauges,
    parse_obaw9_boxes,
    parse_v4tii,
    score_gauge,
    summarize_keypoints,
    v4tii_images,
)
from gauge_reader.keypoints.yolo import detections_from_result, pad_image, shift

V4TII = ROOT / "data/raw/roboflow/gauge-detection-2-2ynzm"
OBAW9 = ROOT / "data/raw/roboflow/analog-gauge-obaw9"


def predict(model, img, *, pad: float, conf: float, imgsz: int, device):
    padded, dx, dy = pad_image(img, pad)
    t0 = time.perf_counter()
    res = model.predict(padded, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    return [shift(d, dx, dy) for d in detections_from_result(res)], ms


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _pct(x) -> str:
    return "" if x is None else f"{x * 100:.0f}%"


def keypoints(a) -> None:
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    items = v4tii_images(a.root)
    if a.limit:
        items = items[:a.limit]
    rows, ms = [], []
    for i, (split, img_path) in enumerate(items):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = parse_v4tii(label_path(img_path).read_text(), W, H)
        dets, t = predict(model, img, pad=a.pad, conf=a.conf, imgsz=a.imgsz, device=a.device)
        ms.append(t)
        for k, (g, d) in enumerate(match_gauges(gts, dets)):
            r = {"image": img_path.name, "split": split, "gauge": k, "n_det": len(dets)}
            r.update(score_gauge(g, d))
            rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(items)}")
    a.out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("image", "split", "gauge"), k))
    with open(a.out / "rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})
    summary = {"all": summarize_keypoints(rows)}
    for split in ("train", "valid", "test"):
        sub = [r for r in rows if r["split"] == split]
        if sub:
            summary[split] = summarize_keypoints(sub)
    ms.sort()
    summary["median_ms"] = ms[len(ms) // 2] if ms else None
    summary["images"] = len(items)
    summary["args"] = {"weights": str(a.weights), "pad": a.pad, "conf": a.conf}
    (a.out / "report.json").write_text(json.dumps(summary, indent=2))

    L = [f"# v4tii keypoints: {a.weights}", "",
         (f"{len(items)} images, pad {a.pad}, conf {a.conf}, median model time "
          f"{summary['median_ms']:.0f} ms"), "",
         ("| split | gauges | detected | scored | center med / PCK5 | tip med / PCK5 | min med / PCK5 "
          "| max med / PCK5 | fraction img: mean / med / ≤2% / ≤5% | fraction face: mean / med / ≤2% / ≤5% |"),
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in summary.items():
        if not isinstance(s, dict) or "gauges" not in s:
            continue
        cells = [name, str(s["gauges"]), f"{s['detected']} ({_pct(s['detection_rate'])})", str(s["scored"])]
        for f in ("center", "tip", "scale_min", "scale_max"):
            cells.append(f"{s[f + '_err_median'] * 100:.1f}% / {_pct(s[f + '_pck05'])}"
                         if f + "_err_median" in s else "")
        for key in ("frac_err_img", "frac_err_face"):
            cells.append(f"{s[key + '_mean']:.2f} / {s[key + '_median']:.2f} / {_pct(s[key + '_within2'])}"
                         f" / {_pct(s[key + '_within5'])}" if key + "_mean" in s else "")
        L.append("| " + " | ".join(cells) + " |")
    s = summary["all"]
    L += ["", (f"Labels: {s['complete_labels']} of {s['gauges']} gauges have all four points; "
               f"needle drawn tip-first in {s['needle_swapped']}. Point errors in % of the gauge size, "
               "after undoing the stretch.")]
    (a.out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"-> {a.out}")


def detection(a) -> None:
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    sets = []
    if not a.skip_obaw9:
        imgs = list_images(a.obaw9 / "train")
        sets.append(("obaw9 (meter box)", [(p, "obaw9") for p in imgs]))
    if not a.skip_v4tii:
        sets.append(("v4tii (Gauge box)", [(p, "v4tii") for _, p in v4tii_images(a.root)]))
    summary, L = {}, [f"# Gauge detection: {a.weights}", "", f"pad {a.pad}, imgsz {a.imgsz}", "",
                      "| set | images | gauges | AP50 | AP50-95 | precision@0.25 | recall@0.25 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, items in sets:
        if a.limit:
            items = items[:a.limit]
        recs = []
        for i, (p, kind) in enumerate(items):
            img = cv2.imread(str(p))
            if img is None:
                continue
            H, W = img.shape[:2]
            text = label_path(p).read_text() if label_path(p).exists() else ""
            gt = parse_obaw9_boxes(text, W, H) if kind == "obaw9" else [g.box for g in parse_v4tii(text, W, H)]
            dets, _ = predict(model, img, pad=a.pad, conf=0.001, imgsz=a.imgsz, device=a.device)
            recs.append(DetRecord(gt, [(d.box_conf, d.box) for d in dets]))
            if (i + 1) % 200 == 0:
                print(f"  {name}: {i + 1}/{len(items)}")
        r50 = average_precision(recs, 0.5)
        r = {"images": len(recs), **r50, "ap50_95": ap_range(recs)}
        summary[name] = r
        L.append(f"| {name} | {r['images']} | {r['n_gt']} | {r['ap']:.3f} | {r['ap50_95']:.3f} | "
                 f"{r['precision@0.25']:.3f} | {r['recall@0.25']:.3f} |")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(summary, indent=2))
    (a.out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"-> {a.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("keypoints", "detection"):
        p = sub.add_parser(name)
        p.add_argument("--weights", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--root", type=Path, default=V4TII, help="v4tii folder")
        p.add_argument("--pad", type=float, default=0.0, help="grey margin (fraction of the image)")
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--device", default=None)
        p.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    sub.choices["keypoints"].add_argument("--conf", type=float, default=0.1)
    d = sub.choices["detection"]
    d.add_argument("--obaw9", type=Path, default=OBAW9)
    d.add_argument("--skip-obaw9", action="store_true")
    d.add_argument("--skip-v4tii", action="store_true")
    a = ap.parse_args(argv)
    keypoints(a) if a.cmd == "keypoints" else detection(a)


if __name__ == "__main__":
    main()
