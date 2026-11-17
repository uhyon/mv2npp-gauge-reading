"""How well one prediction column separates large errors (a RETAKE candidate signal).

    uv run python scripts/retake_table.py --pred runs/pose/arc/meterfl_tta.csv \\
        --rows runs/pose/arc/meterfl_tta/rows.csv --signal spread --thresholds 3 5 8 10

Joins the prediction CSV (signal column) with an evaluate.py rows.csv (fs_error) on id and,
for each threshold, prints how many rows are flagged (signal > t), how many large errors
(> --big % FS) are caught, how many flagged rows were within 2% FS anyway, and the error of
the rows kept. Rows without an fs_error (declined) are left out.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np


def table(signal: np.ndarray, err: np.ndarray, thresholds, big: float = 10.0) -> list[dict]:
    out = []
    for t in thresholds:
        m = signal > t
        k = ~m
        out.append({"t": t, "flagged": int(m.sum()), "caught": int((m & (err > big)).sum()),
                    "big": int((err > big).sum()), "flagged_ok": int((m & (err <= 2)).sum()),
                    "kept": int(k.sum()),
                    "kept_mean": float(err[k].mean()) if k.any() else float("nan"),
                    "kept_median": float(np.median(err[k])) if k.any() else float("nan"),
                    "kept_within2": float(np.mean(err[k] <= 2)) if k.any() else float("nan")})
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--signal", default="spread")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[3, 5, 8, 10])
    ap.add_argument("--big", type=float, default=10.0)
    a = ap.parse_args(argv)
    with open(a.pred, encoding="utf-8-sig") as f:
        P = {r["id"]: r for r in csv.DictReader(f)}
    with open(a.rows, encoding="utf-8-sig") as f:
        R = [r for r in csv.DictReader(f) if r.get("fs_error")]
    pairs = [(float(P[r["id"]][a.signal]), float(r["fs_error"])) for r in R
             if r["id"] in P and P[r["id"]].get(a.signal) not in (None, "")]
    s, e = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    print(f"{len(e)} rows; all: mean {e.mean():.2f}, median {np.median(e):.2f}, "
          f"<=2% {np.mean(e <= 2) * 100:.0f}%; large errors (>{a.big:g}) {int((e > a.big).sum())}; "
          f"Spearman({a.signal}, error) {spearman(s, e):+.2f}")
    print(f"| {a.signal} > t | flagged | large caught | flagged but <=2% | kept mean / median / <=2% |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for r in table(s, e, a.thresholds, a.big):
        print(f"| {r['t']:g} | {r['flagged']} ({r['flagged'] / len(e) * 100:.0f}%) | {r['caught']}/{r['big']} | "
              f"{r['flagged_ok']} | {r['kept_mean']:.2f} / {r['kept_median']:.2f} / "
              f"{r['kept_within2'] * 100:.0f}% |")


if __name__ == "__main__":
    main()
