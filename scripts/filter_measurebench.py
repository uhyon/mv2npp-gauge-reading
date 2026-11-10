"""Build the MeasureBench subset used as a secondary test set.

    pip install -e ".[data]"   (or: uv run --with datasets ...)
    python scripts/filter_measurebench.py --out data/processed/measurebench_dial --report
    python scripts/filter_measurebench.py --keep-type sphygmomanometer   (adds 21 aneroid BP dials)

Writes the kept images plus labels.csv (question_id, image_type, unit,
interval_low, interval_high, n_intervals, source, scale_min, scale_max).
`alternatives` holds every acceptable (units, interval) pair, e.g. both scales of a
bar/psi dial; evaluate with gauge_reader.eval.interval_hit. Scale ranges are not in MeasureBench: fill scale_min/scale_max by hand for
the % FS subset. --report prints drop reasons and the raw units of every
sample dropped for its unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gauge_reader.units import flatten, unit_kind  # noqa: E402

# Dial counts in real_world (2026-11-10): pressure_gauge 118, thermometer 31, sphygmomanometer 21.
# sphygmomanometer (aneroid, usually a metal-capsule mechanism; only the dial looks like a
# pressure gauge) is optional: --keep-type sphygmomanometer
KEEP_TYPES = {"pressure_gauge", "thermometer"}

def unit_ok(units) -> bool:
    return any(unit_kind(u) for u in flatten(units))


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def intervals_of(interval) -> list[tuple[float, float]]:
    """[lo, hi] -> [(lo, hi)];  [[lo, hi], [lo2, hi2]] -> both;  a single number v -> [(v, v)]."""
    if interval is None or interval == [] or interval == "":
        return []
    if not isinstance(interval, (list, tuple)):
        v = _num(interval)
        return [] if v is None else [(v, v)]
    if isinstance(interval[0], (list, tuple)):
        return [iv for item in interval for iv in intervals_of(item)]
    return [(float(interval[0]), float(interval[-1]))]


def first_interval(interval) -> tuple[float, float] | None:
    ivs = intervals_of(interval)
    return ivs[0] if ivs else None


ANSWER_KEYS = ("interval", "intervals", "value", "answer", "target", "reading", "range")


def answer_field(kw: dict):
    """The first answer-like field present: an interval, a list of intervals, or a single value."""
    for k in ANSWER_KEYS:
        if k in kw and kw[k] not in (None, "", []):
            return kw[k]
    return None


def unit_groups(units) -> list[list[str]]:
    """["bar", "Bar"] -> one group of synonyms; [["bar"], ["psi"]] -> one group per scale."""
    if isinstance(units, (list, tuple)) and units and all(isinstance(u, (list, tuple)) for u in units):
        return [[v for v in flatten(g) if isinstance(v, str)] for g in units]
    return [[v for v in flatten(units) if isinstance(v, str)]]


def alternatives(kw: dict) -> list[dict]:
    """Every acceptable (units, interval) answer of a sample.

    Dual-scale dial (n intervals, n unit groups): one entry per scale, paired by position.
    Otherwise every interval shares the single unit group.
    A single-value answer is stored with "point": true so evaluation can add a tolerance.
    """
    raw = answer_field(kw)
    point = raw is not None and not isinstance(raw, (list, tuple))
    ivs = intervals_of(raw)
    groups = unit_groups(kw.get("units", []))
    if len(ivs) > 1 and len(groups) == len(ivs):
        pairs = zip(groups, ivs)
    else:
        pairs = ((groups[0] if groups else [], iv) for iv in ivs)
    return [{"units": g, "interval": [lo, hi], "point": point} for g, (lo, hi) in pairs]


def pick(kw: dict) -> tuple[str | None, tuple[float, float] | None, int, str]:
    """Choose (unit, interval) for a sample. Returns unit, interval, n_intervals, reason.

    When there are several intervals and as many unit groups (a dual-scale dial),
    intervals and unit groups are paired by position and the first pair whose unit
    is a pressure or temperature unit is used. Several intervals with one unit group
    are alternatives on the same scale (tolerance bands), and the first is used.
    """
    ivs = intervals_of(answer_field(kw))
    groups = unit_groups(kw.get("units", []))
    if not ivs:
        return None, None, 0, "no_interval"
    if len(ivs) > 1 and len(groups) == len(ivs):
        for iv, g in zip(ivs, groups):
            u = next((x for x in g if unit_kind(x)), None)
            if u:
                return u, iv, len(ivs), "ok_paired"
        return None, None, len(ivs), "unit"
    if len(ivs) > 1 and len(groups) > 1:
        return None, None, len(ivs), "interval_unit_count_mismatch"
    u = next((x for x in groups[0] if unit_kind(x)), None) if groups else None
    if u is None:
        return None, None, len(ivs), "unit"
    return u, ivs[0], len(ivs), "ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/processed/measurebench_dial")
    p.add_argument("--keep-type", action="append", default=[], help="extra image_type to keep")
    p.add_argument("--exclude-type", action="append", default=[], help="image_type to drop")
    p.add_argument("--report", action="store_true", help="print drop reasons and dropped units")
    a = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset("FlagEval/MeasureBench", split="real_world")
    dial = [r for r in ds if r["design"] == "dial"]
    print(f"real_world: {len(ds)}, dial: {len(dial)}")
    print("dial by image_type:", Counter(r["image_type"] for r in dial).most_common())

    keep_types = (KEEP_TYPES | set(a.keep_type)) - set(a.exclude_type)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    kept, reasons, dropped_units, multi = [], Counter(), Counter(), Counter()
    no_iv_eval, no_iv_examples, paired_examples = Counter(), [], []
    for r in dial:
        t = r["image_type"]
        if t not in keep_types:
            reasons["type"] += 1
            continue
        kw = json.loads(r["evaluator_kwargs"])
        unit, iv, n, why = pick(kw)
        if n > 1:
            multi[(t, why)] += 1
        if unit is None:
            reasons[why] += 1
            if why == "unit":
                dropped_units[(t, json.dumps(kw.get("units"), ensure_ascii=False))] += 1
            if why == "no_interval":
                no_iv_eval[(t, r["evaluator"])] += 1
                if len(no_iv_examples) < 10:
                    no_iv_examples.append((r["question_id"], r["evaluator"], r["evaluator_kwargs"]))
            continue
        reasons[why] += 1
        alts = alternatives(kw)
        if why == "ok_paired" and len(paired_examples) < 10:
            paired_examples.append((r["question_id"], r["evaluator_kwargs"], unit, iv, alts))
        r["image"].save(out / "images" / f"{r['question_id']}.png")
        kept.append({"question_id": r["question_id"], "image_type": t, "unit": unit,
                     "interval_low": iv[0], "interval_high": iv[1], "n_intervals": n,
                     "alternatives": json.dumps(alts, ensure_ascii=False),
                     "source": (r.get("meta_info") or {}).get("source", ""),
                     "scale_min": "", "scale_max": ""})

    fields = ["question_id", "image_type", "unit", "interval_low", "interval_high", "n_intervals",
              "alternatives", "source", "scale_min", "scale_max"]
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    print(f"kept: {len(kept)}", Counter(k["image_type"] for k in kept).most_common())
    print("samples with several intervals (type, outcome):", dict(multi) or "none")
    if a.report:
        print("outcome counts:", dict(reasons))
        print("dropped for unit (type, raw units -> count):")
        for (t, u), c in dropped_units.most_common():
            print(f"  {t:18s} {u}  x{c}")
        print("ok_paired examples (raw kwargs -> primary pair; all alternatives):")
        for qid, kws, u, iv, alts in paired_examples:
            print(f"  {qid:22s} {kws}\n      -> ({u}, {iv});  alternatives={json.dumps(alts, ensure_ascii=False)}")
        print("no_interval by (type, evaluator):", dict(no_iv_eval) or "none")
        for qid, ev, kws in no_iv_examples:
            print(f"  {qid:22s} {ev:22s} {kws}")
    print(f"-> {out}/labels.csv (CC BY-SA 4.0: keep attribution, share derived sets alike)")


if __name__ == "__main__":
    main()
