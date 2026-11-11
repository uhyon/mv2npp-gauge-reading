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
from gauge_reader.units import convert, flatten, norm_unit, unit_kind  # noqa: E402

# Dial counts in real_world (2026-11-10): pressure_gauge 118, thermometer 31, sphygmomanometer 21.
# sphygmomanometer (aneroid, usually a metal-capsule mechanism; only the dial looks like a
# pressure gauge) is optional: --keep-type sphygmomanometer
KEEP_TYPES = {"pressure_gauge", "thermometer"}

# Misclassified samples found by hand with --inspect (question_id -> reason). Always dropped.
EXCLUDE_IDS = {
    "pressure_gauge_6": "barometer; question reads the auxiliary 0-50 °C thermometer (checked 2026-11-10)",
    "pressure_gauge_39": "sphygmomanometer, needle-type 0-300 mmHg with bulb (checked 2026-11-10)",
}

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
    # A zero-width interval (lo == hi) is a single-value answer too.
    alts = [{"units": g, "interval": [lo, hi], "point": point or lo == hi, "tol": "interval"}
            for g, (lo, hi) in pairs]
    return widen_points_from_other_scale(alts)


def widen_points_from_other_scale(alts: list[dict]) -> list[dict]:
    """Give a point answer the tolerance the labeller used on another scale of the same dial.

    For each point alternative, the first non-point alternative whose unit converts to the
    point's unit is converted (e.g. 800-900 psi -> 5516-6205 kPa) and used as its interval,
    with tol = "converted:<source unit>". A point with no such partner keeps tol = "point_tol"
    and is widened by point_tol at evaluation time.
    """
    out = []
    for a in alts:
        if not a["point"]:
            out.append(a)
            continue
        new = dict(a, tol="point_tol")
        for b in alts:
            if b is a or b["point"]:
                continue
            pair = next(((su, du) for du in a["units"] for su in b["units"]
                         if isinstance(su, str) and isinstance(du, str)
                         and convert(1.0, su, du) is not None), None)
            if pair is None:
                continue
            su, du = pair
            lo, hi = sorted(convert(v, su, du) for v in b["interval"])
            new = dict(a, interval=[round(lo, 6), round(hi, 6)], point=False, tol=f"converted:{su}")
            break
        out.append(new)
    return out


def sanity_flags(image_type: str, alts: list[dict]) -> list[str]:
    """Cheap checks for likely mislabelled samples; reviewed by hand, never auto-dropped."""
    flags = []
    kinds = {unit_kind(u) for a in alts for u in a["units"]} - {None}
    if image_type == "pressure_gauge" and kinds == {"temperature"}:
        flags.append("temperature_unit_on_pressure_gauge")
    if image_type == "thermometer" and kinds == {"pressure"}:
        flags.append("pressure_unit_on_thermometer")
    if image_type == "pressure_gauge":
        for a in alts:
            if any(isinstance(u, str) and norm_unit(u) in ("mmhg", "millimeterofmercury") for u in a["units"]):
                mid = sum(a["interval"]) / 2
                if 0 <= mid < 320:  # negative mmHg = vacuum gauge, not a BP dial
                    flags.append("possible_sphygmomanometer")
                    break
    return flags


MANUAL_FIELDS = ["question_id", "image_path", "unit", "reading", "scale_min", "scale_max", "minor_div", "notes"]


def apply_manual(alts: list[dict], row: dict) -> tuple[list[dict], dict]:
    """Replace point_tol scales with reading +/- 0.5 * minor_div from a filled manual-label row.

    Returns the new alternatives and the full-scale range {scale_min, scale_max} for % FS.
    Rows with an empty minor_div are left unchanged.
    """
    try:
        d = float(row["minor_div"])
        lo_s, hi_s = float(row["scale_min"]), float(row["scale_max"])
    except (KeyError, TypeError, ValueError):
        return alts, {}
    out = []
    for a in alts:
        if a["tol"] == "point_tol":
            v = a["interval"][0]
            a = dict(a, interval=[v - d / 2, v + d / 2], point=False, tol="minor_div")
        out.append(a)
    return out, {"scale_min": lo_s, "scale_max": hi_s}


def tol_method(alts: list[dict]) -> str:
    """Per-sample summary for labels.csv: interval, converted, point_tol (or a + join)."""
    kinds = sorted({a["tol"].split(":")[0] for a in alts})
    return "+".join(kinds)


# Atmospheric-pressure band per barometer unit (normalized spelling -> (low, high)),
# about 870-1085 hPa, the recorded extremes of sea-level pressure.
BARO_BANDS = {
    "hpa": (870, 1085), "hectopascal": (870, 1085), "mbar": (870, 1085), "millibar": (870, 1085),
    "kpa": (87, 108.5), "kilopascal": (87, 108.5),
    "inhg": (25.7, 32.1), "inchofmercury": (25.7, 32.1), "inchesofmercury": (25.7, 32.1),
    "mmhg": (652, 814), "millimeterofmercury": (652, 814), "millimetersofmercury": (652, 814),
    "millimetreofmercury": (652, 814), "millimetresofmercury": (652, 814), "torr": (652, 814),
}


def is_barometer(alts: list[dict]) -> bool:
    """A pressure_gauge sample that reads atmospheric pressure (a barometer face).

    True when some scale has an hPa/mbar/kPa/inHg/mmHg unit and its interval midpoint lies
    in the atmospheric band. kPa alone is common on process gauges, so it needs the band too.
    """
    for a in alts:
        mid = sum(a["interval"]) / 2
        for u in a["units"]:
            band = BARO_BANDS.get(norm_unit(u)) if isinstance(u, str) else None
            if band and band[0] <= mid <= band[1]:
                return True
    return False


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


def main(argv: list[str] | None = None, rows=None) -> None:
    """Filter MeasureBench real_world dials. `rows` (list of dicts) replaces the HF download in tests."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/processed/measurebench_dial")
    p.add_argument("--keep-type", action="append", default=[], help="extra image_type to keep")
    p.add_argument("--exclude-type", action="append", default=[],
                   help="type to drop; 'barometer' drops pressure_gauge samples flagged as barometers")
    p.add_argument("--report", action="store_true", help="print drop reasons and dropped units")
    p.add_argument("--label-template", metavar="CSV",
                   help="write a manual-label template for point_tol samples (range, minor division)")
    p.add_argument("--manual-labels", metavar="CSV",
                   help="filled template: point_tol scales become reading +/- 0.5 minor_div")
    p.add_argument("--inspect", nargs="+", metavar="QID",
                   help="print the raw record and save the image of these question_ids, then exit")
    a = p.parse_args(argv)

    if rows is None:
        from datasets import load_dataset

        ds = load_dataset("FlagEval/MeasureBench", split="real_world")
    else:
        ds = rows
    dial = [r for r in ds if r["design"] == "dial"]
    print(f"real_world: {len(ds)}, dial: {len(dial)}")
    print("dial by image_type:", Counter(r["image_type"] for r in dial).most_common())

    if a.inspect:
        out = Path(a.out) / "inspect"
        out.mkdir(parents=True, exist_ok=True)
        for r in ds:
            if r["question_id"] in a.inspect:
                r["image"].save(out / f"{r['question_id']}.png")
                print(json.dumps({k: r[k] for k in ("question_id", "question", "image_type", "design",
                                                    "evaluator", "evaluator_kwargs", "meta_info")},
                                 ensure_ascii=False, indent=1, default=str))
        print(f"images -> {out.resolve()}")
        return

    manual = {}
    if a.manual_labels:
        with open(a.manual_labels, encoding="utf-8-sig") as f:
            manual = {row["question_id"]: row for row in csv.DictReader(f)}

    keep_types = (KEEP_TYPES | set(a.keep_type)) - set(a.exclude_type)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    kept, reasons, dropped_units, multi = [], Counter(), Counter(), Counter()
    no_iv_eval, no_iv_examples, paired_examples = Counter(), [], []
    zero_width, baro = [], []
    for r in dial:
        t = r["image_type"]
        if r["question_id"] in EXCLUDE_IDS:
            reasons["manual_exclude"] += 1
            continue
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
        scale = {}
        if r["question_id"] in manual:
            alts, scale = apply_manual(alts, manual[r["question_id"]])
        flags = sanity_flags(t, alts)
        subtype = "barometer" if t == "pressure_gauge" and is_barometer(alts) else t
        if subtype == "barometer":
            baro.append((r["question_id"], unit, iv))
            if "barometer" in a.exclude_type:
                reasons["excluded_barometer"] += 1
                continue
        if any(x["tol"] != "interval" for x in alts):
            zero_width.append((r["question_id"], unit, iv, tol_method(alts)))
        if why == "ok_paired" and len(paired_examples) < 10:
            paired_examples.append((r["question_id"], r["evaluator_kwargs"], unit, iv, alts))
        r["image"].save(out / "images" / f"{r['question_id']}.png")
        kept.append({"question_id": r["question_id"], "image_type": t, "subtype": subtype, "unit": unit,
                     "interval_low": iv[0], "interval_high": iv[1], "n_intervals": n,
                     "tol_method": tol_method(alts),
                     "flags": ";".join(flags),
                     "alternatives": json.dumps(alts, ensure_ascii=False),
                     "source": (r.get("meta_info") or {}).get("source", ""),
                     "scale_min": scale.get("scale_min", ""), "scale_max": scale.get("scale_max", "")})

    fields = ["question_id", "image_type", "subtype", "unit", "interval_low", "interval_high", "n_intervals",
              "tol_method", "flags", "alternatives", "source", "scale_min", "scale_max"]
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    # all files are written before any report output, so a report bug cannot lose them
    if a.label_template:
        todo = [k for k in kept if "point_tol" in k["tol_method"]]
        with open(a.label_template, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=MANUAL_FIELDS)
            w.writeheader()
            for k in todo:
                w.writerow({"question_id": k["question_id"],
                            "image_path": str(out / "images" / f"{k['question_id']}.png"),
                            "unit": k["unit"], "reading": k["interval_low"],
                            "scale_min": "", "scale_max": "", "minor_div": "", "notes": ""})
        print(f"label template for {len(todo)} point_tol samples -> {a.label_template}")
    flagged = [k for k in kept if k["flags"]]
    by_flag = Counter(f for k in flagged for f in k["flags"].split(";"))
    print(f"kept: {len(kept)}", Counter(k["image_type"] for k in kept).most_common(),
          "| by subtype:", Counter(k["subtype"] for k in kept).most_common())
    print(f"barometer-like pressure_gauge samples: {len(baro)}"
          + (" (excluded)" if "barometer" in a.exclude_type else " (kept, subtype=barometer)"))
    print(f"point answers (zero-width or single value): {len(zero_width)}",
          dict(Counter(z[3] for z in zero_width)))
    print("samples with several intervals (type, outcome):", dict(multi) or "none")
    if a.report:
        print("outcome counts:", dict(reasons))
        for flag in ("temperature_unit_on_pressure_gauge", "possible_sphygmomanometer",
                     "pressure_unit_on_thermometer"):
            ids = [k["question_id"] for k in flagged if flag in k["flags"].split(";")]
            print(f"  {flag}: {len(ids)} {ids}")
        print("dropped for unit (type, raw units -> count):")
        for (t, u), c in dropped_units.most_common():
            print(f"  {t:18s} {u}  x{c}")
        print("ok_paired examples (raw kwargs -> primary pair; all alternatives):")
        for qid, kws, u, iv, alts in paired_examples:
            print(f"  {qid:22s} {kws}\n      -> ({u}, {iv});  alternatives={json.dumps(alts, ensure_ascii=False)}")
        for qid, u, iv in baro:
            print(f"  barometer  {qid:22s} {u} {iv}")
        for qid, u, iv, how in zero_width:
            print(f"  point      {qid:22s} {u} {iv} -> {how}")
        print("no_interval by (type, evaluator):", dict(no_iv_eval) or "none")
        for qid, ev, kws in no_iv_examples:
            print(f"  {qid:22s} {ev:22s} {kws}")
    print(f"manually excluded: {len(EXCLUDE_IDS)} ({', '.join(EXCLUDE_IDS)})")
    print(f"sanity flags (check by hand with --inspect): {len(flagged)}", dict(by_flag) or "")
    for k in flagged:
        print(f"  {k['question_id']:22s} {k['unit']} [{k['interval_low']}, {k['interval_high']}] {k['flags']}")
    print(f"-> {out}/labels.csv (CC BY-SA 4.0: keep attribution, share derived sets alike)")


if __name__ == "__main__":
    main()
