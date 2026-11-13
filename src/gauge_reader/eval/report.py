"""Score reading predictions on ``EvalSample`` lists and summarize them by group.

A prediction is a value and, optionally, a unit. A missing value means the reader
declined (e.g. RETAKE); it lowers coverage and counts as a miss in the ``*_all`` metrics.

- % FS (``full_scale_error``) on samples that have a range and a ground-truth reading.
  A predicted unit different from the sample unit is converted first when possible.
- Interval hit (``interval_hit``) on samples with MeasureBench alternatives.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from gauge_reader.eval.datasets import EvalSample
from gauge_reader.eval.interval_hit import interval_hit
from gauge_reader.reading.geometry import full_scale_error
from gauge_reader.units import convert, same_unit


@dataclass
class Prediction:
    value: float | None
    unit: str | None = None


@dataclass
class Scored:
    sample: EvalSample
    pred: Prediction
    fs_error: float | None = None      # % of full scale, None if not computable
    hit: bool | None = None            # interval hit, None if no alternatives

    @property
    def answered(self) -> bool:
        return self.pred.value is not None and math.isfinite(self.pred.value)


def load_predictions(path: Path | str) -> dict[str, Prediction]:
    """CSV with columns id, value and optionally pred_unit (or unit). Blank value = declined.

    The manifest written by ``write_manifest`` has this shape once value is filled.
    """
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            v = (row.get("value") or "").strip()
            u = (row.get("pred_unit") or "").strip() or None
            out[row["id"]] = Prediction(float(v) if v else None, u)
    return out


def score_one(s: EvalSample, p: Prediction) -> Scored:
    r = Scored(s, p)
    if not r.answered:
        return r
    if s.has_range and s.reading is not None:
        v = p.value
        if p.unit and not same_unit(p.unit, s.unit):
            v = convert(p.value, p.unit, s.unit)
        if v is not None:
            r.fs_error = full_scale_error(v, s.reading, s.scale_min, s.scale_max)
    if s.alternatives:
        r.hit = interval_hit(p.value, p.unit, s.alternatives)
    return r


def score(samples: Iterable[EvalSample], preds: dict[str, Prediction]) -> list[Scored]:
    return [score_one(s, preds.get(s.id, Prediction(None))) for s in samples]


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(rows: list[Scored]) -> dict:
    """Metrics for one group of scored samples."""
    n = len(rows)
    ans = [r for r in rows if r.answered]
    out: dict = {"n": n, "coverage": len(ans) / n if n else float("nan")}
    fs_rows = [r for r in rows if r.sample.has_range and r.sample.reading is not None]
    fs = [r.fs_error for r in fs_rows if r.fs_error is not None]
    if fs_rows:
        out.update({
            "fs_n": len(fs),
            "fs_mean": sum(fs) / len(fs) if fs else float("nan"),
            "fs_median": _pct(fs, 0.5),
            "fs_p90": _pct(fs, 0.9),
            "within_1fs": sum(e <= 1 for e in fs) / len(fs) if fs else float("nan"),
            "within_2fs": sum(e <= 2 for e in fs) / len(fs) if fs else float("nan"),
            # declined or unconvertible samples count as failures
            "within_2fs_all": sum(e <= 2 for e in fs) / len(fs_rows),
        })
    hit_rows = [r for r in rows if r.sample.alternatives]
    if hit_rows:
        hits = [r.hit for r in hit_rows if r.hit is not None]
        out.update({
            "hit_n": len(hits),
            "hit_rate": sum(hits) / len(hits) if hits else float("nan"),
            "hit_rate_all": sum(hits) / len(hit_rows),
        })
    return out


def by_group(rows: list[Scored], key: Callable[[Scored], str]) -> dict[str, dict]:
    groups: dict[str, list[Scored]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    # largest groups first
    return {g: summarize(v) for g, v in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))}


def by_each_flag(rows: list[Scored]) -> dict[str, dict]:
    """One group per flag with every sample that has it (a sample can be in several),
    plus "(none)" for samples without flags. For sets where flags combine freely."""
    names = sorted({f for r in rows for f in r.sample.flags})
    groups = {f: [r for r in rows if f in r.sample.flags] for f in names}
    groups["(none)"] = [r for r in rows if not r.sample.flags]
    return {g: summarize(v) for g, v in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            if v}


def _bin_order(label: str) -> tuple:
    """Bins such as "0–20°", "20–40°", "40°+" sort by their leading number."""
    m = re.match(r"-?\d+(\.\d+)?", label)
    return (0, float(m.group()), label) if m else (1, 0.0, label)


def report(rows: list[Scored]) -> dict:
    """Overall metrics plus breakdowns by range group, subset, tolerance method, flag and
    any extra groups the samples carry (``EvalSample.groups``)."""
    flagged = [r for r in rows if r.sample.flags]
    out = {
        "overall": summarize(rows),
        "by_range": by_group(rows, lambda r: r.sample.group),
        "by_subset": by_group(rows, lambda r: r.sample.subset or "-"),
    }
    if any(r.sample.tol_method for r in rows):
        out["by_tol_method"] = by_group(rows, lambda r: r.sample.tol_method or "-")
    if flagged:
        out["without_flagged"] = summarize([r for r in rows if not r.sample.flags])
        if any(len(r.sample.flags) > 1 for r in rows):  # combinations: one row per flag instead
            out["by_each_flag"] = by_each_flag(rows)
        else:
            out["by_flag"] = by_group(flagged, lambda r: ";".join(r.sample.flags))
    for key in sorted({k for r in rows for k in r.sample.groups}):
        groups = by_group(rows, lambda r, k=key: r.sample.groups.get(k, "-"))
        out[f"by_{key}"] = dict(sorted(groups.items(), key=lambda kv: _bin_order(kv[0])))
    return out


def write_rows(rows: list[Scored], path: Path | str) -> None:
    """Per-sample results, for looking at the worst cases."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "subset", "unit", "scale_min", "scale_max", "reading",
                    "pred_value", "pred_unit", "fs_error", "hit", "flags", "image"])
        for r in sorted(rows, key=lambda r: -(r.fs_error if r.fs_error is not None else -1)):
            s = r.sample
            w.writerow([s.id, s.group, s.subset, s.unit, s.scale_min, s.scale_max, s.reading,
                        r.pred.value, r.pred.unit or "",
                        "" if r.fs_error is None else round(r.fs_error, 3),
                        "" if r.hit is None else int(r.hit), ";".join(s.flags), s.image.as_posix()])


METRIC_COLS = [("n", "n", "{:d}"), ("coverage", "cover", "{:.0%}"), ("fs_mean", "FS mean", "{:.2f}"),
               ("fs_median", "FS med", "{:.2f}"), ("within_2fs", "≤2%FS", "{:.0%}"),
               ("within_2fs_all", "≤2%FS all", "{:.0%}"), ("hit_rate", "hit", "{:.0%}"),
               ("hit_rate_all", "hit all", "{:.0%}")]


def format_table(title: str, groups: dict[str, dict]) -> str:
    """Markdown table; columns without data in any group are left out."""
    cols = [c for c in METRIC_COLS if any(c[0] in m for m in groups.values())]
    lines = [f"| {title} | " + " | ".join(c[1] for c in cols) + " |",
             "| --- |" + " ---: |" * len(cols)]
    for g, m in groups.items():
        cells = []
        for key, _, fmt in cols:
            v = m.get(key)
            cells.append("" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v))
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_report(rep: dict, max_groups: int = 12) -> str:
    parts = [format_table("overall", {"all": rep["overall"]})]
    if "without_flagged" in rep:
        parts.append(format_table("without flagged", {"all": rep["without_flagged"]}))
    keys = [("by_range", "range"), ("by_subset", "subset"), ("by_tol_method", "tolerance"),
            ("by_flag", "flag"), ("by_each_flag", "flag (each)")]
    keys += [(k, k[3:]) for k in rep if k.startswith("by_") and k not in dict(keys)]
    for key, title in keys:
        if key in rep:
            groups = dict(list(rep[key].items())[:max_groups])
            parts.append(format_table(title, groups))
    return "\n\n".join(parts)
