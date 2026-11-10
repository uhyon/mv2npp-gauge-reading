"""Interval hit rate for MeasureBench-style labels.

A label is a list of alternatives, each {"units": [...synonyms], "interval": [lo, hi], "point": bool}.
A dual-scale dial (e.g. bar and psi) has one alternative per scale, so a reading on either scale
counts as a hit when it lands in that scale's interval.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from gauge_reader.units import same_unit


def _widen(alt: dict, point_tol: float) -> tuple[float, float]:
    lo, hi = alt["interval"]
    if alt.get("point") and point_tol > 0:
        pad = abs(lo) * point_tol
        return lo - pad, hi + pad
    return lo, hi


def interval_hit(pred_value: float, pred_unit: str | None, alternatives: list[dict] | str,
                 point_tol: float = 0.0) -> bool:
    """True if the prediction falls inside the interval of the matching scale.

    pred_unit selects the scale(s) whose units contain it (normalized). If pred_unit is None,
    or matches no scale, every alternative is tried, which is the lenient setting for models
    that do not output a unit; report which setting was used.
    point_tol: relative tolerance added around single-value (point) answers, e.g. 0.02.
    """
    alts = json.loads(alternatives) if isinstance(alternatives, str) else alternatives
    if pred_unit is not None:
        matching = [a for a in alts if any(same_unit(pred_unit, u) for u in a["units"])]
        if matching:
            alts = matching
    for a in alts:
        lo, hi = _widen(a, point_tol)
        if lo <= pred_value <= hi:
            return True
    return False


def hit_rate(rows: Iterable[tuple[float, str | None, list[dict] | str]], point_tol: float = 0.0) -> float:
    rows = list(rows)
    if not rows:
        return float("nan")
    return sum(interval_hit(v, u, alts, point_tol) for v, u, alts in rows) / len(rows)


def hit_rate_by_group(rows: Iterable[tuple[str, float, str | None, list[dict] | str]],
                      point_tol: float = 0.0) -> dict[str, tuple[float, int]]:
    """Hit rate per group, e.g. per labels.csv tol_method, so point_tol samples are reported apart.

    rows: (group, pred_value, pred_unit, alternatives). Returns {group: (rate, n)}.
    """
    groups: dict[str, list] = {}
    for g, v, u, alts in rows:
        groups.setdefault(g, []).append((v, u, alts))
    return {g: (hit_rate(r, point_tol), len(r)) for g, r in sorted(groups.items())}
