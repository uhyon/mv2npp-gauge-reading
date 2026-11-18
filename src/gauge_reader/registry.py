"""Known instruments: the scale range, unit and alarm band of each gauge by its tag.

In a plant round the operator knows which gauge is in front of the camera (the tag on the
panel, a QR code, the round's checklist), so the range is looked up instead of read from the
dial: this avoids the OCR failures on blurred or dark photos (docs/RESULTS.md, synthetic val:
no OCR range on 90% of motion-blurred dials). OCR stays as the backup for an unknown gauge
and, with ``GaugeReader(check_scale=True)``, as a check that the photo shows the gauge the
tag says (``Reading.scale_check``).

Registry CSV columns: ``tag, scale_min, scale_max, unit`` and optional ``alarm_low,
alarm_high, description``. Lines starting with ``#`` are comments.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from gauge_reader.reading.geometry import AlarmBand
from gauge_reader.units import convert, norm_unit

SCALE_MATCH_TOL = 0.02   # OCR end values within 2% of the registered span count as the same


@dataclass(frozen=True)
class Instrument:
    tag: str
    scale_min: float
    scale_max: float
    unit: str
    band: AlarmBand | None = None
    description: str = ""

    @property
    def scale(self) -> tuple[float, float, str]:
        return self.scale_min, self.scale_max, self.unit


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_registry(path: str | Path) -> dict[str, Instrument]:
    """Tag -> Instrument. Raises ValueError on a row without a usable range or a repeated tag."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    out: dict[str, Instrument] = {}
    for row in csv.DictReader(lines):
        tag = (row.get("tag") or "").strip()
        lo, hi = _num(row.get("scale_min")), _num(row.get("scale_max"))
        if not tag or lo is None or hi is None or lo == hi:
            raise ValueError(f"{path}: bad registry row {row}")
        if tag in out:
            raise ValueError(f"{path}: tag {tag} listed twice")
        a_lo, a_hi = _num(row.get("alarm_low")), _num(row.get("alarm_high"))
        band = None
        if a_lo is not None or a_hi is not None:
            band = AlarmBand(a_lo if a_lo is not None else float("-inf"),
                             a_hi if a_hi is not None else float("inf"))
        out[tag] = Instrument(tag, lo, hi, (row.get("unit") or "").strip(), band,
                              (row.get("description") or "").strip())
    return out


def compare_scale(given: tuple[float, float, str], ocr_min: float | None, ocr_max: float | None,
                  ocr_unit: str = "", tol: float = SCALE_MATCH_TOL) -> str:
    """'match', 'mismatch' or 'no_ocr' (OCR found no range). A unit counts against the
    registry only when OCR read one; the end values must agree within ``tol`` of the span."""
    if ocr_min is None or ocr_max is None:
        return "no_ocr"
    lo, hi, unit = given
    span = abs(hi - lo)
    if abs(ocr_min - lo) > tol * span or abs(ocr_max - hi) > tol * span:
        return "mismatch"
    if ocr_unit and unit and not same_unit(ocr_unit, unit):
        return "mismatch"
    return "match"


def same_unit(a: str, b: str) -> bool:
    """One unit, however spelled: "MPa" = "megapascals", "°C" = "degC"."""
    if norm_unit(a) == norm_unit(b):
        return True
    lo, hi = convert(0.0, a, b), convert(100.0, a, b)
    return lo is not None and abs(lo) < 1e-9 and abs(hi - 100.0) < 1e-6
