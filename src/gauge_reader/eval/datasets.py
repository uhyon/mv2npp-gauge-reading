"""Loaders for the reading test sets (MeterFL, MeasureBench, synthetic val) as one sample type.

Both loaders return ``EvalSample`` lists, so the same scoring code
(``gauge_reader.eval.report``) works on either set:

- MeterFL: ``reading`` and the scale range are known, so % of full scale is the metric.
  Hand corrections from ``configs/meterfl_corrections.csv`` are applied on load.
- MeasureBench: ``alternatives`` holds the accepted interval per scale, so interval hit
  rate is the metric; % FS is added for the hand-labelled samples that have a range.
- Synthetic (``load_synth``): the held-out split of a generated set, exact labels. Its % FS
  next to MeterFL's shows the gap between synthetic and real images. Flags are the field
  conditions; extra groups are face tilt and gauge size.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from gauge_reader.units import norm_unit, unit_kind

ROOT = Path(__file__).resolve().parents[3]
METERFL_CORRECTIONS = ROOT / "configs" / "meterfl_corrections.csv"

# Short display names for range groups.
UNIT_SHORT = {
    "megapascal": "MPa", "mpa": "MPa", "kilopascal": "kPa", "kpa": "kPa", "bar": "bar",
    "mbar": "mbar", "poundpersquareinch": "psi", "poundspersquareinch": "psi", "psi": "psi",
    "kgf/cm2": "kgf/cm2", "kg/cm2": "kgf/cm2", "kilogram-forcepersquarecentimeter": "kgf/cm2",
    "kilogramforcepersquarecentimeter": "kgf/cm2", "degreecelsius": "°C", "°c": "°C", "celsius": "°C",
    "degreefahrenheit": "°F", "°f": "°F", "fahrenheit": "°F", "inh2o": "inH2O",
    "inchesofwater": "inH2O", "mmh2o": "mmH2O", "inhg": "inHg", "mmhg": "mmHg", "hpa": "hPa",
}


def short_unit(u: str) -> str:
    return UNIT_SHORT.get(norm_unit(u), u.strip())


def range_group(lo: float | None, hi: float | None, unit: str) -> str:
    """e.g. '0–0.1 MPa'. Samples without a range get 'no range'."""
    if lo is None or hi is None:
        return "no range"
    return f"{lo:g}–{hi:g} {short_unit(unit)}"


@dataclass
class EvalSample:
    id: str
    dataset: str                       # "meterfl" or "measurebench"
    image: Path
    unit: str                          # unit of the scale that reading/scale_* refer to
    scale_min: float | None = None
    scale_max: float | None = None
    reading: float | None = None       # ground-truth value (MeterFL; MeasureBench interval centre)
    alternatives: list[dict] = field(default_factory=list)  # MeasureBench accepted intervals
    subset: str = ""                   # MeterFL client folder, MeasureBench subtype
    tol_method: str = ""               # MeasureBench: interval / converted / minor_div / point_tol
    flags: list[str] = field(default_factory=list)
    corrected: list[str] = field(default_factory=list)  # fields changed by hand corrections
    groups: dict[str, str] = field(default_factory=dict)  # extra breakdowns, e.g. {"tilt": "20–40°"}

    @property
    def group(self) -> str:
        return range_group(self.scale_min, self.scale_max, self.unit)

    @property
    def has_range(self) -> bool:
        return self.scale_min is not None and self.scale_max is not None


# -------------------------------------------------------------------- MeterFL

def load_corrections(path: Path | str | None = METERFL_CORRECTIONS) -> dict[str, dict[str, str]]:
    """{id: {field: corrected_value}} from the hand-check CSV."""
    if not path or not Path(path).exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["id"], {})[row["field"]] = row["corrected_value"]
    return out


def load_meterfl(root: Path | str, corrections: Path | str | None = METERFL_CORRECTIONS,
                 pressure_only: bool = True) -> list[EvalSample]:
    """MeterFL reading test set (``data/readings/*.json``, 274 images; 262 pressure gauges).

    root: the cloned Meter-FL repository (``data/raw/Meter-FL``).
    """
    root = Path(root)
    files = sorted((root / "data" / "readings").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no reading labels under {root / 'data' / 'readings'}")
    fixes = load_corrections(corrections)
    out = []
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        if pressure_only and not r["client"].startswith("pressure_gauge"):
            continue
        sid = f.stem
        changed = []
        for key, val in fixes.get(sid, {}).items():
            r[key] = val if key == "unit" else float(val)
            changed.append(key)
        flags = [k for k in ("seg_anomaly", "annot_issue") if r.get(k)]
        out.append(EvalSample(
            id=sid, dataset="meterfl",
            image=root / "data" / r["client"] / r.get("side", "test") / r["file"],
            unit=r["unit"], scale_min=float(r["scale_min"]), scale_max=float(r["scale_max"]),
            reading=float(r["reading_gt"]), subset=r["client"], flags=flags, corrected=changed))
    unknown = {u for u in fixes if not (root / "data" / "readings" / f"{u}.json").exists()}
    if unknown:
        raise KeyError(f"corrections for ids not in MeterFL: {sorted(unknown)}")
    return out


# --------------------------------------------------------------- MeasureBench

def _float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_measurebench(labels: Path | str, exclude_barometers: bool = False,
                      exclude_flags: tuple[str, ...] = ()) -> list[EvalSample]:
    """The filtered MeasureBench dial subset (``labels.csv`` from scripts/filter_measurebench.py).

    Images are read from ``<labels dir>/images/<question_id>.png``. ``reading`` is the centre
    of the interval of the scale in ``unit``; it is used only for % FS on samples with a range.
    """
    labels = Path(labels)
    if labels.is_dir():
        labels = labels / "labels.csv"
    out = []
    with open(labels, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if exclude_barometers and row.get("subtype") == "barometer":
                continue
            flags = [x for x in (row.get("flags") or "").split(";") if x]
            if any(x in exclude_flags for x in flags):
                continue
            alts = json.loads(row["alternatives"])
            unit = row["unit"]
            reading = None
            for a in alts:
                if any(norm_unit(u) == norm_unit(unit) for u in a["units"]):
                    reading = (a["interval"][0] + a["interval"][1]) / 2
                    break
            if reading is None:
                reading = (float(row["interval_low"]) + float(row["interval_high"])) / 2
            out.append(EvalSample(
                id=row["question_id"], dataset="measurebench",
                image=labels.parent / "images" / f"{row['question_id']}.png",
                unit=unit, scale_min=_float(row.get("scale_min")),
                scale_max=_float(row.get("scale_max")), reading=reading, alternatives=alts,
                subset=row.get("subtype") or row.get("image_type", ""),
                tol_method=row.get("tol_method", ""), flags=flags))
    return out


# ------------------------------------------------------------------ synthetic

TILT_BINS = [(0, 20, "0–20°"), (20, 40, "20–40°"), (40, 91, "40°+")]
SIZE_BINS = [(0, 200, "0–200 px"), (200, 400, "200–400 px"), (400, 1e9, "400+ px")]


def _bin(x: float, bins) -> str:
    for lo, hi, name in bins:
        if lo <= x < hi:
            return name
    return "-"


def face_tilt(yaw_deg: float, pitch_deg: float) -> float:
    """Angle in degrees between the dial face normal and the camera axis.

    The generator rotates the gauge by Rz(roll) Ry(yaw) Rx(pitch); roll turns the face in
    its own plane, so the tilt is acos(cos yaw * cos pitch).
    """
    c = math.cos(math.radians(yaw_deg)) * math.cos(math.radians(pitch_deg))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def load_synth(root: Path | str, split: str = "val") -> list[EvalSample]:
    """One split of a generated set (``scripts/generate_synth.py``), from its labels.jsonl.

    reading = the true value, range = the dial range, flags = field conditions (empty for a
    clean image), subset = needle style. Extra groups: ``tilt`` (face tilt bins) and ``size``
    (longer side of the gauge box in px).
    """
    root = Path(root)
    path = root / "labels.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no labels.jsonl in {root}")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if split and r.get("split") != split:
                continue
            cam = r.get("camera", {})
            x0, y0, x1, y1 = r["bbox_xyxy"]
            out.append(EvalSample(
                id=f"synth_{r['id']}", dataset="synth", image=root / r["file"], unit=r["unit"],
                scale_min=float(r["min_value"]), scale_max=float(r["max_value"]),
                reading=float(r["value"]), subset=(r.get("style") or {}).get("needle") or "",
                flags=list(r.get("conditions") or []),
                groups={"tilt": _bin(face_tilt(cam.get("yaw", 0.0), cam.get("pitch", 0.0)), TILT_BINS),
                        "size": _bin(max(x1 - x0, y1 - y0), SIZE_BINS)}))
    if not out:
        raise ValueError(f"no '{split}' images in {path}")
    return out


def write_manifest(samples: list[EvalSample], path: Path | str) -> None:
    """Inference input: one row per image with the known range (blank when unknown).

    A model run reads this, fills value (and unit if it predicts one) and saves it as the
    predictions CSV for scripts/evaluate.py.
    """
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "scale_min", "scale_max", "unit", "kind", "value", "pred_unit"])
        for s in samples:
            w.writerow([s.id, s.image.as_posix(), "" if s.scale_min is None else s.scale_min,
                        "" if s.scale_max is None else s.scale_max, s.unit,
                        unit_kind(s.unit) or "", "", ""])
