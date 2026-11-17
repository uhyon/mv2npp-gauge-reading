import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_combine_median_and_spread():
    tta = _load("rotation_tta")
    rows = [{"id": "a", "scale_min": "0", "scale_max": "10", "unit": "bar"},
            {"id": "b", "scale_min": "0", "scale_max": "10", "unit": "bar"}]
    up = [{"status": "ok", "fraction": "0.50", "value": "5"}, {"status": "no_detection"}]
    plus = [{"status": "ok", "fraction": "0.52"}, {"status": "ok", "fraction": "0.3"}]
    minus = [{"status": "ok", "fraction": "1.40"}, {"status": "no_detection"}]
    a, b = tta.combine(rows, [up, plus, minus])
    assert a["frac_m"] == "1.1000"          # clipped
    assert float(a["value"]) == 5.2         # median of 0.50, 0.52, 1.10
    assert float(a["spread"]) == 60.0
    assert b["spread"] == "99" and float(b["value"]) == 3.0


def test_retake_table():
    rt = _load("retake_table")
    s = np.array([1, 2, 9, 12.0])
    e = np.array([0.5, 1.0, 30.0, 1.5])
    (r,) = rt.table(s, e, [5])
    assert r["flagged"] == 2 and r["caught"] == 1 and r["flagged_ok"] == 1
    assert r["kept_mean"] == 0.75 and r["kept_within2"] == 1
