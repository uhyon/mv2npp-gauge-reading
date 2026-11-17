import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_retake_table():
    rt = _load("retake_table")
    s = np.array([1, 2, 9, 12.0])
    e = np.array([0.5, 1.0, 30.0, 1.5])
    (r,) = rt.table(s, e, [5])
    assert r["flagged"] == 2 and r["caught"] == 1 and r["flagged_ok"] == 1
    assert r["kept_mean"] == 0.75 and r["kept_within2"] == 1
