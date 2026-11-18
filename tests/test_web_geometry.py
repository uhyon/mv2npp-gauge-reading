"""The browser demo's geometry (demo/web/geometry.js) gives the same dial fit as the Python
reader. Needs Node; skipped without it."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gauge_reader.reading.perspective import fit_dial, frontal_matrix
from tests.test_perspective import dial

NODE = shutil.which("node")
GEOMETRY = Path(__file__).resolve().parents[1] / "demo" / "web" / "geometry.js"
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

CASES = [dict(frac=0.31), dict(frac=0.0, yaw=45, pitch=30, roll=10),
         dict(frac=0.87, sweep=180, a0=180, yaw=-30, pitch=20, noise=1.5, seed=1),
         dict(frac=1.04, sweep=300, a0=120, pitch=50, noise=3.0, seed=2),
         dict(frac=0.55, sweep=90, a0=225, yaw=20, noise=0.5, seed=3)]


def _js(cases):
    script = f"""
import {{ fitDial, frontalMatrix }} from {json.dumps(GEOMETRY.as_uri())};
const cases = {json.dumps(cases)};
console.log(JSON.stringify(cases.map(([kp, arc]) => {{
  const f = fitDial(kp, arc);
  return {{ fraction: f.fraction, sweep: f.sweep, rms: f.rms, frontal: frontalMatrix(f) }};
}})));
"""
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True,
                         text=True, check=True)
    return json.loads(out.stdout)


def test_fit_matches_python():
    dials = [dial(**c) for c in CASES]
    js = _js([[{"center": k.center, "tip": k.tip, "scale_min": k.scale_min,
                "scale_max": k.scale_max}, arc] for k, arc in dials])
    for (kp, arc), got in zip(dials, js):
        fit = fit_dial(kp, arc)
        assert got["fraction"] == pytest.approx(fit.fraction, abs=1e-6)
        assert got["sweep"] == pytest.approx(fit.sweep, abs=1e-6)
        assert got["rms"] == pytest.approx(fit.rms, rel=1e-4, abs=1e-8)
        m = np.array(got["frontal"]).reshape(3, 3)
        ref = frontal_matrix(fit)
        assert m / m[2, 2] == pytest.approx(ref / ref[2, 2], rel=1e-6, abs=1e-6)
