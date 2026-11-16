import json
import os

import cv2
import numpy as np
import pytest
import yaml

from gauge_reader.reading.geometry import GaugeKeypoints, needle_fraction
from gauge_reader.reading.perspective import ARC_FRACTIONS, arc_points, fit_dial
from gauge_reader.synth.arc_labels import build, record_keypoints
from gauge_reader.synth.scene import make_sample

CFG = {"image_size": 320, "seed": 0,
       "dial": {"ranges": [[0, 10], [0, 16], [-1, 1], [0, 150]], "units": ["bar", "psi"]},
       "camera": {"yaw_deg": [-50, 50], "pitch_deg": [-40, 40], "roll_deg": [-180, 180]}}


def _record(i, split, s):
    return {"id": f"{i:06d}", "split": split, "file": f"images/{split}/{i:06d}.jpg",
            "value": s.value, "min_value": s.spec.min_value, "max_value": s.spec.max_value,
            "sweep_deg": s.spec.sweep_deg,
            "keypoints": {k: list(v) for k, v in s.keypoints.items()},
            "bbox_xyxy": list(s.bbox), "homography": np.asarray(s.homography).tolist(),
            "dial_keypoints": {k: list(v) for k, v in s.dial_keypoints.items()}}


@pytest.mark.parametrize("seed", range(6))
def test_arc_points_follow_the_generator(seed):
    s = make_sample(np.random.default_rng(seed), CFG)
    dk, sp = s.dial_keypoints, s.spec
    ends = arc_points(s.homography, dk["center"], dk["scale_min"], sp.sweep_deg, fractions=(0.0, 1.0))
    np.testing.assert_allclose(ends, [s.keypoints["scale_min"], s.keypoints["scale_max"]], atol=1e-6)
    # with the true 7 keypoints the dial-face reading is exact, whatever the view
    pts = record_keypoints(_record(seed, "train", s))
    fit = fit_dial(GaugeKeypoints(*pts[:4]), pts[4:])
    frac = (s.value - sp.min_value) / (sp.max_value - sp.min_value)
    assert fit.fraction == pytest.approx(frac, abs=2e-4)
    assert fit.sweep == pytest.approx(sp.sweep_deg, abs=0.05)


def test_image_plane_reading_is_off_on_tilted_renders():
    errs = []
    for seed in range(12):
        s = make_sample(np.random.default_rng(seed), CFG)
        sp = s.spec
        frac = (s.value - sp.min_value) / (sp.max_value - sp.min_value)
        errs.append(abs(needle_fraction(GaugeKeypoints(**s.keypoints)) - frac))
    assert max(errs) > 0.01


def _dataset(root, n_train=2, n_val=1):
    recs = []
    for i in range(n_train + n_val):
        split = "train" if i < n_train else "val"
        s = make_sample(np.random.default_rng(i), CFG)
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / "images" / split / f"{i:06d}.jpg"), s.image[..., ::-1])
        recs.append(_record(i, split, s))
    with open(root / "labels.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"path": "/somewhere/else", "train": "images/train", "val": "images/val",
         "names": {0: "gauge"}, "kpt_shape": [4, 3], "flip_idx": [0, 1, 3, 2]}))
    return recs


def test_build_writes_seven_keypoint_labels(tmp_path):
    recs = _dataset(tmp_path / "src")
    out = tmp_path / "arc"
    assert build(tmp_path / "src", out) == {"train": 2, "val": 1}
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["kpt_shape"] == [7, 3] and cfg["path"] == str(out.resolve())
    for r in recs:
        img = out / r["file"]
        assert img.is_symlink() and os.path.samefile(img, tmp_path / "src" / r["file"])
        vals = (out / "labels" / r["split"] / f"{r['id']}.txt").read_text().split()
        assert len(vals) == 5 + 7 * 3
        xy = np.array([float(v) for v in vals[5:]]).reshape(7, 3)[:, :2] * 320
        want = np.array(record_keypoints(r))
        inside = (want >= 0).all(axis=1) & (want < 320).all(axis=1)
        np.testing.assert_allclose(xy[inside], want[inside], atol=1e-3)
        assert len(ARC_FRACTIONS) == 3


def test_build_stops_on_images_without_a_record(tmp_path):
    _dataset(tmp_path / "src")
    cv2.imwrite(str(tmp_path / "src" / "images" / "val" / "extra.jpg"), np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(SystemExit):
        build(tmp_path / "src", tmp_path / "arc")


def test_rebuild_leaves_no_stale_images(tmp_path):
    _dataset(tmp_path / "big", n_train=3, n_val=1)
    _dataset(tmp_path / "small", n_train=1, n_val=1)
    out = tmp_path / "arc"
    build(tmp_path / "big", out)
    (out / "labels" / "train.cache").write_text("stale")
    assert build(tmp_path / "small", out) == {"train": 1, "val": 1}
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == ["000000.jpg"]
    assert sorted(p.name for p in (out / "labels" / "train").iterdir()) == ["000000.txt"]
    assert not (out / "labels" / "train.cache").exists()
    assert all(os.path.samefile(p, tmp_path / "small" / "images" / "train" / p.name)
               for p in (out / "images" / "train").iterdir())


def test_build_refuses_a_folder_with_real_images(tmp_path):
    recs = _dataset(tmp_path / "src")
    with pytest.raises(SystemExit):
        build(tmp_path / "src", tmp_path / "src")   # the dataset itself
    for r in recs:   # nothing was touched
        assert (tmp_path / "src" / r["file"]).is_file()
        assert not (tmp_path / "src" / r["file"]).is_symlink()
    assert yaml.safe_load((tmp_path / "src" / "data.yaml").read_text())["kpt_shape"] == [4, 3]


def _train_pose():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "train_pose.py"
    spec = importlib.util.spec_from_file_location("train_pose", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_pose_prepares_the_arc_copy_or_the_set_itself(tmp_path):
    tp = _train_pose()
    _dataset(tmp_path / "src")
    arc = yaml.safe_load(tp.prepare_data(tmp_path / "src", arc=True, work=tmp_path / "work",
                                         out_dir=tmp_path / "run_a").read_text())
    assert arc["kpt_shape"] == [7, 3] and arc["path"] == str((tmp_path / "work").resolve())
    plain = yaml.safe_load(tp.prepare_data(tmp_path / "src", arc=False, work=tmp_path / "unused",
                                           out_dir=tmp_path / "run_b").read_text())
    assert plain["kpt_shape"] == [4, 3] and plain["path"] == str((tmp_path / "src").resolve())
    assert not (tmp_path / "unused").exists()
