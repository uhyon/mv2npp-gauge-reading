# Training and first evaluation (phase 3)

Baseline: one Ultralytics YOLO pose model finds the gauge (box) and its four keypoints
(center, needle tip, scale min, scale max) on the synthetic set. The reading is the needle
angle mapped onto the scale range (`gauge_reader.reading.geometry`). In this first
evaluation the **range comes from the ground truth**, so the error measured is the needle and
keypoint error only; scale OCR (range and unit) is added in phase 4 and scored separately.

## 1. Train

The PC has no CUDA GPU under WSL, so training runs on Kaggle (free T4 x2, 30 GPU hours a week).

Pack the dataset on the PC (2.0 GB, about 20k files):

```bash
cd ~/dev/mv2npp-gauge-reading
uv run python -c "import shutil; shutil.make_archive('synth_2d', 'zip', 'data/synthetic', 'synth_2d')"
explorer.exe .     # opens the folder in Windows to drag synth_2d.zip into the browser
```

On kaggle.com: **Datasets → New Dataset**, upload `synth_2d.zip`, name it `synth-2d`, keep it
private. Kaggle unpacks the zip. Then open `notebooks/train_kaggle.ipynb` there
(**Code → New Notebook → File → Import Notebook**) and follow its first cell.

`scripts/train_pose.py` is the same entry point everywhere (Kaggle, Colab, a local GPU):

```bash
uv run python scripts/train_pose.py --data data/synthetic/synth_2d --epochs 50 --device 0
uv run python scripts/train_pose.py --data data/synthetic/synth_2d --fraction 0.05 --epochs 1 --name smoke
```

- `--data` is the dataset folder; a copy of its `data.yaml` with the right absolute path is
  written next to the run, so the folder can be moved as is.
- `fliplr=0` is forced: a mirrored dial reads counter-clockwise. Other augmentation stays at the
  Ultralytics defaults; the renderer already varies view, roll, scale, light and blur.
- Default model `yolo11s-pose.pt` (COCO-pretrained), 640 px, early stop after 15 epochs without gain.

## 2. Read the test sets (on the PC, CPU is enough)

Copy `best.pt` from the Kaggle output to `runs/pose/base/weights/best.pt`, then:

```bash
uv run python scripts/evaluate.py manifest meterfl --out data/processed/meterfl_manifest.csv
uv run python scripts/predict_readings.py --weights runs/pose/base/weights/best.pt \
    --manifest data/processed/meterfl_manifest.csv --out runs/pose/base/meterfl_pred.csv \
    --overlay runs/pose/base/meterfl_overlay
uv run python scripts/evaluate.py score meterfl --pred runs/pose/base/meterfl_pred.csv --out runs/pose/base/meterfl

uv run python scripts/evaluate.py manifest measurebench --out data/processed/mb_manifest.csv
uv run python scripts/predict_readings.py --weights runs/pose/base/weights/best.pt \
    --manifest data/processed/mb_manifest.csv --out runs/pose/base/mb_pred.csv
uv run python scripts/evaluate.py score measurebench --pred runs/pose/base/mb_pred.csv --out runs/pose/base/mb
```

The held-out synthetic split (2,000 images) is scored the same way. Its error next to the
MeterFL error is the gap between synthetic and real photos; the report also breaks it down by
field condition (each one against the rest), face tilt (0–20°, 20–40°, 40°+) and gauge size.
It is an optimistic reference: the val images share the generator and background photos with
the training images, and Ultralytics picks `best.pt` on them. Synthetic images are whole
scenes, so no margin is added:

```bash
uv run python scripts/evaluate.py manifest synth --out data/processed/synth_val_manifest.csv
uv run python scripts/predict_readings.py --weights runs/pose/base/weights/best.pt --pad 0 \
    --manifest data/processed/synth_val_manifest.csv --out runs/pose/base/synth_val_pred.csv
uv run python scripts/evaluate.py score synth --pred runs/pose/base/synth_val_pred.csv --out runs/pose/base/synth_val
```

`predict_readings.py` uses the most confident detection per image. Two physical checks keep
a keypoint failure from turning into a 100x reading error: a scale arc (min to max keypoint)
under 90° is declined as `bad_geometry` (`--min-sweep`), and the needle fraction is clipped to
[-0.1, 1.1] (`--clip`, the raw value stays in the `fraction` column). Rows without a range in the
manifest (most MeasureBench images: only the hand-labelled ones have one) and images with no
detection are left blank, which `evaluate.py` counts as declined (lower coverage). MeterFL
images are tight crops, so a grey margin of 15% per side is added first (`--pad`, try 0 to
compare). Extra columns (keypoints, box and keypoint confidence, needle fraction) are kept for
error analysis, and `--overlay` draws the predicted keypoints on every image.

## 3. Robustness to field conditions (ER-05)

`scripts/robustness.py` reads the same test set clean and under each degradation of the
protocol in `docs/DATASETS.md` (low light, radiation speckle, motion blur, severity 1–3, one at
a time, fixed seed per image; `gauge_reader.eval.degrade`). Degraded images are made in
memory; the report has one row per setting with the change against clean, plus the model
time per image on this machine (ER-07):

```bash
uv run python scripts/robustness.py meterfl --weights runs/pose/base/weights/best.pt --out runs/pose/base/robust_meterfl
uv run python scripts/robustness.py synth --weights runs/pose/base/weights/best.pt --pad 0 --limit 500 \
    --out runs/pose/base/robust_synth
```

`examples.jpg` shows the first images under every setting, for checking only: MeterFL images
must not be used in demos.

Floor to beat on MeterFL: always reading mid-scale gives 20.7% FS mean error and 3% of
images within 2% FS.
