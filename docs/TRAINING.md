# Training and first evaluation (phase 3)

One Ultralytics YOLO pose model, trained on the synthetic set, finds the gauge (box) and its
keypoints: center, needle tip, scale min and scale max, and since Version 3 three more on the
scale arc, at 25, 50 and 75% of the scale. With the arc points the needle is read on the dial
face: a homography from a round dial to the photo is fitted to the centre and the five scale
points, which undoes the tilt of the view (`gauge_reader.reading.perspective`). A 4-keypoint
model reads the needle angle in the image (`gauge_reader.reading.geometry`). In this first
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
- `fliplr=0` is forced: a mirrored dial reads counter-clockwise.
- Images are rotated at random over the full circle (`--degrees 180`, the default since Version 2):
  a turned dial still reads clockwise, and gauges are mounted and photographed at any angle.
  Version 1 used `--degrees 0` and misread every MeterFL dial turned by 90° or more
  ([RESULTS.md](RESULTS.md)). Other augmentation stays at the Ultralytics defaults; the renderer
  already varies view, scale, light and blur.
- Seven keypoints by default (since Version 3): the dataset's four plus three on the scale arc.
  They are computed from `labels.jsonl` (each image's dial-to-image homography and scale sweep)
  into a copy of the set under `--work` that links the images and holds the new labels, so
  nothing is rendered or uploaded again (`scripts/arc_labels.py` builds the same copy on its
  own). `--no-arc` trains on the four points, as Versions 1 and 2 did.
- `--resume runs/.../weights/last.pt` continues a stopped run. Give the same `--data`, `--project`
  and `--name`: a new Kaggle session has lost the run's copies of the data, so they are built
  again, with the keypoint count of the checkpoint.
- Default model `yolo11s-pose.pt` (COCO-pretrained), 640 px, early stop after 15 epochs without gain.
- The notebook names the run (`NAME`): Version 1 was `base`, Version 2 `rot`, Version 3 is
  `arc`. Below, replace `runs/pose/base` with the run being evaluated.

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
error analysis, and `--overlay` draws the predicted keypoints on every image. For a 7-keypoint
model `fraction` and `sweep` come from the dial-face fit, `fraction_img` keeps the image-plane
reading for comparison, and `fit_rms` is the fit residual as a share of the scale radius: how
well the points agree with a round dial, a candidate RETAKE signal.

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
images within 2% FS. Results so far: [RESULTS.md](RESULTS.md).
