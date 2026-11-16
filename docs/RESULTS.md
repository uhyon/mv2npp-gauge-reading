# Results

All readings below take the scale range from the ground truth, so they measure the needle and
keypoint stage only (see [TRAINING.md](TRAINING.md)); scale OCR is scored separately in phase 4.
Errors are in % of full scale (FS). "≤2% FS" counts declined images as misses. Targets are in
[REQUIREMENTS.md](REQUIREMENTS.md) (ER-01: mean ≤ 2% FS and ≥ 85% of images within 2% FS).

## Version 1 (2026-11-13): baseline

| | |
| --- | --- |
| Model | yolo11s-pose, COCO-pretrained, 640 px |
| Data | synthetic train set, 18,000 images (renders rolled at most ±15°), no rotation augmentation |
| Training | 50 epochs, batch 32, Kaggle T4 x2, 2 h 32 min; code b73c04f, Ultralytics 8.4.211 |
| Weights | `best.pt` = epoch 49 (Ultralytics fitness on the synthetic val split) |

**Training curve.** Pose mAP50-95 on the synthetic val split reached 0.995 at epoch 13 and stayed
there; the val pose loss kept falling to the last epoch (0.200 at epoch 1, 0.0044 at epoch 50), so
there is no sign of overfitting. Train losses drop at epoch 41, where mosaic is switched off
(`close_mosaic=10`). The synthetic val metrics no longer separate the epochs, so models are
compared on MeterFL.

### MeterFL (ER-01, 262 real pressure-gauge photos)

| reader | cover | FS mean | FS median | ≤2% FS |
| --- | ---: | ---: | ---: | ---: |
| always mid-scale (floor) | 100% | 20.71 | 19.19 | 3% |
| Version 1, 15% grey margin (default) | 100% | **2.93** | 1.29 | **68%** |
| Version 1, no margin | 100% | 2.96 | 1.54 | 61% |
| target | | ≤ 2 | | ≥ 85% |

| subset | n | FS mean | FS median | ≤2% FS |
| --- | ---: | ---: | ---: | ---: |
| handheld | 135 | 2.89 | 1.40 | 63% |
| industrial | 115 | 2.57 | 1.24 | 74% |
| catalog | 6 | 6.10 | 1.41 | 67% |
| lab | 6 | 7.63 | 1.23 | 67% |

Model time (ER-07), PyTorch, batch 1, 640 px, median per image: 91 ms on a 4-core cloud CPU
(Intel Xeon, 2.3 GHz); 55 ms (MeasureBench) and 82 ms (synthetic val) on the work PC
(Intel Core i5-12400).

**Where the error comes from** (images inspected with `predict_readings.py --overlay`):

- **Dials turned far from upright.** The 9 worst images (3% of the set, 19–65% FS each) all show
  a gauge turned by 90° or more: eight gauges of one Chinese dial style photographed sideways or
  upside down, and one small hand-held gauge upside down. The model never saw a dial rolled by more than 15°, so it puts the
  scale-min and scale-max points at the bottom of the dial, where they sit on an upright gauge.
  These 9 images carry 38% of the total error; without them the mean is 1.87% FS and 70% are
  within 2% FS. Version 2 trains with rotation over the full circle.
- **Dials unlike the renders, and small gauges.** The next 4 (8–19% FS): a black car oil-pressure
  dial, two gauges that fill a small part of a wide photo (box confidence 0.36 for one), and a
  differential gauge with a centre zero.
- **One gauge, many frames.** The whole 0–0.1 MPa group, 123 of the 262 images, is frames of
  one hand-held gauge from a video (64% within 2% FS). Its errors are mostly 2–9% FS, and the
  overlays show the scale-min and scale-max points off the 0 and 0.1 ticks in many frames. With
  nearly half the test set on one gauge, MeterFL scores lean heavily on it.
- **No overall bias**: the median signed error is −0.17% FS.
- **Keypoint confidence hardly tracks the error** (Spearman −0.17; the worst image has keypoint
  confidence 1.00). Declining the 10% least confident images only lowers the mean from 2.93 to
  2.73% FS, so confidence alone cannot be the RETAKE signal (FR-09).

### Robustness (ER-05), MeterFL

One degradation at a time (`gauge_reader.eval.degrade`, seed 0), same reader as above.

| setting | cover | FS mean | ≤2% FS | Δ FS mean | Δ ≤2% FS |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 100% | 2.93 | 68% | | |
| low light 1 / 2 / 3 | 100 / 100 / 100% | 3.02 / 3.28 / 4.01 | 64 / 60 / 57% | +0.08 / +0.35 / +1.08 | −4 / −8 / −11 pp |
| radiation speckle 1 / 2 / 3 | 100 / 100 / 100% | 2.90 / 2.93 / 2.96 | 69 / 69 / 67% | −0.03 / −0.00 / +0.03 | +1 / +2 / −1 pp |
| motion blur 1 / 2 / 3 | 100 / 97 / 89% | 3.09 / 3.94 / 5.95 | 62 / 52 / 44% | +0.16 / +1.01 / +3.02 | −6 / −16 / −24 pp |

Radiation speckle up to 0.5% of the pixels does not change the readings. Low light and motion
blur do: at 25 px of blur 11% of the gauges are not found and the mean error doubles.

### Synthetic val split (2,000 images, no margin)

| face tilt | n | FS mean | FS median | ≤2% FS |
| --- | ---: | ---: | ---: | ---: |
| all | 2000 | 2.00 | 1.26 | 65% |
| 0–20° | 309 | 0.53 | 0.42 | 100% |
| 20–40° | 1008 | 1.51 | 1.16 | 73% |
| 40°+ | 683 | 3.37 | 2.79 | 39% |

Field conditions change the error little (no condition 1.89% FS, motion blur 2.20%), and
needle style and gauge size not at all. Degrading 500 of these images (ER-05 protocol) costs
at most +0.18 (low light), +0.06 (speckle) and +0.70% FS (motion blur) at severity 3, far less
than on the real MeterFL photos.

**The error is in the reading method, not the keypoints.** Reading the *true* keypoints with
the same image-plane angles (`gauge_reader.reading.geometry`) already gives most of it:

| face tilt | n | true keypoints: FS mean / ≤2% | model vs true-keypoint reading: FS mean / ≤2% | model: FS mean / ≤2% |
| --- | ---: | ---: | ---: | ---: |
| all | 2000 | 1.75 / 68% | 0.63 / 96% | 2.00 / 65% |
| 0–20° | 309 | 0.24 / 100% | 0.46 / 100% | 0.53 / 100% |
| 20–40° | 1008 | 1.27 / 78% | 0.59 / 98% | 1.51 / 73% |
| 40°+ | 683 | 3.14 / 39% | 0.77 / 93% | 3.37 / 39% |

Median keypoint error, in % of the gauge box's longer side: centre 0.59, tip 1.20, scale min
0.74, scale max 0.68. On a tilted face, angles measured around the centre in the image are not
the angles on the dial, so even perfect keypoints misread. Version 3 reads on the dial face.

### MeasureBench (ER-02, 142 real photos)

42 of the 142 images have a scale range (hand-labelled); the GT-range reader answers only
those. It read 38 of them and found no gauge in 4.

| images with a range | cover | FS mean | FS median | ≤2% FS | interval hit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 90% | 18.41 | 11.11 | 21% | 16% |

Main cause in each of the 38 answered images (overlays inspected one by one; no ground-truth
label looked wrong):

| cause | images |
| --- | ---: |
| needle tip put on the counterweight end, on the hub, or short of the tip | 6 |
| scale min/max points off the end ticks (3 of them dual-scale dials) | 5 |
| gauge small in a wide photo | 4 |
| unusual scale arc (differential gauge, near 360°, zero at 6 o'clock) | 3 |
| dial turned far from upright | 3 |
| several dials in the photo | 1 |
| non-linear scale | 1 |
| no clear failure (all under 4.2% FS) | 15 |

The 4 images without a detection: 3 small gauges (2 of them strongly tilted) and 1 dark photo.
Most failures are dial designs the generator does not make yet: two-ended needles, dual scales,
arcs outside 180–270°, and gauges far away in wide shots.

## Version 2 (training)

Same data and settings plus random rotation over the full circle (`--degrees 180`, the default
of `scripts/train_pose.py` since then; with the Version 3 code, add `--no-arc` to repeat it). A
turned dial still reads clockwise, so rotation, unlike a mirror flip, keeps every label valid.
Ultralytics fits the box around the turned box corners, so training boxes of turned gauges are
up to 1.4 times larger than the dial; readings use the keypoints only.

## Version 3 (next): reading on the dial face

Three more keypoints on the scale arc, at 25, 50 and 75% of the scale (7 in all), computed for
the existing synthetic set from its stored homographies (`scripts/arc_labels.py`, run by
`train_pose.py` by default; nothing is rendered again). The centre and the five scale points are
the image of a circle and its centre with the scale points evenly spread, so a homography from
the dial face to the photo can be fitted, and the needle is read on the face, where a tilted
view no longer distorts the angles (`gauge_reader.reading.perspective`). With the true keypoints
of generated renders the reading is exact at any tilt and roll (tests). Simulated on 100
renders, with the true points moved by random noise of 0.6% of the box (1.0% for the tip), close
to the Version 1 keypoint error:

| reading | FS mean | ≤2% FS | 40°+ tilt: FS mean / ≤2% |
| --- | ---: | ---: | ---: |
| image-plane angles, 4 keypoints | 1.70 | 72% | 2.27 / 61% |
| piecewise between the 5 scale points | 0.96 | 90% | 1.29 / 78% |
| dial-face fit (Version 3) | 0.69 | 98% | 0.72 / 97% |

The fit residual (`fit_rms`, as a share of the scale radius) tells how well the points agree
with a round dial; it is a candidate RETAKE signal, which keypoint confidence is not.
