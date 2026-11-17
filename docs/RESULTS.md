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
  within 2% FS. Version 2 trained with rotation over the full circle, which fixed most of these but
  broke the rest (below).
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

## Version 2 (2026-11-16): rotation over the full circle

Same data and settings plus random rotation over the full circle (`--degrees 180`; with the
Version 3 code, add `--no-arc --degrees 180` to repeat it). yolo11s-pose, 50 epochs on Kaggle
(T4 x2). A turned dial still reads clockwise, so rotation keeps every label valid, but the
result is much worse than Version 1:

| MeterFL, 262 images (GT range, margin 15%) | Version 1 | Version 2 |
| --- | ---: | ---: |
| mean % FS | 2.93 | 10.18 |
| median % FS | 1.29 | 3.73 |
| within 2% FS (all images) | 68% | 32% |
| coverage | 100% | 96% (8 bad geometry, 3 no detection) |
| scale-min point distance from centre / needle length (median) | 1.08 | 0.62 |
| synthetic val pose loss, last epoch | 0.0044 | 0.0135 |

It did fix most of the dials turned by 90° or more (for example 65.2 → 6.6% FS and
30.0 → 0.9% FS), and the needle centre and tip stay accurate. But on most upright photos the
scale-min and scale-max points now sit well inside the dial, near the centre. Rotated over the
full circle, the model no longer knows which way is up on a real photo; the renders give it
cues that the real dials do not, and where it is unsure it puts the end points between the
possible positions. The val loss was still falling slowly at epoch 50, but not fast enough to
close a gap of three times.

Test-time rotation with Version 1 does not help either. Feeding each image turned by 0, 90, 180
and 270° and keeping the best of the four would give 1.74% FS (70% within 2%) if the right turn
could be picked, but picking by keypoint confidence gives 24.3% FS and picking the turn that puts
Version 2's scale gap at the bottom gives 10.6% FS.

So Version 3 rotates by up to ±30° only (`--degrees 30`, the default from now on), on top of the
renders' own roll of up to 15°. Field photos of a round are taken upright; a dial turned by 90°
or more is left to RETAKE.

## Version 3 (2026-11-16): reading on the dial face

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

### Version 3 results

yolo11s-pose, 7 keypoints, rotation ±30°, 50 epochs on Kaggle (T4 x2, 2 h 27 min), run name `arc`.

| | Version 1 | Version 3 |
| --- | ---: | ---: |
| MeterFL mean % FS (262) | 2.93 | 2.83 |
| MeterFL median % FS | 1.29 | 1.09 |
| MeterFL within 2% FS | 68% | 74% |
| MeterFL without the 9 dials turned 90°+: mean / within 2% | 1.87 / 70% | 1.77 / 77% |
| Synthetic val mean % FS (2,000, no margin) | 2.00 | 0.66 |
| Synthetic val within 2% FS | 65% | 96% |
| Synthetic val, tilt 0–20° / 20–40° / 40°+ | 0.53 / – / 3.37 | 0.49 / 0.62 / 0.79 |
| MeasureBench (38 read of 42 ranged): mean / median % FS | 18.41 / 11.11 | 15.94 / 8.32 |
| MeasureBench within 2% FS / interval hit | 21% / 16% | 34% / 24% |

On the synthetic val split the dial-face fit removes almost all of the tilt error, as the
simulation predicted; reading the same Version 3 keypoints with image-plane angles
(`fraction_img`) gives 1.91% FS, so the gain comes from the fit, not from better keypoints.
On MeterFL, where most photos are frontal, the dial-face reading and the image-plane reading
are about equal (2.83 vs 2.74% FS), and the gain over Version 1 is small. The scale end points
stay at the dial edge (scale-min distance / needle length 1.10), unlike Version 2. The same 9
MeterFL dials turned by 90° or more still fail and make 39% of the total error; they are left
to RETAKE. The fit residual `fit_rms` hardly tracks the error (Spearman −0.15 on MeterFL, 0.15 on
synthetic val), so it is not a RETAKE signal on its own.


## Version 4 (2026-11-17): larger model

Same as Version 3 with yolo11m-pose instead of yolo11s-pose (run name `arc_m`), 50 epochs on
Kaggle (T4 x2, 4 h 40 min). Final synthetic val pose loss 0.0131 (Version 3: 0.0152).

| | Version 3 (11s) | Version 4 (11m) |
| --- | ---: | ---: |
| MeterFL mean / median % FS (262) | 2.83 / 1.09 | 2.69 / 1.35 |
| MeterFL within 2% FS | 74% | 77% |
| MeterFL without the 9 dials turned 90°+: mean / within 2% | 1.77 / 77% | 1.67 / 80% |
| Synthetic val mean / median % FS | 0.66 / 0.49 | 0.75 / 0.55 |
| Synthetic val within 2% FS | 96% | 95% |
| Synthetic val, tilt 0–20° / 20–40° / 40°+ | 0.49 / 0.62 / 0.79 | 0.56 / 0.71 / 0.89 |
| MeasureBench read (of 42 ranged) | 38 | 41 |
| MeasureBench mean / median % FS | 15.94 / 8.32 | 13.34 / 4.48 |
| MeasureBench within 2% FS / interval hit (read images) | 34% / 24% | 39% / 27% |
| Model time per image, work PC CPU (i5-12400), synthetic val | 79 ms | 237 ms |

The larger model does not change the main result. On MeterFL the paired difference in mean
error is −0.16% FS with a bootstrap 95% interval of [−0.44, +0.08]; 55 images got better and
58 worse by more than 1% FS. It reads the 0–0.1 MPa video gauge better (1.82 → 1.45% FS, 123
images) but the 0–25 MPa gauge worse (1.13 → 1.98% FS, 48 images, about 1.5% FS low on
average). On synthetic val it is slightly worse at every tilt. MeasureBench improves, mostly
on the 3–60 psi dials (7 images, 13.56 → 3.02% FS), and it finds 3 more gauges, but only about
40 images are scored. The 9 MeterFL dials turned by 90°+ still fail, and `fit_rms` still does
not track the error (Spearman 0.04 on MeterFL). Both models meet the 0.5 s target, but Version 4
is three times slower. Version 3 stays the default; the remaining real-photo error looks like a
data gap (dial designs and orientations the generator does not make), not a capacity limit.

## Keypoints (Roboflow v4tii) and detection (Roboflow obaw9), Versions 3 and 4

Run with `scripts/eval_roboflow.py` on the work PC (2026-11-17), no margin unless stated. These
sets have no reading ground truth: the needle fraction from the labelled points (after undoing
the stretch resize) stands in for it, so "% of scale" below is comparable to % FS.

**Keypoints, v4tii** (326 originals, 344 labelled gauges, 339 with all four points):

| | Version 3 | Version 4 | Version 3, margin 15% |
| --- | ---: | ---: | ---: |
| gauges detected (box IoU ≥ 0.3) | 333 (97%) | 326 (95%) | 335 (97%) |
| centre: median error / within 5% of gauge size | 1.4% / 96% | 1.5% / 97% | 1.6% / 96% |
| needle tip | 2.7% / 70% | 2.7% / 73% | 2.6% / 76% |
| scale min | 5.5% / 42% | 5.5% / 41% | 5.9% / 36% |
| scale max | 6.3% / 29% | 6.7% / 25% | 6.4% / 28% |
| needle fraction, image plane: mean / median / ≤2% / ≤5% | 3.00 / 1.22 / 70% / 93% | 3.84 / 1.34 / 65% / 92% | 5.02 / 1.42 / 63% / 89% |
| needle fraction, dial face | 3.13 / 1.42 / 67% / 92% | 3.97 / 1.24 / 66% / 92% | 5.12 / 1.43 / 64% / 89% |
| model time, work PC CPU | 85 ms | 236 ms | 91 ms |

The centre and the needle tip are found well; the scale end points are the weak keypoints
(within 5% of the gauge size for only 29–42%), as on MeterFL and MeasureBench. Part of that may
be the labels: a v4tii Min/Max point marks the end tick, which may sit at a different radius
than the point on the scale arc that our renders label (not checked image by image yet). The needle fraction still lands within 5% of scale for 93% of
gauges, because an end point that is off along its own ray does not change the angle. A few
gross errors pull the mean up: the 5 worst Version 3 gauges are off by 50–93% of scale with a
good box (IoU 0.71–0.95), so the needle or the scale ends were read the wrong way round. Image-
plane and dial-face readings agree here (most v4tii photos are near frontal, and the stretch
correction already assumes it). A margin does not help on these photos, which, unlike the
MeterFL crops, already show background around the gauge. Version 4 is again no better.

**Detection**: AP of the gauge box, all detections down to conf 0.001.

| | Version 3: AP50 / AP50-95 | Version 4 | Version 3 precision / recall at conf 0.25 |
| --- | ---: | ---: | ---: |
| obaw9, 1,471 images, 1,461 `meter` boxes | 0.955 / 0.391 | 0.953 / 0.403 | 0.969 / 0.948 |
| v4tii, 326 images, 344 Gauge boxes | 0.970 / 0.746 | 0.965 / 0.725 | 0.997 / 0.962 |

A model trained only on renders finds about 95% of the real gauges at 97% precision. The low
obaw9 AP50-95 (v4tii Gauge boxes give 0.75) most likely comes from the box definition, the
`meter` polygon outlining a different part of the gauge than our box, rather than from misses;
until that is checked on overlays, only AP50 is used for obaw9.

## Rotation check and RETAKE (Version 3, 2026-11-17)

`predict_readings.py --tta` reads each image upright and turned by ±25°. A reading does not
change when the photo is turned, so the value is the median of the three, and their spread
(max − min needle fraction, % of scale) says how sure the model is. Above 5% the reader
declines with status `retake` and asks for another photo.

First candidate, rejected: a predicted scale arc over 285°. On MeterFL it flags exactly the 10
dials turned by 90°+ (all over 10% FS error), but the generator only makes 180–270° arcs, so it
never fires on the synthetic split, and on real sets it misses most large errors (MeasureBench
4 of 18) and flags real 270–300° dials that read fine (v4tii: 13 of 15 flagged were within 10%).

| Version 3 | MeterFL (262) | MeasureBench (38 read) | Synthetic val (2,000) |
| --- | ---: | ---: | ---: |
| upright: mean / median % FS / ≤2% | 2.83 / 1.09 / 75% | 15.94 / 8.32 / 34% | 0.66 / 0.49 / 96% |
| median of 3: mean / median / ≤2% | 2.44 / 0.99 / 77% | 15.02 / 5.65 / 34% | 0.59 / 0.44 / 97% |
| Spearman(spread, error) | +0.38 | +0.70 | +0.29 |
| flagged at spread > 5% | 9% | 55% | 1% |
| large errors (>10% FS) caught at 5% | 7 of 14 | 14 of 18 | 2 of 2 |
| kept after RETAKE at 5%: mean / median / ≤2% | 1.85 / 0.97 / 81% | 4.80 / 1.41 / 65% | 0.57 / 0.44 / 97% |
| alternatives: flagged at 3% / 8% | 20% / 4% | 71% / 42% | 6% / 0.4% |

The kept rows use the median reading. The spread follows the error far better than keypoint
confidence (Spearman −0.17 on MeterFL, Version 1) or the dial-fit residual `fit_rms` (−0.15).
At 3% the kept MeterFL images reach the ER-01 share (1.60% FS, 85% within 2%) but a fifth of
the photos are sent back; 5% was chosen as the balance for a field round. MeasureBench is sent
back more often because many of its dials are designs the generator does not make. The check
costs three model passes (MeterFL in the cloud container: 342 ms per image; about 240 ms
expected on the work PC).

## Scale OCR (stage 4, FR-07), 2026-11-17

`scripts/ocr_scale.py` reads the numbers on the dial with rapidocr-onnxruntime (PP-OCRv3 models
shipped with the package, CPU) and places each number on the scale by its angle around the
predicted centre, on the dial face when the arc keypoints give a fit
(`gauge_reader.reading.scale_ocr`). A RANSAC line value = a + b × fraction through the numbers
gives the range at fraction 0 and 1, snapped to a printed label or the label grid; at least 3
distinct labels must lie on it, otherwise no range is given (`no_scale`). Two OCR habits are
handled: a dropped decimal point ("08" read for 0.8, "16" for 1.6: each whole number also
tries its decimal versions, the line decides) and a lost superscript ("Kg/cm" → kgf/cm²). A
lone "Pa" is not taken as a unit, since it is nearly always "MPa" or "kPa" with the prefix
missed. Keypoints from Version 3.

| MeterFL, 262 images | GT range (needle only) | OCR range | OCR range + rotation check (RETAKE 5%) |
| --- | ---: | ---: | ---: |
| range given | – | 235 (90%) | 235 (90%) |
| range correct (both ends within 1% of span) | – | 233 of 235 given (99%); 233 of 262 (89%) | same |
| unit read / correct | – | 138 / 138 | same |
| coverage | 100% | 90% | 82% |
| mean / median % FS of read images | 2.83 / 1.09 | 1.78 / 1.09 | 1.24 / 0.97 |
| within 2% FS: read / all images | 74% / 74% | 76% / 68% | 81% / 66% |
| model time per image (cloud container) | – | OCR 208 ms | 342 + 208 ms |

When OCR gives a range it is almost always right: the 2 wrong ranges are a dial turned by 90°+
(keypoints wrong, so the numbers land on the wrong fractions) and one read as −5–25 instead
of 0–25. The images without a range are mostly one 0–10 MPa gauge (17 of 26) whose small scale
numbers the OCR does not detect, even at twice the size. Declining those removes several of the
largest needle errors too, so the read images get better than with the GT range. With the
rotation check as well, the read images reach 1.24% FS mean (ER-01: ≤ 2%) at 82% coverage.
Units: MeterFL dials print "MPa" or "kPa" on 138 images; on the other 124 (mostly the 0–0.1 MPa
video gauge, where the OCR sees "Pa") no unit is read and the label unit is assumed.

MeasureBench (142 images, run on the work PC; OCR on a crop of 1.6 scale radii around the
predicted centre). Rows without a hand-labelled range now keep their keypoints, so OCR gives a
value for any detected dial and the interval hit can be scored on all 142:

| MeasureBench, Version 3 | GT range (42 labelled) | OCR range | OCR range + rotation check |
| --- | ---: | ---: | ---: |
| OCR range given / correct of the 42 labelled | – | 30 / 14 | same |
| coverage, all 142 | 27% | 75% | 42% |
| interval hit: read / all 142 | 24% / 6% | 24% / 18% | 33% / 14% |
| % FS on labelled images read: mean / median / ≤2% | 15.94 / 8.32 / 34% | 26.50 / 2.82 / 43% | 1.86 / 1.47 / 67% |
| OCR time, work PC CPU | – | 397 ms | 426 ms |

MeasureBench is far harder for OCR than MeterFL. Of the 16 wrong ranges among the labelled
images, 8 are tyre gauges labelled 3–60 or 4–100 psi (the hand labels take the first tick, the
dial prints 0 or no number there) that OCR reads as 0–60 / 0–100; the rest are dials with two
scale rings (inner bar ring mixed with the outer psi ring, a hygrometer % ring read as the
thermometer) and extrapolated lines (−40–90 for 0–50 inH2O). The rotation check removes most of
the damage: the read images reach 1.86% FS mean. OCR on the PC CPU (about 400 ms) is the
slowest stage, so the full pipeline is about 0.6 s per image; speeding OCR up (smaller
detector input, recognition only on boxes near the scale) is open.

## End to end: `GaugeReader.read(photo)` (2026-11-17)

`gauge_reader.reader.GaugeReader` runs all stages on one photo and returns value, unit,
status and alarm (LOW / HIGH / NORMAL with a band, RETAKE when no value is given);
`scripts/read_gauge.py` reads single photos (JSON lines) or a manifest (predictions CSV for
`evaluate.py`). Detection runs on the photo with a 15% grey margin; when the box covers at
least 40% of the photo that pass is also the upright keypoint pass, otherwise the box (plus 5%)
is cropped and read again. Turned ±25° passes, the 5% spread RETAKE and OCR follow. Reading a
close-up through a crop of its own box instead let 2–5x more MeterFL photos fall to RETAKE
(the framing matters), so close-ups are read whole. OCR skips the text-angle classifier
(dial numbers are upright): 225 → 171 ms per image in the container with the same ranges.

| MeterFL, 262, Version 3 | coverage | mean / median % FS | ≤2% FS: read / all | time per image (container) |
| --- | ---: | ---: | ---: | ---: |
| GT range (needle only) | 92% | 2.08 / 0.97 | 80% / 74% | 502 ms |
| OCR range (full pipeline) | 83% | 1.88 / 0.95 | 81% / 67% | 603 ms |

Declined: 22 RETAKE, 22 no OCR range, 1 no gauge. On the read photos the full pipeline meets
the ER-01 mean (≤ 2% FS) but not the 85% share within 2% FS.
