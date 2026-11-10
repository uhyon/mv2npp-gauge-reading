# Datasets and licenses

No dataset is committed to this repository. `scripts/download_data.py` fetches them into `data/raw/`.
Checked on 2026-11-10 (Roboflow sets downloaded and inspected by hand).

## Evaluation plan

| # | Evaluation | Data | Metric | Notes |
| --- | --- | --- | --- | --- |
| 1 | **Main: reading accuracy** | MeterFL reading GT, 274 test images | % of full scale error, share within 2% FS | Includes a manual check of 30 images whose range was VLM-verified. Images are dial crops, so **this measures stages 2–5 only; gauge detection is not included** |
| 2 | Secondary: reading accuracy | MeasureBench real_world, pressure and temperature dials (filter below) | interval hit rate; % FS only on the subset whose range we label by hand | |
| 3 | Keypoints / needle fraction | Roboflow hustlegrad v4tii, 326 originals | keypoint error (px, normalized by box size), needle-fraction error | stretch correction below |
| 4 | Detection | Roboflow analog-6kh04 obaw9, 1,471 | mAP@0.5 on our own held-out split | |
| 5 | NPP-condition robustness | the real images of 1–4 with synthetic degradations | drop in each metric per condition and severity, vs. clean | see "Robustness protocol" |
| opt. | Cambridge RealGauges | only if the authors reply with a working link | % FS, angle error | comparability with Howells 2021 and Reitsma 2024 |

### Robustness protocol

Each condition is applied at three severities (1 mild, 2 moderate, 3 severe) to the clean test images, one condition at a time, with a fixed seed so every model sees the same images.

| Condition | What it imitates | Implementation (severity 1 / 2 / 3) |
| --- | --- | --- |
| Low light | dim plant rooms, night rounds | gamma 1.8 / 2.5 / 3.2, then Poisson shot noise scaled to the darkened image |
| Radiation speckle | single-event pixel hits on CMOS sensors in a radiation field | 0.05% / 0.2% / 0.5% of pixels set to saturated hot pixels, plus short streaks for 10% of hits |
| Motion blur | handheld or robot-mounted camera moving | linear kernel 7 / 15 / 25 px, random direction |
| Glare (optional) | glass cover reflection | elliptical specular blob over 5% / 15% / 25% of the dial |

The same degradations, with randomized parameters, are part of the synthetic training generator, so the test asks whether the model learned them, not whether it has seen these exact images.

## Limitations

- **No photos of actual nuclear power plant gauges are in the evaluation.** NPP field conditions are approximated by synthetic degradation of public industrial and web gauge images.
- The main reading test set (MeterFL) is dial crops, so end-to-end performance including detection is only measured separately on Roboflow obaw9.
- MeterFL ranges are partly VLM-verified, and 1,306 of its images have unrecorded sources.

## Datasets

| Dataset | Type / size | Labels | License | Use here |
| --- | --- | --- | --- | --- |
| [Cambridge SyntheticGauges](http://www.jjcvision.com/projects/gauge_reading.html) (Howells et al., CVPRW 2021) | synthetic, 11,000 imgs, 1024 px | COCO box + center/tip/min/max + face corners | CC BY-NC 4.0 | optional: **download link dead as of 2026-11-10** (Google Drive 404, no public mirror found); add only if the authors reply |
| [Cambridge RealGauges](http://www.jjcvision.com/projects/gauge_reading.html) | real, 6 gauges: detection, pose, 2,700 reading frames | center, plane normal, angle, value | CC BY-NC 4.0 | optional: **download link dead as of 2026-11-10**; data request to the authors pending. Never used in demos |
| [Roboflow: hustlegrad/analog-gauge-reading](https://universe.roboflow.com/hustlegrad/analog-gauge-reading-v4tii) (export source: `hustlegrad/gauge-detection-2-2ynzm` v7) | real web images, 326 originals (train 245 / valid 49 / test 32; train tripled by brightness aug to 735) | YOLOv8-pose, `kpt_shape [2,3]`, classes Gauge / Max / Min / Needle. Needle: pt1 = pivot, pt2 = tip. Min, Max: pt1 = tick position. Gauge: box only | CC BY 4.0 | evaluation 3 (keypoints, needle fraction; no reading GT). Demo images allowed with attribution |
| [Roboflow: analog-6kh04/analog-gauge](https://universe.roboflow.com/analog-6kh04/analog-gauge-obaw9) | real, 1,471, no augmentation, train split only | YOLOv8 segmentation polygons: center / meter / needle / scale | CC BY 4.0 | evaluation 4 (detection) on our own held-out split. No reading GT. Demo images allowed with attribution |
| [Synanthropic/reading-analog-gauge](https://huggingface.co/datasets/Synanthropic/reading-analog-gauge) | 42,442 imgs, 2.6 GB | dial corners, keypoints | MIT | auxiliary training, freely redistributable |
| [RPM-10K](https://github.com/Event-AHU/DialBench) (Wang et al., 2025) | real, 10,730, 300+ dial types | text reading only | dataset TBD (code BSD-3) | not used until a license is published |
| [MeasureBench real_world](https://huggingface.co/datasets/FlagEval/MeasureBench) (Lin et al., CVPR 2026) | real, 1,272 imgs over 26 instrument types; 711 with `design == dial` | accepted reading interval + unit, source URL per image. **No scale range** | CC BY-SA 4.0 | evaluation 2 after filtering. Not used in demos (images come from many third-party sources) |
| [MeterFL](https://github.com/weidazhaoooo/Meter-FL) (2026, commit 52f99ad checked 2026-11-10) | real, 1,382 images + 1,382 LabelMe JSON + 1,382 masks (all present in the repo, 174 MB). Folders by client, e.g. pressure_gauge_handheld 679, pressure_gauge_industrial 577 | tick/pointer polygons; `data/readings/*.json` = 274 test images with `unit, scale_min, scale_max, tick_values, reading_gt` | CC BY-SA 4.0 (code MIT) | evaluation 1 (**main % FS test set**): 262 of the 274 are pressure gauges (handheld 135, industrial 115, catalog 6, lab 6). Many crops are small (~200 px). 265 of 274 notes say the range was VLM-verified, so spot-check ~30 before trusting it. 1,306 images have unrecorded sources: evaluation only, never in demos |
| [AAU Pressure Gauge Reader Data](https://www.kaggle.com/datasets/juliusgrassme/pressure-gauge-reader-data) (Lauridsen et al.) | real videos from sewer pump stations | no public reading labels | CC BY 4.0 | qualitative video demo; frames could be labelled by hand |
| KAUST landmark set (León-Alcázar et al., WACV 2024) | real, 4,813 frames, servo-controlled gauge | needle angle, start/end markers | not released ([repo](https://github.com/fuankarion/automatic-gauge-reading) has only an empty README, single commit 2023-08-16) | unavailable |
| ETH test sets (Reitsma et al., ICRA 2024) | web images + proprietary refinery images | reading, unit, range | not released; [repo](https://github.com/ethz-asl/analog_gauge_reader) ships MIT code and a label-studio annotation recipe only, no data or weights | reuse its label format and evaluation script for our own set |
| [Own photos](OWN_TEST_SET.md) | — | — | — | **on hold**: shooting is not possible for this project. Protocol kept for later |

Caveats
- hustlegrad images were exported with a 640x640 stretch resize, and the export does not keep the original image sizes, so the stretch cannot be undone exactly. Estimate it per image from the Gauge box aspect ratio (dials are circular, assuming a roughly frontal view) and rescale x or y to make the box square before measuring angles. Report needle-fraction metrics alongside, and note the residual distortion for oblique views.
- For evaluation, dedupe train to its 245 originals (group files by the name prefix before `.rf.`) and keep valid (49) and test (32), which are not augmented. The evaluation set is the 326 originals.
- Neither Roboflow set has a reading ground truth. % of full scale error is measured on MeterFL (main) and the hand-labelled MeasureBench subset.
- MeasureBench labels are intervals and units only, with no full-scale range, so % FS cannot be computed directly. Primary metric: interval hit rate (prediction inside the accepted interval). % FS is reported only on a subset where we label `scale_min`/`scale_max` by hand.
- MeasureBench filtering (`scripts/filter_measurebench.py`): keep `split == real_world` and `design == dial`; keep `image_type` in pressure gauges, thermometers and other single-pointer process gauges (e.g. barometer, hygrometer, manometer); drop clocks, watches, ammeters, voltmeters, multimeters, utility/energy meters, speedometers, tachometers, compasses and any multi-pointer or multi-dial instrument. Units must be pressure or temperature (bar, kPa, MPa, psi, kgf/cm², mmH2O, inHg, °C, °F). The script prints counts per `image_type` so the final list can be checked by hand.
- CC BY-SA 4.0 is share-alike: a dataset we derive from MeasureBench or MeterFL (crops, relabels) must be released under CC BY-SA. Model weights and code are not affected.

Rules
- CC BY 4.0 data (Roboflow) requires attribution: cite the dataset URL and author in results and the README.
- CC BY-NC data is used for evaluation and research only; raw images are never re-uploaded.
- Demo examples (Hugging Face Space, README GIFs) use only our own synthetic renders and CC BY 4.0 Roboflow images (hustlegrad, analog-6kh04) with the source credited next to each image.
- Never in demos: CC BY-NC images (Cambridge) and images of unrecorded origin (MeterFL `voc_*`). MeasureBench images also stay out of demos.
- Cite every dataset used in results tables.
