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
| [Cambridge RealGauges](http://www.jjcvision.com/projects/gauge_reading.html) | real, 6 gauges: detection, pose, 2,700 reading frames | center, plane normal, angle, value | CC BY-NC 4.0 | optional: **download link dead as of 2026-11-10**. Author contact attempted 2026-11-10, addresses bounced (jjc75@cam.ac.uk, cipolla@eng.cam.ac.uk: 550 5.4.1 recipient not found). Never used in demos |
| [Roboflow: hustlegrad/analog-gauge-reading](https://universe.roboflow.com/hustlegrad/analog-gauge-reading-v4tii) (export source: `hustlegrad/gauge-detection-2-2ynzm` v7) | real web images, 326 originals (train 245 / valid 49 / test 32; train tripled by brightness aug to 735) | YOLOv8-pose, `kpt_shape [2,3]`, classes Gauge / Max / Min / Needle. Needle: pt1 = pivot, pt2 = tip. Min, Max: pt1 = tick position. Gauge: box only | CC BY 4.0 | evaluation 3 (keypoints, needle fraction; no reading GT). Demo images allowed with attribution |
| [Roboflow: analog-6kh04/analog-gauge](https://universe.roboflow.com/analog-6kh04/analog-gauge-obaw9) | real, 1,471, no augmentation, train split only | YOLOv8 segmentation polygons: center / meter / needle / scale | CC BY 4.0 | evaluation 4 (detection) on our own held-out split. No reading GT. Demo images allowed with attribution |
| [Synanthropic/reading-analog-gauge](https://huggingface.co/datasets/Synanthropic/reading-analog-gauge) | 42,442 imgs, 2.6 GB | dial corners, keypoints | MIT | auxiliary training, freely redistributable |
| [RPM-10K](https://github.com/Event-AHU/DialBench) (Wang et al., 2025) | real, 10,730, 300+ dial types | text reading only | dataset TBD (code BSD-3) | not used until a license is published |
| [MeasureBench real_world](https://huggingface.co/datasets/FlagEval/MeasureBench) (Lin et al., CVPR 2026) | real, 1,272 imgs over 26 instrument types; 711 with `design == dial` | accepted reading interval + unit, source URL per image. **No scale range** | CC BY-SA 4.0 | evaluation 2 after filtering (145 pressure and temperature dials after manual exclusions; 142 without barometers). Not used in demos (images come from many third-party sources) |
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
- MeasureBench filtering (`scripts/filter_measurebench.py`, run 2026-11-10): of 711 dials, keep `pressure_gauge` (118) and `thermometer` (31) by default. `sphygmomanometer` (21) is optional via `--keep-type sphygmomanometer`: these are aneroid gauges, usually with a metal-capsule mechanism rather than a Bourdon tube, and are included only because the dial looks like a pressure gauge. Units are normalized (case, spaces, ² → 2, spelled-out names) and must be a pressure or temperature unit, including kgf/cm², kg/cm², mbar, Pa, kPa, MPa, psi, inHg, mmHg, mmH2O, °C, °F. Dual-scale dials (several intervals and as many unit groups) pair each interval with its unit group by position; several intervals with one unit group are treated as alternative tolerance bands on one scale. `--report` prints drop reasons and the raw units of every unit-dropped sample. Runs on 2026-11-10 (653d51d, plural `intervals` field now read): **147 by default** (pressure_gauge 117, thermometer 30) or **168 with sphygmomanometer** (+21). 41 samples are dual-scale (pressure_gauge 39, thermometer 2; e.g. bar/psi); 2 dropped for empty units; no single-value answers. Earlier runs kept 97 (d023fb1) and 127 (21d2f66) because the `intervals` field was not read.
- Dual-scale scoring: `labels.csv` stores every (units, interval) pair in `alternatives`, and `gauge_reader.eval.interval_hit` compares the prediction with the interval of the scale whose unit matches the predicted unit, so a correct reading on either scale is a hit. A prediction without a unit is tried against all scales (lenient; report it separately). Single-value answers and zero-width intervals (lo == hi, e.g. pressure_gauge_3 6000 kPa, _4 850 bar, _5 985 hPa, _13 1017 hPa) are marked as point answers. All of them come from dual-scale samples, so the point takes the labeller's tolerance from the other scale, converted to its unit (e.g. _3: 800–900 psi → 5,516–6,205 kPa; _5: 738–739 mmHg → 983.9–985.3 hPa). `point_tol` (relative) is only a fallback when no convertible partner exists.
- Run 4bf43fa (2026-11-10): 147 kept (144 without barometers); 60 point answers = 13 `converted+interval` and **47 `point_tol`** (single scale, single value). A relative tolerance is unfair across ranges (2100 psi → ±42, 0.02 MPa → ±0.0004), so these 47 get a hand label: `--label-template data/processed/measurebench_point_labels.csv` writes `question_id, image_path, unit, reading, scale_min, scale_max, minor_div, notes`; fill range and smallest division, then `--manual-labels` sets the interval to reading ± 0.5 × minor_div (`tol = minor_div`) and fills `scale_min/scale_max`, so % FS is also computed on this subset. Until labelled, point_tol samples are reported separately (`hit_rate_by_group` on `tol_method`).
- Sanity flags (`flags` column, printed by every run, checked by hand with `--inspect <question_id>`, never auto-dropped; `--inspect` saves images to `<--out>/inspect/`, by default `data/processed/measurebench_dial/inspect/`): `temperature_unit_on_pressure_gauge` (e.g. pressure_gauge_6, °C), `possible_sphygmomanometer` (pressure_gauge in mmHg with 0 ≤ value < 320; negative values are vacuum gauges), `pressure_unit_on_thermometer`. Water-column units (`inches of water`, `inH2O`, `in. w.c.`, `"H2O`, e.g. _32, _88) normalize to inH2O, not inHg.
- Manual exclusions (`EXCLUDE_IDS`, checked with `--inspect` on 2026-11-10): `pressure_gauge_6` (Russian aneroid barometer, mmHg 700–800 / hPa 930–1070; the question reads its auxiliary 0–50 °C thermometer, answer 21 °C, so the barometer check missed it) and `pressure_gauge_39` (A&D Medical needle sphygmomanometer, 0–300 mmHg, bulb visible). Expected subset: **145 by default, 142 without barometers**. `labels.csv` records the method per sample in `tol_method` (`interval`, `converted`, `point_tol`, joined with `+`) and per scale in `alternatives[].tol`; `--report` prints the counts and each point sample.
- Barometers: some `pressure_gauge` samples are barometer faces (e.g. _5, _12, _13 in hPa/mmHg/inHg near 1,000 hPa). A sample is flagged `subtype = barometer` when a scale's unit is hPa, mbar, kPa, inHg or mmHg and its interval midpoint lies in the atmospheric band (870–1,085 hPa or the equivalent). They are kept by default and listed by `--report`; `--exclude-type barometer` drops them. Report results with and without them.
- CC BY-SA 4.0 is share-alike: a dataset we derive from MeasureBench or MeterFL (crops, relabels) must be released under CC BY-SA. Model weights and code are not affected.

Rules
- CC BY 4.0 data (Roboflow) requires attribution: cite the dataset URL and author in results and the README.
- CC BY-NC data is used for evaluation and research only; raw images are never re-uploaded.
- Demo examples (Hugging Face Space, README GIFs) use only our own synthetic renders and CC BY 4.0 Roboflow images (hustlegrad, analog-6kh04) with the source credited next to each image.
- Never in demos: CC BY-NC images (Cambridge) and images of unrecorded origin (MeterFL `voc_*`). MeasureBench images also stay out of demos.
- Cite every dataset used in results tables.
