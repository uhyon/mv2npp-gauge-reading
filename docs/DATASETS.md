# Datasets and licenses

No dataset is committed to this repository. `scripts/download_data.py` fetches them into `data/raw/`.
Checked on 2026-11-10 (Roboflow sets downloaded and inspected by hand).

| Dataset | Type / size | Labels | License | Use here |
| --- | --- | --- | --- | --- |
| [Cambridge SyntheticGauges](http://www.jjcvision.com/projects/gauge_reading.html) (Howells et al., CVPRW 2021) | synthetic, 11,000 imgs, 1024 px | COCO box + center/tip/min/max + face corners | CC BY-NC 4.0 | **unavailable: download link dead as of 2026-11-10** (Google Drive 404, no public mirror found) |
| [Cambridge RealGauges](http://www.jjcvision.com/projects/gauge_reading.html) | real, 6 gauges: detection, pose, 2,700 reading frames | center, plane normal, angle, value | CC BY-NC 4.0 | **unavailable: download link dead as of 2026-11-10**. Data request to the authors pending |
| [Roboflow: hustlegrad/analog-gauge-reading](https://universe.roboflow.com/hustlegrad/analog-gauge-reading-v4tii) (export source: `hustlegrad/gauge-detection-2-2ynzm` v7) | real web images, 326 originals (train 245 / valid 49 / test 32; train tripled by brightness aug to 735) | YOLOv8-pose, `kpt_shape [2,3]`, classes Gauge / Max / Min / Needle. Needle: pt1 = pivot, pt2 = tip. Min, Max: pt1 = tick position. Gauge: box only | CC BY 4.0 | keypoint and needle-fraction evaluation only (no min/max numbers, no reading GT, so no % FS error) |
| [Roboflow: analog-6kh04/analog-gauge](https://universe.roboflow.com/analog-6kh04/analog-gauge-obaw9) | real, 1,471, no augmentation, train split only | YOLOv8 segmentation polygons: center / meter / needle / scale | CC BY 4.0 | detection and needle segmentation evaluation; we create our own val/test split. No reading GT |
| [Synanthropic/reading-analog-gauge](https://huggingface.co/datasets/Synanthropic/reading-analog-gauge) | 42,442 imgs, 2.6 GB | dial corners, keypoints | MIT | auxiliary training, freely redistributable |
| [RPM-10K](https://github.com/Event-AHU/DialBench) (Wang et al., 2025) | real, 10,730, 300+ dial types | text reading only | dataset TBD (code BSD-3) | not used until a license is published |
| [MeasureBench real_world](https://huggingface.co/datasets/FlagEval/MeasureBench) (Lin et al., CVPR 2026) | real, 1,272 imgs over 26 instrument types; 711 with `design == dial` | accepted reading interval + unit, source URL per image. **No scale range** | CC BY-SA 4.0 | secondary test set after filtering (see below). Main metric: interval hit rate. % FS only on a subset whose range we label by hand |
| [MeterFL](https://github.com/weidazhaoooo/Meter-FL) (2026, commit 52f99ad checked 2026-11-10) | real, 1,382 images + 1,382 LabelMe JSON + 1,382 masks (all present in the repo, 174 MB). Folders by client, e.g. pressure_gauge_handheld 679, pressure_gauge_industrial 577 | tick/pointer polygons; `data/readings/*.json` = 274 test images with `unit, scale_min, scale_max, tick_values, reading_gt` | CC BY-SA 4.0 (code MIT) | **% FS test set**: 262 of the 274 are pressure gauges (handheld 135, industrial 115, catalog 6, lab 6). Many crops are small (~200 px). 265 of 274 notes say the range was VLM-verified, so spot-check ~30 before trusting it. 1,306 images have unrecorded sources: evaluation only, never in demos |
| [AAU Pressure Gauge Reader Data](https://www.kaggle.com/datasets/juliusgrassme/pressure-gauge-reader-data) (Lauridsen et al.) | real videos from sewer pump stations | no public reading labels | CC BY 4.0 | qualitative video demo; frames could be labelled by hand |
| KAUST landmark set (León-Alcázar et al., WACV 2024) | real, 4,813 frames, servo-controlled gauge | needle angle, start/end markers | not released ([repo](https://github.com/fuankarion/automatic-gauge-reading) has only an empty README, single commit 2023-08-16) | unavailable |
| ETH test sets (Reitsma et al., ICRA 2024) | web images + proprietary refinery images | reading, unit, range | not released; [repo](https://github.com/ethz-asl/analog_gauge_reader) ships MIT code and a label-studio annotation recipe only, no data or weights | reuse its label format and evaluation script for our own set |
| [Own photos](OWN_TEST_SET.md) (planned, by 2026-11-18) | real, 120 min / 240 target, 20–30+ gauges × 6–8 shots | reading, range, unit, Gauge box, keypoints | ours, CC BY 4.0 | **primary reading test set** and the only images used in demos |

Caveats
- hustlegrad images were exported with a 640x640 stretch resize, and the export does not keep the original image sizes, so the stretch cannot be undone exactly. Estimate it per image from the Gauge box aspect ratio (dials are circular, assuming a roughly frontal view) and rescale x or y to make the box square before measuring angles. Report needle-fraction metrics alongside, and note the residual distortion for oblique views.
- For evaluation, dedupe train to its 245 originals (group files by the name prefix before `.rf.`) and keep valid (49) and test (32), which are not augmented. The evaluation set is the 326 originals.
- Neither Roboflow set has a reading ground truth. % of full scale error is measured on our own photos, MeasureBench (dial subset) and MeterFL test.
- MeasureBench labels are intervals and units only, with no full-scale range, so % FS cannot be computed directly. Primary metric: interval hit rate (prediction inside the accepted interval). % FS is reported only on a subset where we label `scale_min`/`scale_max` by hand.
- MeasureBench filtering (`scripts/filter_measurebench.py`): keep `split == real_world` and `design == dial`; keep `image_type` in pressure gauges, thermometers and other single-pointer process gauges (e.g. barometer, hygrometer, manometer); drop clocks, watches, ammeters, voltmeters, multimeters, utility/energy meters, speedometers, tachometers, compasses and any multi-pointer or multi-dial instrument. Units must be pressure or temperature (bar, kPa, MPa, psi, kgf/cm², mmH2O, inHg, °C, °F). The script prints counts per `image_type` so the final list can be checked by hand.
- CC BY-SA 4.0 is share-alike: a dataset we derive from MeasureBench or MeterFL (crops, relabels) must be released under CC BY-SA. Model weights and code are not affected.

Rules
- CC BY 4.0 data (Roboflow) requires attribution: cite the dataset URL and author in results and the README.
- CC BY-NC data is used for evaluation and research only; raw images are never re-uploaded.
- Demo examples (Hugging Face Space, README GIFs) use only our own photos and our own synthetic renders.
- Cite every dataset used in results tables.
