# Datasets and licenses

No dataset is committed to this repository. `scripts/download_data.py` fetches them into `data/raw/`.
Checked on 2026-11-10 (Roboflow sets downloaded and inspected by hand).

| Dataset | Type / size | Labels | License | Use here |
| --- | --- | --- | --- | --- |
| [Cambridge SyntheticGauges](http://www.jjcvision.com/projects/gauge_reading.html) (Howells et al., CVPRW 2021) | synthetic, 11,000 imgs, 1024 px | COCO box + center/tip/min/max + face corners | CC BY-NC 4.0 | baseline synthetic set to compare against ours |
| [Cambridge RealGauges](http://www.jjcvision.com/projects/gauge_reading.html) | real, 6 gauges: detection, pose, 2,700 reading frames | center, plane normal, angle, value | CC BY-NC 4.0 | **main real-world test set** |
| [Roboflow: hustlegrad/analog-gauge-reading](https://universe.roboflow.com/hustlegrad/analog-gauge-reading-v4tii) (export source: `hustlegrad/gauge-detection-2-2ynzm` v7) | real web images, 326 originals (train 245 / valid 49 / test 32; train tripled by brightness aug to 735) | YOLOv8-pose, `kpt_shape [2,3]`, classes Gauge / Max / Min / Needle. Needle: pt1 = pivot, pt2 = tip. Min, Max: pt1 = tick position. Gauge: box only | CC BY 4.0 | keypoint and needle-fraction evaluation only (no min/max numbers, no reading GT, so no % FS error) |
| [Roboflow: analog-6kh04/analog-gauge](https://universe.roboflow.com/analog-6kh04/analog-gauge-obaw9) | real, 1,471, no augmentation, train split only | YOLOv8 segmentation polygons: center / meter / needle / scale | CC BY 4.0 | detection and needle segmentation evaluation; we create our own val/test split. No reading GT |
| [Synanthropic/reading-analog-gauge](https://huggingface.co/datasets/Synanthropic/reading-analog-gauge) | 42,442 imgs, 2.6 GB | dial corners, keypoints | MIT | auxiliary training, freely redistributable |
| [RPM-10K](https://github.com/Event-AHU/DialBench) (Wang et al., 2025) | real, 10,730, 300+ dial types | text reading only | dataset TBD (code BSD-3) | not used until a license is published |

Caveats
- hustlegrad images were exported with a 640x640 stretch resize, which distorts the aspect ratio and therefore needle angles. Evaluate angles after undoing the stretch (scale x and y back to the original size) or use the needle fraction between the Min and Max points, which is less sensitive.
- Use only the 245 original train images (drop the brightness-augmented copies) when the set is used for evaluation, so no image is counted three times.
- Neither Roboflow set has a reading ground truth. % of full scale error is measured on Cambridge RealGauges and on our own photos.

Rules
- CC BY 4.0 data (Roboflow) requires attribution: cite the dataset URL and author in results and the README.
- CC BY-NC data is used for evaluation and research only; raw images are never re-uploaded.
- Demo examples (Hugging Face Space, README GIFs) use only our own photos and our own synthetic renders.
- Cite every dataset used in results tables.
