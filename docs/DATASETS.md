# Datasets and licenses

No dataset is committed to this repository. `scripts/download_data.py` fetches them into `data/raw/`.
Checked on 2026-11-10.

| Dataset | Type / size | Labels | License | Use here |
| --- | --- | --- | --- | --- |
| [Cambridge SyntheticGauges](http://www.jjcvision.com/projects/gauge_reading.html) (Howells et al., CVPRW 2021) | synthetic, 11,000 imgs, 1024 px | COCO box + center/tip/min/max + face corners | CC BY-NC 4.0 | baseline synthetic set to compare against ours |
| [Cambridge RealGauges](http://www.jjcvision.com/projects/gauge_reading.html) | real, 6 gauges: detection, pose, 2,700 reading frames | center, plane normal, angle, value | CC BY-NC 4.0 | **main real-world test set** |
| [Roboflow: hustlegrad/analog-gauge-reading](https://universe.roboflow.com/hustlegrad/analog-gauge-reading-v4tii) | real web images, 326 (816 augmented) | keypoints: gauge, needle center/tip, min, max | verify on page | keypoint evaluation on varied dials |
| [Roboflow: analog-6kh04/analog-gauge](https://universe.roboflow.com/analog-6kh04/analog-gauge-obaw9) | real, ~1,500 | instance masks: gauge, needle, center, numbers | verify on page | detection / OCR evaluation |
| [Synanthropic/reading-analog-gauge](https://huggingface.co/datasets/Synanthropic/reading-analog-gauge) | 42,442 imgs, 2.6 GB | dial corners, keypoints | MIT | auxiliary training, freely redistributable |
| [RPM-10K](https://github.com/Event-AHU/DialBench) (Wang et al., 2025) | real, 10,730, 300+ dial types | text reading only | dataset TBD (code BSD-3) | not used until a license is published |

Rules
- CC BY-NC data is used for evaluation and research only; raw images are never re-uploaded.
- Demo examples (Hugging Face Space, README GIFs) use only our own photos and our own synthetic renders.
- Cite every dataset used in results tables.
