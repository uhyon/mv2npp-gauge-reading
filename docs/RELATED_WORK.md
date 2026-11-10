# Related work (phase 1 notes, 2026-11-10)

| # | Work | Key idea | Takeaway |
| --- | --- | --- | --- |
| 1 | León-Alcázar et al., [Learning to Read Analog Gauges from Synthetic Data](https://arxiv.org/abs/2308.14583), WACV 2024 | 3D synthetic gauges, two-stage CNN, 4,813 real test images, 4.55° lower mean error | closest design; code/data not released |
| 2 | Howells, Charles, Cipolla, [Real-time analogue gauge transcription on mobile phone](http://www.jjcvision.com/projects/gauge_reading.html), CVPRW 2021 | Blender with 50 parameters, synthetic-only training, <1° needle error, 25 fps | generator design reference; keypoint format |
| 3 | Reitsma et al., [Under Pressure](https://arxiv.org/abs/2404.08785), ICRA 2024 | detect, segment, ellipse fit, OCR + RANSAC; ~2% relative error | open baseline ([MIT code](https://github.com/ethz-asl/analog_gauge_reader)) |
| 4 | Wang et al., [DialBench / RPM-10K](https://arxiv.org/abs/2511.21982), 2025 | 10,730 real images, 18 MLLMs benchmarked | VLM comparison |
| 5 | Salomon et al., [Image-based Automatic Dial Meter Reading](https://arxiv.org/abs/2005.03106), IJCNN 2020 | UFPR-ADMR dial meters | error analysis style |
| 6 | Cho et al. (KAERI), [Dataset Construction System for AI-based Analog Gauge Reading in Nuclear Facility](https://www.kns.org/files/pre_paper/54/25A-420-%EC%A1%B0%EC%9A%A9%ED%98%B8.pdf), KNS Autumn 2025 | motor-driven real gauges for auto labels, quadruped robot patrol | same problem solved with hardware; we use synthesis |
| 7 | Cho et al. (KAERI), [Automated Analog Gauge Reading System for Mobile Robots](https://doi.org/10.23919/iccas66577.2025.11301236), ICCAS 2025 | ellipse + perspective + DenseNet169; <1° needle error | Korean NPP state of the art |
| 8 | Cho et al. (KAERI), Chain of Thought with MLLM for Analog Gauge Reading, UR 2025 | MLLM + CoT | VLM comparison |
| 9 | Al Rashdan et al. (INL), [CVML in Safety-Related NPP Applications](https://www.osti.gov/biblio/2279025), INL/RPT-23-74920 | gauge reading vs. digital I&C safety requirements | motivation; confidence-based retake |
| 10 | Kim et al., [Digitalization of Analog Gauge using Image Processing](https://koreascience.or.kr/article/JAKO202322843272481.page), 2023 | classic CV for ship engine-room robots | classic baseline |
