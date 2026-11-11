# Related work (phase 1 notes, updated 2026-11-10)

"How we differ" refers to this project: synthetic-only training, NPP degradation robustness
(low light, radiation speckle, motion blur), a RETAKE decision when the reading is unreliable,
and a CPU target of <= 0.5 s per image. The RETAKE decision and the latency are design targets
(README), not results yet.

| # | Work | Key idea | Takeaway | How we differ |
| --- | --- | --- | --- | --- |
| 1 | León-Alcázar et al., [Learning to Read Analog Gauges from Synthetic Data](https://arxiv.org/abs/2308.14583), WACV 2024 | 3D synthetic gauges, two-stage CNN, 4,813 real test images, 4.55° lower mean error | closest design; code and data not released | open code; NPP degradation tests; RETAKE output |
| 2 | Howells, Charles, Cipolla, [Real-time analogue gauge transcription on mobile phone](http://www.jjcvision.com/projects/gauge_reading.html), CVPRW 2021 | Blender with 50 parameters, synthetic-only training, <1° needle error, 25 fps | generator design reference; keypoint format. **Dataset link dead as of 2026-11-10; author emails bounced** (see [DATASETS.md](DATASETS.md)) | NPP conditions and a reject option; their data cannot be used |
| 3 | Reitsma et al., [Under Pressure](https://arxiv.org/abs/2404.08785), ICRA 2024 | detect, segment, ellipse fit, OCR + RANSAC; ~2% relative error | open baseline ([MIT code](https://github.com/ethz-asl/analog_gauge_reader)) | single learned keypoint model, CPU budget, degradation tests |
| 4 | Wang et al., [DialBench / RPM-10K](https://arxiv.org/abs/2511.21982), 2025 | 10,730 real images, 18 MLLMs benchmarked | VLM comparison | small specialist model instead of an MLLM |
| 5 | Lin et al., [MeasureBench](https://arxiv.org/abs/2510.26865), CVPR 2026 ([data](https://huggingface.co/datasets/FlagEval/MeasureBench), CC BY-SA 4.0) | 1,272 real + 1,170 synthetic instrument images, 26 types; best VLM ~30% on real images; 2D and Blender generators | **our secondary eval set** (pressure and temperature dials, interval hit rate) | we only use the dial subset, and score % FS where a range is labelled |
| 6 | Shen et al., [InSituMeasure](https://arxiv.org/abs/2609.04014), arXiv 2026-09 (AAAI 2027 not confirmed on arXiv v1) | 2,922 real industrial scenes, 8 instrument categories incl. pressure and oil temperature gauges; noise tags; unanswerable tasks; best MLLM 25.7% value+unit | supports the reject option: abstention and overconfidence are scored. No public data found as of 2026-11-10 | specialist model; NPP-specific degradation; RETAKE plays the role of their abstention |
| 7 | Fu et al., [Lost in the Vibrations: VLMs Fail the Dynamic Gauges Test](https://arxiv.org/abs/2604.22829), arXiv 2026-04, accepted to IEEE Instrum. Meas. Mag. ([code](https://github.com/aMa2210/VLM_gauge); [data](https://doi.org/10.5281/zenodo.20624895), CC BY 4.0) | gauge videos (dial indicator, vernier depth gauges) at 6 needle speeds; GPT-5 and Gemini 3 fail on moving needles | evidence that motion matters; possible real motion-blur check (see DATASETS.md) | single frames from a specialist model; we test blur, not tracking |
| 8 | Salomon et al., [Image-based Automatic Dial Meter Reading](https://arxiv.org/abs/2005.03106), IJCNN 2020 | UFPR-ADMR dial meters | error analysis style | process gauges, not utility meters |
| 9 | Cho et al. (KAERI), [Preliminary Design of a Dataset Construction System for AI-based Analog Gauge Reading in Nuclear Facility](https://www.kns.org/files/pre_paper/54/25A-420-%EC%A1%B0%EC%9A%A9%ED%98%B8.pdf), KNS Autumn 2025 | motor-driven real gauges for auto labels, quadruped robot patrol | same problem solved with hardware; we use synthesis | software-only data, no rig |
| 10 | Cho et al. (KAERI), [Evaluation System for Recognition Accuracy of Gauge Reading in Nuclear Facility](https://www.kns.org/files/pre_paper/55/26S-578-%EC%A1%B0%EC%9A%A9%ED%98%B8.pdf), KNS Spring 2026 | built rig from #9: real dial plate, DC-motor pointer, viewpoint −60°..+60° in 10° steps, labels by linear interpolation; reflector for uniform light. Accuracy evaluation is listed as future work | the label formula matches our `read_value`; their viewpoint sweep is a good protocol for our synthetic pose tests | we vary lighting and add NPP degradation, which they list as future work; no public data release mentioned |
| 11 | Cho et al. (KAERI), [Automated Analog Gauge Reading System for Mobile Robots](https://doi.org/10.23919/iccas66577.2025.11301236), ICCAS 2025 | ellipse + perspective + DenseNet169; <1° needle error | Korean NPP state of the art | synthetic training, RETAKE, CPU budget |
| 12 | Cho et al. (KAERI), "Preliminary Design of Chain of Thought with Multimodal Large Language Model for Analog Gauge Reading in Robotic Surveillance", UR 2025 — **unverified: no online copy found**; title from reference lists of #9 and #10 | MLLM + CoT | VLM comparison | specialist model instead of an MLLM |
| 13 | Al Rashdan et al. (INL), [CVML in Safety-Related NPP Applications](https://www.osti.gov/biblio/2279025), INL/RPT-23-74920 | gauge reading vs. digital I&C safety requirements | motivation; confidence-based retake | we implement the retake idea and measure it |
| 14 | Kim et al., [Digitalization of Analog Gauge using Image Processing](https://koreascience.or.kr/article/JAKO202322843272481.page), 2023 | classic CV for ship engine-room robots | classic baseline | learned keypoints |

## How this project differs

Most prior work reports accuracy on clean or lightly varied images. The MLLM benchmarks (#4–#7)
show that general models still miss the needle position, and #6 shows that they answer
when they should abstain. KAERI (#9–#12) addresses NPP gauges with a hardware rig and real panels,
and leaves lighting and harsher conditions for future work. This project trains on synthetic data
only and tests on public real sets. It measures robustness under conditions approximated for NPPs
(low light, radiation speckle, motion blur), and returns RETAKE instead of a number when the reading
is unreliable, so the retake rate and the accuracy of the readings it accepts are reported together.
It targets a CPU budget of 0.5 s per image so it can run on a patrol robot without a GPU.
Limitation: no real NPP gauge photos are in the evaluation.
