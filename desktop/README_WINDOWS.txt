mv2npp gauge reader (Windows desktop app)
=========================================

Run mv2npp-gauge-reader.exe. The build is not code-signed, so Windows SmartScreen may warn on
the first start: choose "More info" -> "Run anyway".

- "예시 사진" (examples) opens the bundled example photos; "사진 열기" / "폴더 열기" or drag and
  drop read your own photos. A photo named TAG__anything.jpg (for example PI-301__round1.jpg)
  uses the registered gauge TAG from configs/gauges.csv; other photos read the range with OCR.
- "보고서 저장 (CSV)" saves the round as a CSV file.

Model: YOLO11s-pose version 5b, trained only on synthetic renders, ONNX Runtime on the CPU.
Not validated on real nuclear power plant photos.
Code (MIT): https://github.com/uhyon/mv2npp-gauge-reading
Example photos: own synthetic renders (two on CC BY 2.0 background photos) and Roboflow
Universe "Analog Gauge Reading" photos (CC BY 4.0); credits in demo/examples/ATTRIBUTION.md.
