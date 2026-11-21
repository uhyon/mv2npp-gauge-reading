# 3단계 마무리 점검 (2026-11-18)

요구사항(`docs/REQUIREMENTS.md`)별 현재 상태입니다. 모델은 합성 데이터만으로 학습한 Version 3(yolo11s-pose, 키포인트 7개)이고, 2026-11-21부터 기본 모델은 같은 구조로 데이터만 바꾼 Version 5b입니다(아래 v5b 행). 표의 다른 수치는 Version 3 기준이며, 수치의 출처는 `docs/RESULTS.md`의 해당 절입니다. 상태: ✅ 충족, 🟡 일부 충족 또는 기준 미정, ⏳ 4–5단계에서 할 일, ❌ 미충족.

## 판독 파이프라인 (FR)

| ID | 상태 | 근거 |
| --- | --- | --- |
| FR-01 합성 생성기 | ✅ | `configs/synth_2d.yaml` 20,000장, seed 고정 (`tests/test_synth.py`) |
| FR-02 자동 라벨 | ✅ | 박스, 키포인트 7개, 범위, 단위, 값 (`tests/test_synth.py`, `test_arc_labels.py`) |
| FR-03 현장 조건 | ✅ | 조건별 비율을 라벨에 기록 (`tests/test_synth.py`) |
| FR-04 검출 | ✅ | ER-04 참조 |
| FR-05 정면 보정 | ✅ | 합성 val (yaw ±50°, pitch ±40°): 보정 0.56% FS / 98% vs 보정 없음 1.86% / 66%. 정면 보기 이미지 `perspective.frontal_view` |
| FR-06 키포인트 | ✅ | ER-03 참조. 점마다 신뢰도 출력 |
| FR-07 눈금 OCR | ✅ | MeterFL 262장: 범위 제시 90%, 그중 99% 정답, 단위 138/138. MeasureBench는 이중 눈금·타이어 게이지에서 약함 |
| FR-08 값 계산 | ✅ | `tests/test_geometry.py` |
| FR-09 RETAKE | ✅ | 회전 판독(±25°) 세 값의 차이 > 5% (사용자 결정 11-17), 눈금 호 < 90°, 범위 없음 |
| FR-10 경보 | ✅ | `check_alarm` 경계 테스트. 정상 범위는 계기 등록표 `configs/gauges.csv`에서 받음 (11-18) |
| FR-11 출력 | ✅ | `Reading` (값, 단위, 상태, spread, 키포인트) + `gauge_reader.draw.draw_reading` 이미지. 데모와 `read_gauge.py`가 같은 `Reading`을 씀 |
| FR-12 웹 데모 | 🟡 | `demo/app.py` 로컬 동작 확인 (예제 8장). Hugging Face Spaces 공개는 4단계 |
| FR-13 열화 모듈 | ✅ | `gauge_reader.eval.degrade`, 고정 seed (`tests/test_degrade.py`) |

## 평가 (ER)

| ID | 상태 | 결과 |
| --- | --- | --- |
| ER-01 판독 정확도 (MeterFL, 목표 평균 ≤ 2% FS, 2% 이내 ≥ 85%) | 🟡 | **v5b(기본)**: 판독 88%, 평균 1.36% FS, 2% 이내 판독분 86%(ONNX 85%), 전체 76%. v3: 범위 등록 + RETAKE: 판독 90%, 평균 1.79% FS (충족), 2% 이내는 판독분 81%, 전체 73% (미달). OCR 범위 전체 파이프라인: 판독 83%, 1.88%, 판독분 81%. RETAKE 없이 전체: 2.42% / 77% |
| ER-02 MeasureBench | ✅ (보고) | OCR 전체 파이프라인 판독 44%, 2.68% FS, 2% 이내 44%(판독분), 구간 적중 29%(판독분) / 13%(전체). 표본이 작고 범위 라벨 42장 |
| ER-03 키포인트 (v4tii) | ✅ (보고) | 검출 97%, 바늘 위치 평균 3.00% / 2% 이내 70% / 5% 이내 93%. 눈금 끝점 5% 이내 29–42%가 약점 |
| ER-04 검출 (obaw9, mAP@0.5 ≥ 0.90) | ✅ | AP50 0.955 (v4tii 0.970) |
| ER-05 견고성 | ✅ (보고) | v5b: 모든 조건에서 평균 오차 1.2–2.0% FS(MeterFL), 강한 블러에서 RETAKE 45%(v3 39%). v3: 범위 등록 시 읽어낸 사진의 평균 오차가 모든 조건·강도에서 1.6–2.2% FS(MeterFL), 1.1–1.9%(합성). 블러 3단계 RETAKE 39%(MeterFL). OCR 범위는 저조도·블러에서 실패. 허용 기준(Q-05) 후보: 읽어낸 사진 평균 ≤ 2% FS |
| ER-06 RETAKE | ✅ (보고) | MeterFL 재촬영 10%, 받아들인 판독 1.79% FS. 열화본별 비율은 ER-05 표 |
| ER-07 처리 시간 (≤ 0.5 s, ONNX) | 🟡 | ONNX Runtime, 작업 PC i5-12400, MeterFL 장당 중간값: 범위 등록 0.29 s (충족), OCR 포함 전체 0.54 s (약간 초과), 빠른 모드 0.10 s. PyTorch는 0.27 / 0.51 s |
| ER-08, ER-09 | – | 선택 항목, 하지 않음 (실사 학습 데이터 없음, Cambridge 링크 중단) |

## 비기능·데이터 (NFR, DR)

| ID | 상태 | 비고 |
| --- | --- | --- |
| NFR-01 처리 시간 | 🟡 | ER-07. 등록 계기는 충족, OCR 경로는 0.54 s |
| NFR-02 재현성 | 🟡 | seed와 설정 파일은 있음. Weights & Biases 기록은 쓰지 않음 (Kaggle 출력 `results.csv`로 대신). README 명령 정리는 5단계 |
| NFR-03 코드 품질 | ✅ | 단계별 모듈, `pytest` 201개 통과 |
| NFR-04 ONNX | ✅ | `best.onnx`로 데모 동작 확인 (결과는 PyTorch와 거의 같음) |
| NFR-05 README | ⏳ | 5단계 (성능표는 이 문서와 RESULTS에서) |
| NFR-06 한계 명시 | 🟡 | DATASETS.md에 있음. README 한계 절은 5단계 |
| NFR-07 MIT | ✅ | LICENSE |
| DR-01–04, DR-08 | ✅ | DATASETS.md (MeterFL 30장 손 검수: 범위·단위 30/30 일치) |
| DR-05 obaw9 분할 | 🟡 | 학습에 쓰지 않아 1,471장 전체를 평가에 사용. 분할 목록 대신 "전체 사용"으로 기록 |
| DR-06 데모 이미지 | ✅ | `demo/examples/` 합성 4장 + CC BY 4.0 Roboflow v4tii 4장, `ATTRIBUTION.md` |
| DR-07 파생 데이터 | ✅ | 파생 데이터 공개 없음 (보정값 CSV만, CC BY-SA 표기 필요 여부는 5단계 확인) |

## 3단계 판정

- 필수 기준 중 수치 목표가 있는 두 가지: ER-04 충족, ER-01은 평균 오차는 충족하고, 2% 이내 85%는 v5b에서 판독분 기준으로 충족(86%), 전체 기준으로는 미달(76%)입니다.
- ER-01 85% 미달의 주원인은 (1) 90° 이상 돌아간 MeterFL 계기 9장(합성에 없음, RETAKE 대상), (2) MeterFL 라벨 잡음(손 검수에서 0.4–1.0% FS 차이 3/30), (3) 눈금 끝점 키포인트 정확도입니다. Q-09(목표 재설정)에서 "RETAKE 후 판독분 기준 85%"로 바꿀지 정해야 합니다.
- 나머지 필수 항목은 4–5단계(데모, ONNX, README) 작업만 남았습니다. 4단계로 넘어가도 됩니다.
