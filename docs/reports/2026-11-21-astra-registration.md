# ASTRA 잔여 이동 보정 실험 재개 기록

2026-11-21 KST · 작업 브랜치 `hyh4867/astra-registration` · 결과·코드 커밋 `e6ce179c3027cf9143d24e376477f893a3489992`

128쌍에서 (+5,+5) px 이동 후 자동 보정은 F1을 0.7293에서 0.7917로 회복했다. 이동을 주입하지 않은 입력에서는 0.9117에서 0.8994로 낮아졌다. 앱은 기본으로 원본과 추정 이동량을 보여 준다. 겹쳐 보기에서 어긋남이 보이면 자동 보정을 선택할 수 있다. 이 UI 기본값은 test 관찰 뒤 결정했다. 아래 표의 자동 보정 정책은 test를 열기 전에 validation으로 고정했다.

## 중단과 재개

- 직전 작업 지시는 두 시점의 건물 후보를 비교하는 앱에 잔여 x/y 이동 보정을 연결하고, 정상 입력의 부작용과 이동 입력의 회복을 실제로 평가하는 것이었다. 이어진 사용자 메시지는 “다시 이어서 하세요”였다.
- 이전 세션의 대기 프로세스는 메모리 부족으로 종료됐다. `outputs/registration/logs/val.log`는 64/64, `test32.log`는 32/32에서 완료됐다. `test128.log`에는 사라진 Python 경로 오류만 있었다. 재시동이 추론 도중 어느 쌍을 끊었다는 증거는 없었다.
- 재개 시점에는 screen 세션이 없었다. 미커밋 README와 CSV 864행을 보존했다. 잘린 행이나 중복 키가 없어 지울 평가 행은 없었다. 끊어진 로컬 심볼릭 링크를 교체하고 공개 test.zip과 원본 TinyCD 가중치를 복구했다. SHA-256은 [PROVENANCE.md](../../PROVENANCE.md)의 값과 일치했다.
- `tools/heavy-status.sh`로 점유를 확인하고 설치·추론·테스트·브라우저 시연을 분리된 screen에서 `tools/heavy-run.sh`로 직렬화했다. 평가기는 미완료 이름만 선별하고 한 쌍씩 읽는다. 재개분은 타일 배치 4, PyTorch 2.15.0, torchvision 0.29.2, MPS였다. 기존 32쌍은 이전 환경의 결과이므로 실행 환경이 완전히 같지는 않다.
- `test128-resume.log`에서 추가 96쌍이 끝났다. 보존한 헤더와 864행의 SHA-256은 `0dd13b5cfb259321d178e22c99f1f58523b688a5f64908bed6ade34715bd37a5`다. 실행 로그는 로컬 `outputs/registration/logs/`에 있다.

## 평가 방법과 출처

원본 사전학습 TinyCD를 사용했다. 재학습은 하지 않았다. 정답은 A의 격자에 고정했다. 추정기는 영상만 받아 256 px 타일별 이동을 집계하고, 장면이 평탄하거나 반복되거나 타일이 서로 불일치하면 보류한다. 자동 모드는 신뢰 조건을 통과한 이동 벡터 길이가 4~12 px일 때 B를 반대 방향으로 정수 이동한다. 4 px는 정상 validation F1 손실이 0.002 이하인 가장 작은 후보였다. [validation 선택표](../../outputs/registration/val_deadband_selection.csv)

기존 결과와의 재현 비교는 고정 8 px 테두리를 제외한 [원래 채점표](../../outputs/registration/test_summary.csv)를 사용했다. 기본 테두리 안에도 반사 패딩이 남을 수 있어 아래 표는 쌍·이동별로 **모든 조건에서 실제 영상이 있는 공통 사각형**을 사용했다. 주입과 역보정 두 번의 좌표 변환을 모두 고려했다. 고정 영역이 이미 유효하면 기존 계수를 재사용하고, 두 영역이 다른 경우만 별도로 채점했다. `x0,x1,y0,y1`은 끝 좌표를 제외하는 범위다. 모든 쌍의 TP·FP·FN을 합산한 후 precision, recall, F1을 계산했다. 보류된 쌍은 B를 그대로 탐지한 값으로 포함했다. [쌍별 계수와 채점 영역](../../outputs/registration/test_overlap_per_pair.csv) · [전체 조건별 P/R/F1, 오차, 보류율](../../outputs/registration/test_overlap_summary.csv)

### 전체 test 128쌍

| B에 넣은 이동 (dx, dy), px | 무보정 F1 | 자동 보정 F1 | 실제 이동량 역보정 F1 | 자동 보정 P / R | 보류 / 전체 | 실제 이동 적용 | 추정 오차 중앙값, px |
|---|---:|---:|---:|---:|---:|---:|---:|
| (0, 0) | 0.9117 | 0.8994 | 해당 없음 | 0.9155 / 0.8840 | 63 / 128 | 4 | 2.28 |
| (3, 3) | 0.8118 | 0.8175 | 0.9116 | 0.8326 / 0.8030 | 63 / 128 | 11 | 2.28 |
| (5, 5) | 0.7293 | 0.7917 | 0.9115 | 0.8065 / 0.7774 | 63 / 128 | 62 | 2.24 |
| (-4, 7) | 0.6964 | 0.7786 | 0.9114 | 0.7916 / 0.7660 | 58 / 128 | 70 | 2.23 |

### 고정 seed 42 부분집합 32쌍

| B에 넣은 이동 (dx, dy), px | 무보정 F1 | 자동 보정 F1 | 실제 이동량 역보정 F1 | 자동 보정 P / R | 보류 / 전체 | 실제 이동 적용 | 추정 오차 중앙값, px |
|---|---:|---:|---:|---:|---:|---:|---:|
| (0, 0) | 0.9211 | 0.8956 | 해당 없음 | 0.9033 / 0.8881 | 11 / 32 | 1 | 2.32 |
| (1, 1) | 0.9026 | 0.9025 | 0.9211 | 0.9104 / 0.8947 | 12 / 32 | 1 | 2.32 |
| (2, 2) | 0.8606 | 0.8608 | 0.9211 | 0.8684 / 0.8533 | 12 / 32 | 1 | 2.37 |
| (3, 3) | 0.8128 | 0.8138 | 0.9211 | 0.8215 / 0.8063 | 12 / 32 | 2 | 2.37 |
| (5, 5) | 0.7229 | 0.7857 | 0.9210 | 0.7942 / 0.7774 | 12 / 32 | 20 | 2.33 |
| (-5, -5) | 0.7146 | 0.7828 | 0.9210 | 0.7870 / 0.7787 | 12 / 32 | 20 | 2.28 |
| (6, -4) | 0.7183 | 0.7797 | 0.9210 | 0.7856 / 0.7738 | 12 / 32 | 20 | 2.29 |
| (-4, 7) | 0.6863 | 0.7743 | 0.9210 | 0.7820 / 0.7669 | 11 / 32 | 21 | 2.23 |

이동 주입 0은 촬영 원본을 뜻한다. 두 날짜의 지상 기준점으로 실제 정합 오차를 측정한 데이터는 없다. 표의 오차는 추정 벡터와 **주입량** 사이 거리다. 실제 이동량 역보정은 주입한 값만 되돌린 대조군이다. 깨끗한 입력에서는 무보정 열이 해당 대조군에 해당한다.

### 독립 검산과 재현 명령

[README 실행 명령](../../README.md#registration) · [test 이름 목록](../../data/manifests/test_filenames.txt) · [32쌍 이름 목록](../../data/manifests/robustness_test_seed42.txt)

```text
test_per_pair.csv 2016 unique condition keys; 640 pair-shifts; same area across conditions; summary micro-F1 verified
test_overlap_per_pair.csv 2016 unique condition keys; 640 pair-shifts; same area across conditions; summary micro-F1 verified
Preserved original 864 rows and header byte-for-byte: 0dd13b5cfb259321d178e22c99f1f58523b688a5f64908bed6ade34715bd37a5
Scored 2016 condition rows. Narrowed the common rectangle in 25 pair-shifts.
outputs/robustness_test_seed42.csv (TinyCD, 32 pairs, 8 px crop):
  shift 0: F1 new 0.921145 / published 0.921145 · FP new 133432 / published 133432 · max |ΔP,ΔR,ΔF1| = 0.00e+00
  shift 1: F1 new 0.902552 / published 0.902552 · FP new 168342 / published 168342 · max |ΔP,ΔR,ΔF1| = 0.00e+00
  shift 2: F1 new 0.860602 / published 0.860602 · FP new 247849 / published 247849 · max |ΔP,ΔR,ΔF1| = 0.00e+00
  shift 3: F1 new 0.812827 / published 0.812827 · FP new 336669 / published 336669 · max |ΔP,ΔR,ΔF1| = 0.00e+00
  shift 5: F1 new 0.722945 / published 0.722945 · FP new 506215 / published 506215 · max |ΔP,ΔR,ΔF1| = 0.00e+00
```

## 화면 확인 및 한계

공개 NAIP에서 B에 (+6, −4) px를 주입하자 앱이 (−6, +4) px 보정을 표시했다. (+2, +2) px는 추정 크기 2.69 px로 보정하지 않았고, (+10, −8) px는 검토용 후보로 표시했다. 공개 NAIP 두 시점의 GeoTIFF를 복구하고 macOS의 로컬 Streamlit 일반 진입점을 headless Chrome에서 실행했다. NAIP에는 정답 라벨이 없어 이 화면은 조작과 표시 확인에만 썼다. 화면 캡처 5장과 260프레임 MP4의 디코딩을 확인했다. 실제 iPhone 조작은 시험하지 않았다. 선택형 ROBUST 체크포인트와 준비 마커는 복구할 공개 파일이 없어 해당 모델의 실추론 테스트는 건너뛰었다.

화면 자료는 공개 트리에서 제외했다. 시연 생성 코드: [capture_registration_screenshots.py](../../scripts/capture_registration_screenshots.py).

**이제 사용자가 어떤 판단을 할 수 있는가?** 같은 건물 윤곽이 어긋날 때 원본과 보정본을 겹쳐 보고, 겹침 영역 안에서 지도 갱신 후보를 검토할 수 있다.

**언제 사람이 원본을 확인해야 하는가?** 타일 간 방향이 다르거나 평탄·반복 무늬, 회전·축척 변화, 그림자 또는 지형 시차가 있으면 원본 두 장을 대조해 후보를 판정해야 한다.

공개본에는 조건별 요약과 공통 영역 검산용 쌍별 계수를 실었다. 고정 crop 중간 산출물인 `test_per_pair.csv`와 `val_per_pair.csv`는 [평가 코드](../../scripts/evaluate_registration.py)로 생성할 수 있다.

## 전체 테스트 출력

`projects/astra`에서 실행했다. 실행 명령은 `.venv/bin/python -m pytest -o console_output_style=classic --tb=short tests`였다. 아래 출력에는 로컬 절대경로를 프로젝트 상대경로로 표시했다.

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.8, pytest-8.4.2, pluggy-1.7.0
rootdir: .
plugins: anyio-4.16.0
collected 43 items

tests/test_detector.py ..
tests/test_geospatial.py ....
tests/test_inference.py .....s
tests/test_registration.py ...............................

=============================== warnings summary ===============================
tests/test_inference.py::test_streamlit_app_smoke_on_synthetic_pair[TinyCD]
  ./.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:207: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
    warnings.warn(

tests/test_inference.py::test_streamlit_app_smoke_on_synthetic_pair[TinyCD]
  ./.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:222: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
    warnings.warn(msg)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 42 passed, 1 skipped, 2 warnings in 21.03s ==================
```
