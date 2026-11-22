# ASTRA 외부 일반화와 잔여 이동 보정

연구 등록일 2026-11-21. 보고서 생성 시각 2026-11-21T23:56+09:00.

## 핵심 결과

공개 LEVIR TinyCD의 FOTBCD-Binary test F1은 **0.2431**, 프랑스 train 부분집합으로 미세조정한 뒤에는 **0.5320**였다. 원본 LEVIR F1은 **0.9107**, 미세조정 모델은 **0.8957**였다. 두 데이터셋은 촬영 지역·해상도·라벨 정의·표본 구성이 함께 달라 단일 원인을 식별할 수 없다.

외부 시험은 고정 리비전의 공식 979쌍 전체를 사용했다. 두 파일 SHA-256을 확인하고 IDs, EPSG:2154 공간 범위, 연도, 지역 메타데이터를 감사했다. 숫자 ID는 각 분할에서 다시 부여되므로 분할과 실제 공간 범위를 구분했다. [분할 감사](../../data/manifests/fotbcd/audit.json) · [사전등록 및 독립 예측](../2026-11-21-generalization-preregistration.md)

## 예측 대 실제

| 지표 | 작업 에이전트 사전 예측 | 독립 리뷰 Claude 사전 예측 | 실측 |
|---|---|---|---|
| 외부 원본 F1 | 0.40–0.65 | 0.35 (0.25–0.50) | 0.2431 |
| 외부 미세조정 F1 | 원본보다 +0.05 이상 | 0.65 (0.55–0.75) | 0.5320 (차이 +0.2890) |
| LEVIR (+5,+5) 새 후보 중앙 오차·회복률 | ≤2.0 px 및 ≥0.40 | 제출 없음 | 2.2071 px, 0.2544 |

독립 예측은 사용자 요청으로 별도 Claude 리뷰 세션이 작성하고 사용자가 검토 후 채택했다. 승인 후 첫 실험 전에 추가 커밋 `3255e82`로 기록했다. 외부 원본은 작업 예측 범위 밖, 독립 예측 범위 밖이었다. 미세조정 향상 예측은 충족, 독립 미세조정 범위는 미충족이었다. 정렬 예측은 두 조건 중 하나라도 미달하면 실패이며 결과는 미충족이다.

## A. FOTBCD-Binary 공식 test

| 구성 | P | R | F1 | IoU | 객체 완전도 | null | skill | 오경보/km² | ≥5 m² 오경보/km² |
|---|---|---|---|---|---|---|---|---|---|
| original | 0.6940 | 0.1473 | 0.2431 | 0.1383 | 0.4019 | 0.0097 | 0.3961 | 346.5958 | 64.9745 |
| calibrated | 0.7276 | 0.2284 | 0.3476 | 0.2104 | 0.4538 | 0.0186 | 0.4434 | 60.7858 | 23.6714 |
| adapted | 0.7904 | 0.4009 | 0.5320 | 0.3624 | 0.5794 | 0.0340 | 0.5646 | 79.3917 | 32.3412 |

original은 원본 해상도와 cutoff 0.5, 정렬 끔이다. calibrated는 가중치를 유지하고 영상을 약 0.5 m/px로 맞춘 뒤 임계값 0.3을 적용했다. adapted는 원본 가중치에서 시작해 프랑스 train 25,908쌍 중 해시순 1,024쌍으로 3 epoch 학습했다. validation 984쌍에서 3번째 epoch가 선택됐다. 이 모델의 입력은 약 0.5 m/px, 임계값은 0.5이며 세 구성 모두 정렬을 껐다. IoU와 F1은 픽셀 단위 전체 혼동계수에서 구했다. 객체는 8-연결이며 한 GT 성분의 30% 이상을 덮으면 탐지다. 우연 일치는 서로 다른 패치에 예측을 재배치해 계산했다. 표본 패치 면적 합은 10.2656 km²로 전국 대표 발생률을 추정하는 면적은 아니다. [최종 설정과 패키지 버전](../generalization-final-config.json) · [원본 집계](../../outputs/generalization/final/original/metrics.json) · [미세조정 집계](../../outputs/generalization/final/adapted/metrics.json)

### 객체 크기별 완전도

| GT 면적 m² | 객체 수 | original | calibrated | adapted |
|---|---|---|---|---|
| <25 | 137 | 0.2190 | 0.1168 | 0.1168 |
| 25–62.5 | 147 | 0.3537 | 0.3061 | 0.3605 |
| 62.5–125 | 290 | 0.6241 | 0.6000 | 0.7103 |
| 125–250 | 297 | 0.5185 | 0.6330 | 0.8047 |
| 250–625 | 175 | 0.2686 | 0.4286 | 0.5657 |
| 625–1250 | 83 | 0.2410 | 0.4578 | 0.6988 |
| ≥1250 | 105 | 0.1143 | 0.2286 | 0.4190 |

### 방향 라벨 분석

Abdelrrahman Moubane이 배포한 전체판 `92b677c881f0d973d76460204431c711ccbac25b`과 Binary 모두 CC BY 4.0이며 IGN 원천은 Licence Ouverte 2.0이다. 분할, 영역, 좌표계, 연도, 지역으로 JSON 레코드를 일대일 대응시켰다. 시험 영상을 시각적으로 선별하지 않았다. NEW/DEMOLISHED 폴리곤을 원래 격자에서 픽셀 중심 규칙으로 래스터화하고 이미 저장한 이진 예측을 그대로 대조했다. 설명용 두 recall은 방향 분류 정확도가 아니다. [전체판 저자 카드](https://huggingface.co/datasets/abdevd/FOTBCD/blob/92b677c881f0d973d76460204431c711ccbac25b/README.md) · [방향별 계수](../../outputs/generalization/final/direction.json)

| 구성 | 신축 TP / 픽셀 | 신축 recall | 철거 TP / 픽셀 | 철거 recall |
|---|---|---|---|---|
| original | 2154201 / 11556239 | 0.1864 | 33631 / 3573225 | 0.0094 |
| calibrated | 3268669 / 11556239 | 0.2828 | 118627 / 3573225 | 0.0332 |
| adapted | 5722640 / 11556239 | 0.4952 | 239776 / 3573225 | 0.0671 |

폴리곤 합집합과 공식 이진 GT의 픽셀 IoU는 0.9881다. 두 클래스가 겹치는 픽셀은 각 클래스 recall의 자기 분모에 포함된다. 이진 GT로 구한 헤드라인 F1은 그대로 유지했다.

원본이 놓친 신축 픽셀은 9,402,038개, 철거 픽셀은 3,539,594개였다. 방향별 라벨은 어느 쪽에서 누락이 컸는지 보여 준다. 학습 라벨 정의만의 인과 효과는 이 비교에서 분리할 수 없다.

### 개발 진단과 LEVIR 회귀

validation에서 같은 원본 체크포인트·cutoff 0.5·정렬 끔으로 입력만 0.5 m에 맞췄을 때 F1은 0.3640였다(원래 0.2 m 대비 +0.1001). 같은 0.5 m 입력에서 cutoff를 0.3으로 내리면 0.3923이었고, 여기에 선택된 정렬을 적용하면 0.3921이었다. 원본의 방향별 validation recall은 신축 0.1988, 철거 0.0222였다. 관측각이 제공되지 않아 각도 단독 효과는 추정할 수 없었다.

| validation 축 정렬 bbox 종횡비 | 면적 | GT 수 | 30% 완전도 |
|---|---|---|---|
| compact | at_least_25m2 | 865 | 0.4393 |
| compact | under_25m2 | 42 | 0.1667 |
| elongated | at_least_25m2 | 200 | 0.2600 |
| elongated | under_25m2 | 113 | 0.2212 |

validation의 원본 F1은 0.2639, 같은 쌍의 시간 순서를 바꾼 F1은 0.0384였다. 시간 순서를 바꾸면 신축·철거가 함께 포함된 이진 GT는 그대로다. 원본의 FN 픽셀 수 상위 validation ID 658과 728을 전후 영상·GT·예측으로 대조했다. 두 사례 모두 후시점에 화면 대부분을 덮는 규칙적인 지붕 모양의 구조가 나타났고, GT가 넓었지만 예측은 각각 0픽셀과 111픽셀뿐이었다. 따라서 철거 라벨만으로 모든 누락을 설명할 수 없다. FP가 많은 ID 850은 후시점의 넓은 직사각형 구조에 반응했지만 GT는 두 작은 영역으로 제한됐다. 이 관찰만으로 라벨 오류 여부를 판정할 수 없다. [학습 없는 해상도·임계값·정렬 대조](../../outputs/generalization/calibration_summary.csv) · [건물 면적·종횡비별 탐지율](../../outputs/generalization/dev_shape.csv) · [행정구역·연도별 P/R](../../outputs/generalization/dev_strata.csv) · [오류 사례의 validation ID와 픽셀 수](../../outputs/generalization/dev_errors.csv)

| epoch | 평균 훈련 BCE | validation F1 |
|---|---|---|
| 1 | 0.1048 | 0.5039 |
| 2 | 0.0633 | 0.5426 |
| 3 | 0.0525 | 0.5579 |

최종 시험 전 원본 체크포인트를 LEVIR 128쌍의 전체 영상에서 다시 실행했다. 새 TP/FP/FN은 (6121218, 484873, 716186), F1은 0.910658724였다. 기존 (6121218, 484873, 716186)과의 계수 차이는 (0, 0, 0)이다. 미세조정 모델의 같은 LEVIR F1은 0.895721304로 원본 대비 -0.0149였다. [원본 쌍별 계수](../../outputs/generalization/levir_original.csv) · [미세조정 쌍별 계수](../../outputs/generalization/levir_adapted.csv)

저자 배포 코드의 비교 모델은 DINOv3 ViT-L 기반 HybridChangeDetector로, 256 px 입력과 시험 전체 픽셀 계수를 쓴다. FOTBCD-Binary 시험표의 현지 학습 모델은 P/R/F1/IoU 0.915/0.886/0.900/0.818, LEVIR-CD+ 학습 모델은 0.802/0.324/0.462/0.300이다. 학습 표본 수, 모델, 입력 크기와 학습 도메인이 달라 직접 순위만으로 원인을 판정할 수 없다. [저자 비교표](https://github.com/abdelpy/FOTBCD-datasets/tree/26daefbaec07b5bdd49ee65a84f4113c1913b108) · [저자 모델 설정](https://github.com/abdelpy/FOTBCD-datasets/blob/26daefbaec07b5bdd49ee65a84f4113c1913b108/config.py) · [픽셀 집계](https://github.com/abdelpy/FOTBCD-datasets/blob/26daefbaec07b5bdd49ee65a84f4113c1913b108/metrics.py)

## B. 실제 이동 가능성과 LEVIR 합성 이동

자연 이동은 해시순 validation 24쌍을 고정하고 변화 라벨·모델 출력 없이 SIFT 상호 대응을 검사했다. 영상당 공간 분산, inlier 비율, 양분한 위치별 추정 차이, 별도 대응점 잔차를 검사했다. 24쌍 중 22쌍은 공간적으로 중복을 제거한 상호 대응점이 12개 미만이었다. 나머지 두 쌍도 사분면 분산·inlier 비율·잔차 기준을 모두 만족하지 않았다. 측정 실패에서 자연 어긋남이 없었다는 결론을 낼 수 없다. 기하 기준과 ≥2 native px를 통과한 예비 쌍은 **0개**, 실제 이동 정량평가 실시 여부는 **False**이다. [쌍별 판정](../../outputs/generalization/landmarks/survey.json) · [근거 좌표](../../outputs/generalization/landmarks/points.csv) · [자연 이동 판정](../../outputs/generalization/landmarks/result.json)

기존 centroid/정수 이동, 10배 국소 DFT, translation ECC를 비교했다. LEVIR validation 64쌍에서 정상 F1 손실 허용량 0.002로 거른 뒤 선택한 정책은 **centroid_4**, 비교용 새 후보는 **dft_5**다. validation에서 기존 정책은 정상 F1 손실 0.00175, 평균 회복률 0.2561이었다. 하한 4 px의 DFT는 0.2696까지 회복했으나 정상 F1 손실이 0.00284로 허용량을 넘었다. 하한 5 px의 DFT는 정상 F1 손실이 0이었고 평균 회복률은 0.2019였다. 다음 값은 128쌍 모두를 포함하며 보간 지지영역 교집합에서 채점했다. 추정 벡터 자체의 주입 대비 오차와 clean 추정량을 뺀 상대 오차를 구분했다. 회복률은 (보정−무보정)/(주입량 역보정−무보정) F1이다. [선택표](../../outputs/generalization/registration_choice.json) · [공통 영역별 결과](../../outputs/generalization/reg_test_summary.csv)

| 이동 | 정책 | F1 | 회복률 | 중앙 오차 px | 상대 오차 px | 보류 수 |
|---|---|---|---|---|---|---|
| 0,0 | raw | 0.9116 | — | — | — | 0 |
| 0,0 | oracle | 0.9116 | — | — | — | 0 |
| 0,0 | centroid_4 | 0.8994 | — | 2.2753 | 0.0000 | 63 |
| 0,0 | dft_5 | 0.9032 | — | 2.1897 | 0.0000 | 63 |
| 3,3 | raw | 0.8118 | 0.0000 | — | — | 0 |
| 3,3 | oracle | 0.9116 | 1.0000 | — | — | 0 |
| 3,3 | centroid_4 | 0.8175 | 0.0571 | 2.2803 | 0.0139 | 63 |
| 3,3 | dft_5 | 0.8121 | 0.0028 | 2.1659 | 0.0000 | 65 |
| 5,5 | raw | 0.7293 | 0.0000 | — | — | 0 |
| 5,5 | oracle | 0.9115 | 1.0000 | — | — | 0 |
| 5,5 | centroid_4 | 0.7916 | 0.3420 | 2.2373 | 0.0247 | 63 |
| 5,5 | dft_5 | 0.7757 | 0.2544 | 2.2071 | 0.0500 | 66 |
| -4,7 | raw | 0.6964 | 0.0000 | — | — | 0 |
| -4,7 | oracle | 0.9114 | 1.0000 | — | — | 0 |
| -4,7 | centroid_4 | 0.7786 | 0.3822 | 2.2330 | 0.0531 | 58 |
| -4,7 | dft_5 | 0.7767 | 0.3733 | 2.1581 | 0.1000 | 60 |

(+5,+5)에서 새 후보의 중앙 벡터 오차는 같은 영역의 기존 정책보다 -0.0302 px, 회복률은 -0.0876였다. 이전 연구의 같은 이동에서 기존 정책은 2.2373 px와 회복률 0.3422를 기록했다. 이번에는 8 px 기본 여백 안에서 모든 후보의 보간에 필요한 이웃까지 유효한 사각형을 공통 적용했다. 아래 면적은 이동별 128쌍에서 실제 채점한 픽셀 수다. [이전 연구 결과](../../outputs/registration/test_overlap_summary.csv) · [이번 공통 채점 좌표](../../outputs/generalization/reg_test.csv)

| 주입 이동 px | 이전 공통 픽셀 | 이번 공통 픽셀 | 차이 |
|---|---|---|---|
| 0,0 | 130002794 | 130056192 | 53398 |
| 3,3 | 130037072 | 130056192 | 19120 |
| 5,5 | 130033056 | 130054176 | 21120 |
| -4,7 | 130048128 | 130041072 | -7056 |

## 독립 검산·실행 기록

최종 실행 소스 `8eb35fe859453f270a173b49801f922e44e37a78`, 정규 설정 SHA-256 `7e33a106d22b7fc42b155b7fcdd4bdc054a0e1ec2b62d6b9845d691b20d77f80`, git_dirty=false. 새 test 구성은 validation으로 모두 동결한 뒤 실행했다. 최종 결과는 저장된 예측으로 분석했고 변경된 조건으로 외부 test를 재선택하지 않았다. 별도 표준 라이브러리 스크립트가 각 모델의 979개 쌍별 TP/FP/FN과 집계 일치를 검사했다. [독립 스크립트](../../scripts/verify_generalization.py) · [원본 검산](../../outputs/generalization/final/original/verification.json)

| 구성 | TP | FP | FN | 독립 F1 | 절대 차이 |
|---|---|---|---|---|---|
| original | 2193659 | 967263 | 12695973 | 0.2431 | 0.0 |
| calibrated | 3400231 | 1273082 | 11489401 | 0.3476 | 0.0 |
| adapted | 5969652 | 1582599 | 8919980 | 0.5320 | 0.0 |

재시동 후 이전 정렬 연구의 완료 커밋 `07c88a0`에서 이어갔다. 19:10에 분할 명단(`4e56b99`), 20:12에 LEVIR 재현(`19bc68c`), 22:02에 외부 validation 기준선(`bc999db`)을 각각 보존했다.

첫 미세조정 시도는 첫 배치 RGB 정규화에서 배열 축이 맞지 않아 옵티마이저 갱신 전에 종료됐다. 채널 축을 명시한 커밋 `4d30300`에서 실제 train 입력 텐서 모양을 확인한 뒤 같은 표본·seed·학습 설정으로 실행했다. 23:34에 잠금 대기가 끝나 학습을 재개했다. 이미 끝난 LEVIR 재현과 validation 기준선은 해당 체크포인트에서 읽었다. 23:43에 설정을 동결했고 23:56에 합성 이동 재채점과 전체 테스트까지 마쳤다. 결과 파일과 전체 테스트 출력을 처음 보존한 커밋은 `009225e`다.

실행 순서는 fetch_generalization → freeze_generalization → reproduce_generalization → study_baseline → study_registration → study_landmarks → study_calibrate → study_train → study_lock → study_final → study_diagnostics였다. 각 스크립트는 projects/astra/scripts 아래에 있다. 추론 및 학습은 screen에서 heavy-run 잠금을 사용했다. 배포 파일 SHA-256과 이름 manifest를 보존했다. LEVIR val.zip의 기존 출처 문서에는 해시 문자 하나가 빠져 있어 실제 파일과 배포 LFS 해시 및 기존 공개 MD5가 일치하는 것을 확인하고 바로잡았다. [데이터와 가중치 출처](../../PROVENANCE.md)

**사이트 카드 문구 제안** — “공개 LEVIR-CD 학습 TinyCD 체크포인트의 픽셀 F1은 LEVIR-CD 128쌍에서 0.911, 프랑스 FOTBCD-Binary 979쌍에서 0.243였다. 프랑스 자료로 미세조정한 뒤 같은 외부 시험 F1은 0.532였다.”

## 전체 ASTRA 테스트 출력

아래는 프로젝트에서 실행한 `.venv/bin/python -m pytest -ra --tb=short tests` 출력이다. 로컬 경로 접두어만 상대경로로 바꿨다.

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.8, pytest-8.4.2, pluggy-1.7.0
rootdir: .
plugins: anyio-4.16.0
collected 48 items

tests/test_detector.py ..                                                [  4%]
tests/test_generalization.py .....                                       [ 14%]
tests/test_geospatial.py ....                                            [ 22%]
tests/test_inference.py .....s                                           [ 35%]
tests/test_registration.py ...............................               [100%]

=============================== warnings summary ===============================
tests/test_inference.py::test_streamlit_app_smoke_on_synthetic_pair[TinyCD]
  ./.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:207: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
    warnings.warn(

tests/test_inference.py::test_streamlit_app_smoke_on_synthetic_pair[TinyCD]
  ./.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:222: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
    warnings.warn(msg)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
SKIPPED [1] tests/test_inference.py:64: Optional ROBUST checkpoint and readiness marker required
================== 47 passed, 1 skipped, 2 warnings in 8.45s ===================
```
