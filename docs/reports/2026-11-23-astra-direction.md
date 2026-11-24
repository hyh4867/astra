# ASTRA 시간 순서와 사라진 건물 탐지

- **판정 개선.** validation으로 선택한 A+T의 사라진 건물 recall은 A의 5.35배, F1 차이는 +0.0519였다. 사전 기준은 각각 ≥2배, ≥−0.010이었다.
- **핵심 수치.** FOTBCD에서 A → A+T의 3 seed 평균 F1은 0.5397 → 0.5916, 신축 recall은 0.5152 → 0.6190, 사라진 건물 recall은 0.0684 → 0.3655였다.
- **남은 문제.** 외부 시험은 같은 프랑스 데이터셋 1개다. 사라진 건물의 픽셀 누락률은 선택 모델에서도 63.4%였으며, 새 지역과 센서에서의 성능은 확인할 수 없다.

## 설계와 실행

기준 브랜치의 `595c3bb`에서 새 브랜치를 만들고 사용자 요청의 후보와 독립 예측을 `2c9d34b`에 기록해 학습 전 push했다. “선택된 구성 대 A”는 A를 기준으로 두고 A+T, S, S+T 중 세 seed 평균 validation F1이 가장 높은 도전자를 택하는 것으로 해석했다. 원래 adapted seed 42는 동결 SHA-256이 일치한 기존 가중치를 재사용했다. 신규 학습은 A 2회, S 3회이고 T 계열은 같은 seed의 가중치를 공유한다. 다섯 실행 모두 기존 해시순 1,024쌍, 3 epoch, 배치 2, BCE와 AdamW, lr·weight decay 각 1e-4를 유지했다. 같은 seed끼리 샘플 순서와 좌우 반전 결정을 맞추고 S의 시간 교환만 독립 난수로 뽑았다. 교환해도 이진 정답은 같다. [사전등록](../PREREGISTRATION_DIRECTION.md) · [학습 코드](../../scripts/direction_train.py) · [epoch별 손실·교환 수·validation](../../outputs/direction/epochs.csv)

이전 연구에서 공개 원본의 validation F1은 순방향 0.2639, 역순 0.0384였다. 아래의 A는 이미 프랑스 자료로 미세조정한 가중치다. [이전 순방향](../../outputs/generalization/baseline_validation.json) · [이전 역순 쌍별 계수](../../outputs/generalization/reverse_validation.csv)

각 seed의 epoch는 순방향 validation F1으로 골랐다. 그 뒤 A+T와 S+T에 대해 양방향 확률의 평균·최댓값 및 세 임계값을 validation 984쌍에서 모두 비교했다. 다음 표의 순방향·역순 값은 42/1042/2042 순이다. T의 역순 점수는 동일한 순·역 확률의 교환 대칭성으로 정해지며 단위 시험으로 결합 결과 일치를 검사했다.

| 가족 | 결합 | 임계값 | 순방향 validation F1 | 역순 validation F1 | 순방향 평균 | 역순 평균 |
|---|---|---|---|---|---|---|
| A | forward | 0.5 | 0.5579/0.5959/0.5678 | 0.2962/0.3106/0.3117 | 0.5739 | 0.3061 |
| A+T | max | 0.3 | 0.6337/0.6261/0.6465 | 0.6337/0.6261/0.6465 | 0.6355 | 0.6355 |
| S | forward | 0.5 | 0.5574/0.5935/0.5510 | 0.3965/0.4253/0.3911 | 0.5673 | 0.4043 |
| S+T | max | 0.4 | 0.6168/0.6289/0.6139 | 0.6168/0.6289/0.6139 | 0.6199 | 0.6199 |

[전체 후보 비교](../../outputs/direction/validation_candidates.csv) · [seed별 validation 계수](../../outputs/direction/validation.csv) · [동결한 가중치 해시와 설정](../../outputs/direction/frozen.json)

## 동결한 후보의 공식 시험

아래 열은 각 seed에서 픽셀 TP/FP/FN을 먼저 합산한 뒤 낸 점수의 산술평균이다. 방향별 recall은 같은 공식 979쌍에서 전체판 NEW 및 DEMOLISHED 폴리곤을 따로 래스터화하고 이진 예측이 덮은 픽셀을 각 방향의 GT 픽셀로 나눈 값이다. 한 픽셀이 두 클래스에 속하면 각 분모에 포함한다. LEVIR는 128쌍 전체의 순방향 또는 동결된 대칭 추론에 해당하며 기존 프랑스 adapted seed 42 기준값은 0.8957이다.

| 가족 | 프랑스 F1 | P | R | 새로 생긴 건물 R | 사라진 건물 R | LEVIR F1 |
|---|---|---|---|---|---|---|
| A | 0.5397 | 0.7725 | 0.4165 | 0.5152 | 0.0684 | 0.8971 |
| A+T | 0.5916 | 0.6295 | 0.5635 | 0.6190 | 0.3655 | 0.8618 |
| S | 0.5435 | 0.7554 | 0.4259 | 0.5108 | 0.1240 | 0.8975 |
| S+T | 0.5869 | 0.6646 | 0.5283 | 0.5891 | 0.3071 | 0.8635 |

- 학습 때 순서를 바꾼 S와 A의 프랑스 F1 차이는 +0.0038, 신축 recall 차이는 -0.0043, 사라진 건물 recall 차이는 +0.0557였다.
- 선택한 A+T의 precision은 A 대비 -0.1430, LEVIR 평균 F1은 -0.0353 달라졌다. [seed별 출처](../../outputs/direction/scores.csv) · [가족 평균과 범위](../../outputs/direction/family_summary.csv)
- T는 한 쌍당 순·역 두 방향을 계산하므로 단방향 모델의 두 배에 해당하는 forward pass가 필요하다. 이번 시험에서는 가중치가 같은 단방향·대칭 후보끼리 확률을 공유했다.

### 개별 seed와 범위

| 가족 | seed | 프랑스 F1 | P | R | 신축 R | 사라진 건물 R | LEVIR F1 |
|---|---|---|---|---|---|---|---|
| A | 42 | 0.5320 | 0.7904 | 0.4009 | 0.4952 | 0.0671 | 0.8957 |
| A | 1042 | 0.5585 | 0.7232 | 0.4549 | 0.5634 | 0.0734 | 0.8977 |
| A | 2042 | 0.5286 | 0.8039 | 0.3938 | 0.4868 | 0.0646 | 0.8979 |
| A+T | 42 | 0.5927 | 0.6523 | 0.5431 | 0.6005 | 0.3358 | 0.8688 |
| A+T | 1042 | 0.5872 | 0.5617 | 0.6152 | 0.6633 | 0.4440 | 0.8526 |
| A+T | 2042 | 0.5950 | 0.6745 | 0.5323 | 0.5931 | 0.3168 | 0.8639 |
| S | 42 | 0.5428 | 0.7672 | 0.4199 | 0.5028 | 0.1240 | 0.8976 |
| S | 1042 | 0.5583 | 0.7129 | 0.4588 | 0.5487 | 0.1398 | 0.8983 |
| S | 2042 | 0.5294 | 0.7860 | 0.3991 | 0.4810 | 0.1084 | 0.8966 |
| S+T | 42 | 0.5879 | 0.6815 | 0.5170 | 0.5808 | 0.2829 | 0.8647 |
| S+T | 1042 | 0.5939 | 0.6188 | 0.5708 | 0.6228 | 0.3800 | 0.8610 |
| S+T | 2042 | 0.5790 | 0.6934 | 0.4969 | 0.5638 | 0.2583 | 0.8646 |

| 가족 | 프랑스 F1 범위 | 신축 R 범위 | 사라진 건물 R 범위 | LEVIR F1 범위 |
|---|---|---|---|---|
| A | 0.5286–0.5585 (폭 0.0299) | 0.4868–0.5634 (폭 0.0766) | 0.0646–0.0734 (폭 0.0088) | 0.8957–0.8979 (폭 0.0022) |
| A+T | 0.5872–0.5950 (폭 0.0078) | 0.5931–0.6633 (폭 0.0702) | 0.3168–0.4440 (폭 0.1273) | 0.8526–0.8688 (폭 0.0162) |
| S | 0.5294–0.5583 (폭 0.0290) | 0.4810–0.5487 (폭 0.0677) | 0.1084–0.1398 (폭 0.0314) | 0.8966–0.8983 (폭 0.0017) |
| S+T | 0.5790–0.5939 (폭 0.0149) | 0.5638–0.6228 (폭 0.0590) | 0.2583–0.3800 (폭 0.1217) | 0.8610–0.8647 (폭 0.0037) |

세 seed는 한 학습 표본과 시험 분할을 공유한다. 위 범위는 관측한 세 실행의 최솟값과 최댓값이다. P와 R의 범위도 [가족별 통계](../../outputs/direction/family_summary.csv)에 기록했다.

### seed 42의 쌍 단위 bootstrap

각 행은 같은 seed의 비교 모델 점수에서 A 점수를 뺀 차이다. 각 복원추출에서 979쌍의 픽셀 계수를 합한 뒤 지표를 다시 계산했다. 동일한 표본 인덱스를 양쪽 모델에 적용했다(B=1,000, RNG seed 42). 구간은 2.5·97.5 백분위수다. 인접 패치 간 공간 상관과 학습 seed의 변이는 이 구간에 포함되지 않는다.

| 비교 | 지표 | 관측 차이 | 95% 구간 |
|---|---|---|---|
| A+T minus A | f1 | +0.0607 | +0.0337 ~ +0.0901 |
| A+T minus A | precision | -0.1382 | -0.1640 ~ -0.1131 |
| A+T minus A | recall | +0.1421 | +0.1179 ~ +0.1695 |
| A+T minus A | new_recall | +0.1053 | +0.0915 ~ +0.1204 |
| A+T minus A | disappeared_recall | +0.2687 | +0.1822 ~ +0.3638 |
| S minus A | f1 | +0.0108 | +0.0016 ~ +0.0203 |
| S minus A | precision | -0.0232 | -0.0326 ~ -0.0141 |
| S minus A | recall | +0.0190 | +0.0104 ~ +0.0275 |
| S minus A | new_recall | +0.0076 | -0.0014 ~ +0.0170 |
| S minus A | disappeared_recall | +0.0569 | +0.0367 ~ +0.0809 |
| S+T minus A | f1 | +0.0559 | +0.0328 ~ +0.0812 |
| S+T minus A | precision | -0.1090 | -0.1281 ~ -0.0895 |
| S+T minus A | recall | +0.1161 | +0.0937 ~ +0.1411 |
| S+T minus A | new_recall | +0.0856 | +0.0716 ~ +0.1011 |
| S+T minus A | disappeared_recall | +0.2158 | +0.1392 ~ +0.3013 |

[bootstrap 결과](../../outputs/direction/bootstrap_seed42.csv) · [프랑스 쌍별 계수](../../outputs/direction/fotbcd_counts.csv) · [LEVIR 쌍별 계수](../../outputs/direction/levir_counts.csv)

## 예측 대 실제

| 질문 | 작업 에이전트 사전 예측 | Claude 독립 예측 | 관측값 | 작업 / Claude 범위 |
|---|---|---|---|---|
| S 사라진 건물 R | 0.36 (0.23–0.48) | 0.30 (0.15–0.45) | 0.1240 | 범위 밖 / 범위 밖 |
| S F1 | 0.56 (0.52–0.60) | 0.57 (0.53–0.61) | 0.5435 | 범위 안 / 범위 안 |
| S 신축 R | 0.46 (0.39–0.52) | 0.47 (0.40–0.52) | 0.5108 | 범위 안 / 범위 안 |
| A의 seed별 F1 폭 | 0.025 (0.008–0.055) | 0.02 (0.005–0.05) | 0.0299 | 범위 안 / 범위 안 |

## 검산·편차·한계

최종 평가 코드는 `780ad81067b8a76af17e7f5c181eabb0041b87d1`, 동결 설정 파일 SHA-256은 `8af5058f073d668a173b02df7e6580988067800a1fb32209009267683b276d44`, 시작 상태는 `git_dirty=false`였다. 공식 두 시험의 24개 F1을 표준 라이브러리만 쓰는 별도 스크립트가 쌍별 TP/FP/FN에서 재합산했다. 최대 절대 차이는 0였다. 이전 연구 A42와 동일 입력을 대조한 결과는 아래와 같다. [검산 코드](../../scripts/verify_direction.py) · [24개 독립 계산](../../outputs/direction/independent_f1.json) · [실행 출처](../../outputs/direction/run.json)

| 데이터 | 이전 A42 F1 | 새 A42 F1 | 계수가 다른 쌍 / 전체 |
|---|---|---|---|
| fotbcd | 0.5320 | 0.5320 | 0 / 979 |
| levir | 0.8957 | 0.8957 | 0 / 128 |

프랑스 A42 비교에는 방향별 적중 수와 분모도 포함했다. [비교 원시 계수 차이](../../outputs/direction/baseline_reproduction.json)

실행한 후보·epoch·선택·주 판정과 쌍 단위 구간은 사전등록을 따랐다. 기존 test의 취약점을 보고 기획한 후속 연구라서, 이 데이터셋의 test가 연구 질문 자체와 독립적이지 않다. 시점을 교환해도 이진 변화 정답은 유지되지만 모델이 신축·사라짐 방향을 직접 분류하도록 학습한 것은 아니다. 세 seed에는 데이터 선정·라벨 오차의 불확실성이 포함되지 않는다. T 계열은 양방향 결합과 임계값을 함께 선택했다. 순방향 모델은 임계값 0.5에 고정했으므로, A+T와 A의 차이만으로 양방향 결합과 임계값 각각의 기여를 분리할 수 없다.

학습과 평가를 screen과 caffeinate -i에서 실행했다. 긴 단계 전 기록한 25회 점검에서 사용 스왑은 1.61–2.38 GiB, 시스템 여유 메모리는 34–75%였다. 공유 잠금은 사용하지 않았고 학습은 한 번에 하나만 진행했다. 각 epoch가 끝나면 재개 가능한 로컬 체크포인트를 원자적으로 갱신했다. 보고서 작성 시각은 2026-11-22T21:26+09:00다. [메모리 점검 시각과 측정값](../../outputs/direction/resource_checks.csv)

**사이트 문구 제안** — “외부 시험은 프랑스 FOTBCD-Binary 한 데이터셋입니다. 같은 1,024쌍에서 두 학습 방식을 각각 세 seed로 비교했습니다. 검증에서 고른 구성의 외부 F1은 평균 0.592, 새로 생긴 건물 픽셀 재현율은 0.619, 사라진 건물은 0.366였습니다.”

## 전체 pytest 출력

`python -m pytest -ra --tb=short tests` 출력이다. 로컬 경로 접두어만 상대 경로로 바꿨다.

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.8, pytest-8.4.2, pluggy-1.7.0
rootdir: .
plugins: anyio-4.16.0
collected 52 items

tests/test_detector.py ..                                                [  3%]
tests/test_direction.py ....                                             [ 11%]
tests/test_generalization.py .....                                       [ 21%]
tests/test_geospatial.py ....                                            [ 28%]
tests/test_inference.py .....s                                           [ 40%]
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
================== 51 passed, 1 skipped, 2 warnings in 11.82s ==================
```
