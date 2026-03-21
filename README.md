# 2025 K League Pass Prediction

K리그 경기 이벤트 로그를 활용해 마지막 패스 도착 좌표(`end_x`, `end_y`)를 예측한 프로젝트입니다.  
데이콘 대회를 바탕으로 진행했으며, 단순 제출 코드 재현이 아니라 이벤트 단위 원본 데이터를 `game_episode` 단위 학습 데이터로 재구성하고, 모델 비교, OOF 기반 가중 앙상블, 오차 분석까지 포함한 포트폴리오형 프로젝트로 확장했습니다.

- Competition: [데이콘 | K리그 경기 패스 좌표 예측 AI 경진대회](https://dacon.io/competitions/official/236647/overview/description)

---

## Summary

- Raw event logs를 `game_episode` 단위 학습 데이터로 재구성
- CatBoost, ExtraTrees, Ridge 모델 비교
- OOF 기반 Optuna 가중 앙상블 설계
- 최종 OOF ensemble score **18.9466**
- 예측 오차 분포 및 worst-case 사례 분석

---

## 1. Project Overview

이 프로젝트의 목표는 경기 내 이벤트 흐름을 바탕으로 **마지막 패스의 도착 좌표**를 예측하는 것입니다.

원본 데이터는 이벤트 단위로 기록되어 있어 그대로는 예측 모델에 바로 사용하기 어렵습니다.  
따라서 `game_episode`를 기준으로 이벤트를 묶고, 마지막 이벤트를 타깃으로 두는 방식으로 **학습 가능한 샘플 단위로 재구성**했습니다.

이번 프로젝트에서는 단순히 모델 성능을 확인하는 데 그치지 않고, 다음 두 가지에 집중했습니다.

- 원천 이벤트 로그를 예측 가능한 구조로 변환하는 데이터 재구성
- 단일 모델 비교를 넘어 OOF 기반 앙상블과 오차 분석까지 연결하는 실험 구조 설계

---

## 2. Dataset

프로젝트에 사용한 주요 데이터는 다음과 같습니다.

### Input Files
- `train.csv`: 경기 이벤트 로그
- `test.csv`: 예측 대상 episode 정보
- `match_info.csv`: 경기 메타 정보
- `sample_submission.csv`: 제출 형식

### Raw Train Columns
- `game_id`
- `period_id`
- `episode_id`
- `time_seconds`
- `team_id`
- `player_id`
- `action_id`
- `type_name`
- `result_name`
- `start_x`
- `start_y`
- `end_x`
- `end_y`
- `is_home`
- `game_episode`

### Data Shape
- train raw: **356,721 rows**
- test raw: **2,414 rows**
- reconstructed train episode: **15,428 rows**
- reconstructed test episode: **2,414 rows**

---

## 3. Problem Definition

입력은 경기 중 발생한 이벤트 시퀀스이고, 출력은 마지막 패스 도착 좌표 `(end_x, end_y)`입니다.

이 프로젝트에서는 원본 이벤트 로그를 그대로 모델에 넣지 않고, 각 `game_episode`를 하나의 학습 샘플로 재구성했습니다.

- **Input**: episode 내 마지막 이벤트 이전까지의 이벤트 흐름
- **Target**: 마지막 이벤트의 `end_x`, `end_y`

즉, 이벤트 로그를 **episode 단위 supervised learning 문제**로 변환한 것이 핵심입니다.

---

## 4. Data Processing

### 4.1 Event → Episode Reconstruction
`train.csv`를 `game_episode` 기준으로 그룹화한 뒤, 각 episode의 마지막 이벤트를 타깃으로 두고 그 이전 이벤트들로 특징을 생성했습니다.

### Main Features
- 이벤트 수
- 고유 선수 수 / 고유 액션 수
- 시간 범위 및 진행 길이
- 시작 좌표 통계량(평균, 표준편차, 최소/최대)
- 마지막 이벤트 직전 위치
- `Pass`, `Carry`, `Shot` 비율
- 성공/실패 이벤트 비율
- 이동 거리 기반 특징
- `path` 문자열 기반 토큰/숫자 특징
- 경기 메타정보(`match_info`) 병합

### 4.2 Train / Test Alignment
train과 test는 구조가 다르기 때문에 다음 과정을 통해 feature space를 정렬했습니다.

- 공통 feature 컬럼 구성
- 수치형 / 범주형 컬럼 분리
- 범주형 결측치 `"MISSING"` 처리
- 수치형 컬럼 numeric coercion 적용
- CatBoost 입력용 데이터와 OHE 기반 모델 입력용 데이터 분리

---

## 5. Modeling

세 가지 모델을 비교했습니다.

### CatBoost
- 범주형 데이터 처리에 강점이 있어 메인 모델로 사용
- `end_x`, `end_y`를 각각 학습

### ExtraTrees
- 비선형 관계를 포착하기 위한 보조 모델

### Ridge
- 선형 베이스라인 및 앙상블 다양성 확보용 모델

모든 모델은 교차검증 기반으로 학습했으며, OOF 예측값과 test 예측값을 함께 저장했습니다.

---

## 6. Ensemble Strategy

단순 평균 대신 **OOF(Out-of-Fold) 예측값**을 기반으로 Optuna를 사용해 최적 가중치를 탐색했습니다.

### Best Weights
- CatBoost: **0.6735**
- ExtraTrees: **0.3257**
- Ridge: **0.0008**

최종 앙상블은 사실상 **CatBoost + ExtraTrees 중심 구조**였습니다.

---

## 7. Results

### Cross Validation Score

| Model | CV Mean Score |
|------|---------------:|
| CatBoost | 18.9628 |
| ExtraTrees | 19.3920 |
| Ridge | 21.2035 |

### OOF Weighted Ensemble Score

| Model | Score |
|------|------:|
| Ensemble | 18.9466 |

CatBoost 단일 모델 대비, OOF 기반 가중 앙상블을 통해 validation 기준 성능을 소폭 개선했습니다.

---

## 8. Error Analysis

최종 앙상블 예측값에 대해 오차 분포와 worst-case 사례를 분석했습니다.

| Metric | Value |
|------|------:|
| Mean Error | 18.9466 |
| Median Error | 16.3242 |
| Std Error | 11.6420 |
| P90 Error | 35.5961 |
| Max Error | 87.7799 |

평균 오차보다 중앙값이 더 낮게 나타나 일부 어려운 샘플이 전체 분포의 꼬리를 길게 만드는 형태를 확인했습니다.  
즉, 대부분의 일반적인 episode에서는 비교적 안정적인 예측이 가능했지만 특정 상황에서는 큰 오차가 발생했습니다.

---


## 9. Project Structure

```text
2025KLeaguePassPrediction/
├─ src/
│  ├─ config.py
│  ├─ utils.py
│  ├─ metrics.py
│  ├─ data_utils.py
│  └─ feature_engineering.py
├─ scripts/
│  ├─ 01_eda.py
│  ├─ 02_train_models.py
│  ├─ 03_find_best_weights.py
│  ├─ 04_make_submission.py
│  └─ 05_error_analysis.py
├─ README.md
└─ requirements.txt
```

---

## 10. Run

```bash
python scripts/01_eda.py
python scripts/02_train_models.py
python scripts/03_find_best_weights.py
python scripts/04_make_submission.py
python scripts/05_error_analysis.py
```

---

## 11. Outputs

주요 산출물은 다음과 같습니다.

- `cv_score_summary.csv`
- `best_ensemble_weights.json`
- `ensemble_result.csv`
- `error_summary.csv`
- `oof_ensemble.csv`
- `final_submission.csv`

---

## 12. Key Takeaways

- 이벤트 단위 raw log를 예측 가능한 episode 단위 데이터로 재구성
- 단일 모델 비교를 넘어 OOF 기반 앙상블 설계
- 최종 submission 생성과 오차 분석까지 포함한 결과 정리
- 재현 가능한 프로젝트 구조로 데이터 처리와 모델링 단계 분리

## 📊 Visualization

본 프로젝트에서는 예측 결과를 다양한 관점에서 분석하기 위해 시각화를 수행했습니다.  
이를 통해 모델이 공간 패턴을 얼마나 잘 반영하는지와 오차 발생 특성을 확인했습니다.

→ 이벤트 로그 기반 좌표 예측 결과를 공간적으로 분석하여 모델의 성능과 한계를 파악했습니다.

<br>

### 1. Actual vs Predicted Final Pass Location / Mean Error Heatmap

<p align="center">
  <img src="./images/01_actual_vs_pred.png" width="48%" />
  <img src="./images/02_error_heatmap.png" width="48%" />
</p>

- **Actual vs Predicted Final Pass Location**  
  실제 패스 도착 좌표와 예측 좌표를 비교한 시각화입니다. 주요 도착 지점 분포와 예측 결과가 유사한 패턴을 보이는지 확인할 수 있습니다.

- **Mean Error Heatmap by Actual End Location**  
  실제 패스 도착 위치 기준 평균 오차를 히트맵으로 나타낸 결과입니다. 특정 구역에서 오차가 상대적으로 크게 발생하는 패턴을 확인할 수 있습니다.

<br>

### 2. Error Distribution / Error Vectors

<p align="center">
  <img src="./images/03_error_distribution.png" width="48%" />
  <img src="./images/04_error_vectors.png" width="48%" />
</p>

- **Euclidean Error Distance Distribution**  
  예측 좌표와 실제 좌표 간 거리 오차 분포입니다. 모델이 어느 수준의 오차 범위에서 가장 많이 분포하는지 확인할 수 있습니다.

- **Error Vectors (Actual → Predicted)**  
  실제 좌표에서 예측 좌표까지의 오차 방향을 벡터로 시각화한 결과입니다. 오차가 특정 방향으로 치우치는지 확인할 수 있습니다.
