# 2025 K League Pass Prediction Portfolio Project

K리그 경기 이벤트 로그를 활용해 마지막 패스 도착 좌표(`end_x`, `end_y`)를 예측한 프로젝트입니다.  
데이콘 대회를 바탕으로 진행했으며, 단순 제출 코드 재현이 아니라 이벤트 단위 원본 데이터를 `game_episode` 단위 학습 데이터로 재구성하고, 모델 비교, OOF 기반 가중 앙상블, 오차 분석까지 포함한 포트폴리오형 프로젝트로 확장했습니다.

## Competition Reference

- [데이콘 | K리그 경기 패스 좌표 예측 AI 경진대회](https://dacon.io/competitions/official/236647/overview/description)

---

## 1. Project Overview

이 프로젝트의 목표는 경기 내 이벤트 흐름을 바탕으로 **마지막 패스의 도착 좌표**를 예측하는 것입니다.  
원본 데이터는 이벤트 단위로 기록되어 있어 그대로는 예측 모델에 바로 사용하기 어렵기 때문에, `game_episode`를 기준으로 이벤트를 묶고 마지막 이벤트를 타깃으로 두는 방식으로 **학습 가능한 샘플 단위로 재구성**했습니다.

이번 프로젝트에서는 단순히 모델 성능을 확인하는 데 그치지 않고, 다음 두 가지에 집중했습니다.

- 원천 이벤트 로그를 예측 가능한 구조로 변환하는 데이터 재구성
- 단일 모델 비교를 넘어 OOF 기반 앙상블과 오차 분석까지 연결하는 실험 구조 설계

---

## 2. Dataset

프로젝트에 사용한 주요 데이터는 다음과 같습니다.

### `train.csv`
경기 이벤트 로그

주요 컬럼:
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

### `test.csv`
예측 대상 episode 정보

주요 컬럼:
- `game_id`
- `game_episode`
- `path`

### `match_info.csv`
경기 메타 정보  
시즌, 경기일, 대회명, 홈/원정팀, 점수 정보 등을 포함합니다.

### `sample_submission.csv`
제출 형식

### Data Shape
- train raw: **356,721 rows**
- test raw: **2,414 rows**
- reconstructed train episode: **15,428 rows**
- reconstructed test episode: **2,414 rows**

---

## 3. Problem Definition

입력은 경기 중 발생한 이벤트 시퀀스이고, 출력은 마지막 패스 도착 좌표 `(end_x, end_y)`입니다.

이 프로젝트에서는 원본 이벤트 로그를 그대로 모델에 넣지 않고, 각 `game_episode`를 하나의 학습 샘플로 재구성했습니다.

- **입력(feature)**: episode 내 마지막 이벤트 이전까지의 이벤트 흐름
- **타깃(target)**: 마지막 이벤트의 `end_x`, `end_y`

즉, 이벤트 로그를 **episode 단위 supervised learning 문제**로 변환한 것이 핵심입니다.

---

## 4. Data Processing

### 4.1 Event → Episode Reconstruction
`train.csv`를 `game_episode` 기준으로 그룹화한 뒤, 각 episode의 마지막 이벤트를 타깃으로 두고 그 이전 이벤트들로 특징을 생성했습니다.

생성한 주요 특징은 다음과 같습니다.

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
train과 test는 구조가 다르기 때문에 다음 과정을 거쳐 feature space를 정렬했습니다.

- 공통 feature 컬럼 구성
- 수치형 / 범주형 컬럼 분리
- 범주형 결측치 `"MISSING"` 처리
- 수치형 컬럼 numeric coercion 적용
- CatBoost 입력용 데이터와 OHE 기반 모델 입력용 데이터를 분리 구성

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
이를 통해 validation 기준으로 가장 좋은 앙상블 조합을 구성했습니다.

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

단일 모델 중 가장 성능이 좋았던 CatBoost 대비, OOF 기반 가중 앙상블을 통해 validation 기준 성능을 소폭 개선했습니다.

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

평균 오차보다 중앙값이 더 낮게 나타나, 일부 어려운 샘플이 전체 분포의 꼬리를 길게 만드는 형태를 확인했습니다.  
즉 대부분의 일반적인 episode에서는 비교적 안정적인 예측이 가능했지만, 특정 상황에서는 큰 오차가 발생했습니다. 이를 위해 오차 상위 샘플을 별도로 저장하고 예측이 어려운 사례를 분석할 수 있도록 구성했습니다.

---

## 9. Project Structure

```text
2025KLeaguePassPrediction/
├─ data/
│  └─ raw/
│     ├─ train.csv
│     ├─ test.csv
│     ├─ sample_submission.csv
│     └─ match_info.csv
├─ src/
│  ├─ __init__.py
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
├─ outputs/
│  ├─ figures/
│  ├─ models/
│  ├─ oof/
│  ├─ reports/
│  └─ submissions/
├─ .gitignore
├─ requirements.txt
└─ README.md
10. Run
1) EDA
python scripts/01_eda.py
2) Train models
python scripts/02_train_models.py
3) Find best ensemble weights
python scripts/03_find_best_weights.py
4) Make final submission
python scripts/04_make_submission.py
5) Error analysis
python scripts/05_error_analysis.py
11. Outputs

주요 산출물은 다음과 같습니다.

outputs/reports/cv_score_summary.csv

outputs/reports/best_ensemble_weights.json

outputs/reports/ensemble_result.csv

outputs/reports/error_summary.csv

outputs/oof/oof_catboost.csv

outputs/oof/oof_extratrees.csv

outputs/oof/oof_ridge.csv

outputs/oof/oof_ensemble.csv

outputs/submissions/final_submission.csv

12. Key Takeaways

이 프로젝트를 통해 다음 내용을 확인할 수 있었습니다.

이벤트 단위 raw log를 예측 가능한 샘플 단위로 재구성할 수 있었다.

단일 모델 비교를 넘어 OOF 기반 앙상블 구조를 설계할 수 있었다.

단순 제출 파일 생성에 그치지 않고, 오차 분석까지 포함한 해석 가능한 결과 정리가 가능했다.

재현 가능한 프로젝트 구조(src, scripts, outputs)를 기반으로 데이터 처리와 모델링 과정을 분리할 수 있었다.

13. Retrospective

이번 프로젝트에서는 대회 코드 재현 자체보다, 설명 가능한 포트폴리오 프로젝트로 재구성하는 데 집중했습니다.

특히 의미 있었던 부분은 다음과 같습니다.

이벤트 로그를 episode 단위 예측 문제로 변환

CatBoost / ExtraTrees / Ridge 모델 비교

OOF 기반 가중 앙상블 설계

최종 submission 생성

오차 분석까지 포함한 결과 해석

향후에는 다음 방향으로 확장할 수 있습니다.

path 파싱 로직 고도화

sequence 모델 적용

추가 feature engineering

실험 추적 도구 연동
