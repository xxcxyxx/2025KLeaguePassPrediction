import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURE_DIR, OOF_DIR, REPORT_DIR
from src.data_utils import add_match_features, basic_preprocess, load_raw_data
from src.feature_engineering import build_train_episode_dataset
from src.metrics import euclidean_distance


def main():
    # 1. 데이터 다시 로드
    train_df, test_df, sample_sub, match_info_df = load_raw_data()
    train_df, test_df, match_info_df = basic_preprocess(train_df, test_df, match_info_df)

    train_episode_df = build_train_episode_dataset(train_df)
    train_model_df = add_match_features(train_episode_df, match_info_df)

    # 2. 정답 / 예측값
    y_true = train_model_df[["end_x", "end_y"]].values
    ensemble_oof = pd.read_csv(OOF_DIR / "oof_ensemble.csv")[["end_x", "end_y"]].values

    errors = euclidean_distance(y_true, ensemble_oof)

    # 3. 분석용 데이터프레임
    err_df = train_model_df.copy()
    err_df["pred_end_x"] = ensemble_oof[:, 0]
    err_df["pred_end_y"] = ensemble_oof[:, 1]
    err_df["error"] = errors
    err_df.to_csv(REPORT_DIR / "train_error_analysis.csv", index=False)

    # 4. 요약 통계
    summary = pd.DataFrame([{
        "mean_error": errors.mean(),
        "median_error": np.median(errors),
        "std_error": errors.std(),
        "p90_error": np.quantile(errors, 0.9),
        "max_error": errors.max()
    }])
    summary.to_csv(REPORT_DIR / "error_summary.csv", index=False)

    # 5. 큰 오차 샘플 저장
    worst = err_df.sort_values("error", ascending=False).head(100)
    worst.to_csv(REPORT_DIR / "top100_worst_predictions.csv", index=False)

    # 6. 에러 히스토그램
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=40)
    plt.title("Ensemble Error Distribution")
    plt.xlabel("Euclidean Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ensemble_error_hist.png", dpi=150)
    plt.close()

    # 7. 실제값 vs 예측값 (end_x)
    plt.figure(figsize=(6, 6))
    plt.scatter(err_df["end_x"], err_df["pred_end_x"], alpha=0.3)
    min_v = min(err_df["end_x"].min(), err_df["pred_end_x"].min())
    max_v = max(err_df["end_x"].max(), err_df["pred_end_x"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Actual end_x")
    plt.ylabel("Predicted end_x")
    plt.title("Actual vs Predicted: end_x")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_pred_end_x.png", dpi=150)
    plt.close()

    # 8. 실제값 vs 예측값 (end_y)
    plt.figure(figsize=(6, 6))
    plt.scatter(err_df["end_y"], err_df["pred_end_y"], alpha=0.3)
    min_v = min(err_df["end_y"].min(), err_df["pred_end_y"].min())
    max_v = max(err_df["end_y"].max(), err_df["pred_end_y"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Actual end_y")
    plt.ylabel("Predicted end_y")
    plt.title("Actual vs Predicted: end_y")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_pred_end_y.png", dpi=150)
    plt.close()

    print("✅ Error analysis finished")
    print(summary)


if __name__ == "__main__":
    main()