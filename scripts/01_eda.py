import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURE_DIR, REPORT_DIR
from src.data_utils import add_match_features, basic_preprocess, load_raw_data
from src.feature_engineering import build_test_episode_dataset, build_train_episode_dataset


def main():
    # 1) 원본 데이터 로드
    train_df, test_df, sample_sub, match_info_df = load_raw_data()
    train_df, test_df, match_info_df = basic_preprocess(train_df, test_df, match_info_df)

    # 2) 이벤트 단위 -> episode 단위 데이터셋 변환
    train_episode_df = build_train_episode_dataset(train_df)
    test_episode_df = build_test_episode_dataset(test_df)

    # 3) 경기 메타정보 병합
    train_model_df = add_match_features(train_episode_df, match_info_df)
    test_model_df = add_match_features(test_episode_df, match_info_df)

    # 4) 기본 shape 출력
    print("===== RAW DATA SHAPE =====")
    print("train raw:", train_df.shape)
    print("test raw:", test_df.shape)
    print("sample_submission:", sample_sub.shape)
    print("match_info:", match_info_df.shape)

    print("\n===== EPISODE DATA SHAPE =====")
    print("train episode:", train_episode_df.shape)
    print("test episode:", test_episode_df.shape)

    print("\n===== MODEL DATA SHAPE =====")
    print("train model:", train_model_df.shape)
    print("test model:", test_model_df.shape)

    # 5) 컬럼 요약 저장
    summary_df = pd.DataFrame({
        "column": train_model_df.columns,
        "dtype": [str(train_model_df[c].dtype) for c in train_model_df.columns],
        "missing_ratio": [train_model_df[c].isna().mean() for c in train_model_df.columns],
        "nunique": [train_model_df[c].nunique(dropna=False) for c in train_model_df.columns]
    }).sort_values(["missing_ratio", "nunique"], ascending=[False, False])

    summary_df.to_csv(REPORT_DIR / "train_column_summary.csv", index=False)

    # 6) preview 저장
    train_model_df.head(30).to_csv(REPORT_DIR / "train_model_preview.csv", index=False)
    test_model_df.head(30).to_csv(REPORT_DIR / "test_model_preview.csv", index=False)

    # 7) 타깃 분포 저장
    plt.figure(figsize=(7, 5))
    train_model_df["end_x"].hist(bins=40)
    plt.title("Target Distribution: end_x")
    plt.xlabel("end_x")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "target_dist_end_x.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    train_model_df["end_y"].hist(bins=40)
    plt.title("Target Distribution: end_y")
    plt.xlabel("end_y")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "target_dist_end_y.png", dpi=150)
    plt.close()

    # 8) 숫자형 상관관계 저장
    numeric_cols = train_model_df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) > 1:
        corr_df = train_model_df[numeric_cols].corr(numeric_only=True)
        corr_df.to_csv(REPORT_DIR / "train_numeric_correlation.csv")

        if "end_x" in corr_df.columns:
            corr_df["end_x"].sort_values(ascending=False).to_csv(REPORT_DIR / "corr_with_end_x.csv")
        if "end_y" in corr_df.columns:
            corr_df["end_y"].sort_values(ascending=False).to_csv(REPORT_DIR / "corr_with_end_y.csv")

    print("\n✅ EDA finished")
    print(f"Saved reports to: {REPORT_DIR}")
    print(f"Saved figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()