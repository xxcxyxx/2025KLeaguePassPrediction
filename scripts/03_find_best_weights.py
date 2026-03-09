import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import optuna
import pandas as pd

from src.config import OOF_DIR, OPTUNA_N_TRIALS, REPORT_DIR
from src.data_utils import add_match_features, basic_preprocess, load_raw_data
from src.feature_engineering import build_train_episode_dataset
from src.metrics import mean_euclidean_distance
from src.utils import save_json


def normalize_weights(weight_dict):
    total = sum(weight_dict.values())
    return {k: v / total for k, v in weight_dict.items()}


def main():
    # 1. 정답(y_true) 준비
    train_df, test_df, sample_sub, match_info_df = load_raw_data()
    train_df, test_df, match_info_df = basic_preprocess(train_df, test_df, match_info_df)

    train_episode_df = build_train_episode_dataset(train_df)
    train_model_df = add_match_features(train_episode_df, match_info_df)
    y_true = train_model_df[["end_x", "end_y"]].values

    # 2. OOF 예측 불러오기
    cat_oof = pd.read_csv(OOF_DIR / "oof_catboost.csv")[["end_x", "end_y"]].values
    et_oof = pd.read_csv(OOF_DIR / "oof_extratrees.csv")[["end_x", "end_y"]].values
    ridge_oof = pd.read_csv(OOF_DIR / "oof_ridge.csv")[["end_x", "end_y"]].values

    print("cat_oof shape:", cat_oof.shape)
    print("et_oof shape:", et_oof.shape)
    print("ridge_oof shape:", ridge_oof.shape)
    print("y_true shape:", y_true.shape)

    # 3. Optuna objective
    def objective(trial):
        raw_weights = {
            "catboost": trial.suggest_float("w_catboost", 1e-6, 1.0),
            "extratrees": trial.suggest_float("w_extratrees", 1e-6, 1.0),
            "ridge": trial.suggest_float("w_ridge", 1e-6, 1.0),
        }
        weights = normalize_weights(raw_weights)

        ensemble_pred = (
            cat_oof * weights["catboost"]
            + et_oof * weights["extratrees"]
            + ridge_oof * weights["ridge"]
        )

        score = mean_euclidean_distance(y_true, ensemble_pred)
        return score

    # 4. 최적화
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

    # 5. best weights 정리
    best_raw = {
        "catboost": study.best_params["w_catboost"],
        "extratrees": study.best_params["w_extratrees"],
        "ridge": study.best_params["w_ridge"],
    }
    best_weights = normalize_weights(best_raw)

    # 6. 최적 앙상블 OOF 계산
    ensemble_oof = (
        cat_oof * best_weights["catboost"]
        + et_oof * best_weights["extratrees"]
        + ridge_oof * best_weights["ridge"]
    )

    ensemble_score = mean_euclidean_distance(y_true, ensemble_oof)

    # 7. 저장
    pd.DataFrame({
        "game_episode": train_model_df["game_episode"],
        "end_x": ensemble_oof[:, 0],
        "end_y": ensemble_oof[:, 1]
    }).to_csv(OOF_DIR / "oof_ensemble.csv", index=False)

    save_json(best_weights, REPORT_DIR / "best_ensemble_weights.json")

    pd.DataFrame([{
        "model": "ensemble",
        "score": ensemble_score,
        "catboost_weight": best_weights["catboost"],
        "extratrees_weight": best_weights["extratrees"],
        "ridge_weight": best_weights["ridge"],
    }]).to_csv(REPORT_DIR / "ensemble_result.csv", index=False)

    print("\n✅ Best weights found")
    print(best_weights)
    print(f"✅ Ensemble OOF score: {ensemble_score:.6f}")


if __name__ == "__main__":
    main()