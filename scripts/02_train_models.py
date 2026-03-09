import time
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CATBOOST_PARAMS,
    EXTRATREES_PARAMS,
    MODEL_DIR,
    N_SPLITS,
    OOF_DIR,
    REPORT_DIR,
    RIDGE_ALPHA,
    SEED,
)
from src.data_utils import add_match_features, basic_preprocess, load_raw_data
from src.feature_engineering import (
    align_train_test_features,
    build_test_episode_dataset,
    build_train_episode_dataset,
    infer_feature_types,
)
from src.metrics import mean_euclidean_distance
from src.utils import fix_seed, save_pickle


def log_message(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def save_pred_csv(ids, pred, out_path):
    out = pd.DataFrame({
        "game_episode": ids.astype(str).values,
        "end_x": pred[:, 0],
        "end_y": pred[:, 1]
    })
    out.to_csv(out_path, index=False)


def build_ohe_pipeline(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric_cols
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_cols
            ),
        ]
    )


def train_catboost_cv(X, y, X_test, categorical_cols):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros((len(X), 2))
    test_pred = np.zeros((len(X_test), 2))
    scores = []
    fi_list = []

    cat_indices = [X.columns.get_loc(c) for c in categorical_cols if c in X.columns]

    total_start = time.time()
    log_message("CatBoost training started")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        fold_start = time.time()
        log_message(f"[CatBoost] fold={fold} started")

        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx].copy(), y.iloc[va_idx].copy()

        fold_val_pred = np.zeros((len(va_idx), 2))
        fold_test_pred = np.zeros((len(X_test), 2))

        for target_i, target_col in enumerate(["end_x", "end_y"]):
            model = CatBoostRegressor(**CATBOOST_PARAMS)

            train_pool = Pool(X_tr, y_tr[target_col], cat_features=cat_indices)
            valid_pool = Pool(X_va, y_va[target_col], cat_features=cat_indices)
            test_pool = Pool(X_test, cat_features=cat_indices)

            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

            fold_val_pred[:, target_i] = model.predict(valid_pool)
            fold_test_pred[:, target_i] = model.predict(test_pool)

            fi = pd.DataFrame({
                "feature": X.columns,
                "importance": model.get_feature_importance(),
                "target": target_col,
                "fold": fold
            })
            fi_list.append(fi)

            save_pickle(model, MODEL_DIR / f"catboost_{target_col}_fold{fold}.joblib")

        oof[va_idx] = fold_val_pred
        test_pred += fold_test_pred / N_SPLITS

        score = mean_euclidean_distance(y_va.values, fold_val_pred)
        scores.append(score)

        fold_elapsed = time.time() - fold_start
        total_elapsed = time.time() - total_start
        avg_fold = total_elapsed / fold
        remaining = avg_fold * (N_SPLITS - fold)

        log_message(
            f"[CatBoost] fold={fold} done | "
            f"score={score:.6f} | "
            f"fold_time={fold_elapsed:.1f}s | "
            f"elapsed={total_elapsed/60:.1f}m | "
            f"eta={remaining/60:.1f}m"
        )

    fi_df = pd.concat(fi_list, ignore_index=True)
    fi_df.to_csv(REPORT_DIR / "feature_importance_catboost.csv", index=False)

    log_message("CatBoost training finished")
    return oof, test_pred, scores


def train_extratrees_cv(X, y, X_test):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros((len(X), 2))
    test_pred = np.zeros((len(X_test), 2))
    scores = []

    total_start = time.time()
    log_message("ExtraTrees training started")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        fold_start = time.time()
        log_message(f"[ExtraTrees] fold={fold} started")

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y.iloc[tr_idx].values, y.iloc[va_idx].values

        model = ExtraTreesRegressor(**EXTRATREES_PARAMS)
        model.fit(X_tr, y_tr)

        val_pred = model.predict(X_va)
        fold_test_pred = model.predict(X_test)

        oof[va_idx] = val_pred
        test_pred += fold_test_pred / N_SPLITS

        score = mean_euclidean_distance(y_va, val_pred)
        scores.append(score)

        fold_elapsed = time.time() - fold_start
        total_elapsed = time.time() - total_start
        avg_fold = total_elapsed / fold
        remaining = avg_fold * (N_SPLITS - fold)

        log_message(
            f"[ExtraTrees] fold={fold} done | "
            f"score={score:.6f} | "
            f"fold_time={fold_elapsed:.1f}s | "
            f"elapsed={total_elapsed/60:.1f}m | "
            f"eta={remaining/60:.1f}m"
        )

        save_pickle(model, MODEL_DIR / f"extratrees_fold{fold}.joblib")

    log_message("ExtraTrees training finished")
    return oof, test_pred, scores


def train_ridge_cv(X, y, X_test):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros((len(X), 2))
    test_pred = np.zeros((len(X_test), 2))
    scores = []

    total_start = time.time()
    log_message("Ridge training started")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        fold_start = time.time()
        log_message(f"[Ridge] fold={fold} started")

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y.iloc[tr_idx].values, y.iloc[va_idx].values

        model = MultiOutputRegressor(Ridge(alpha=RIDGE_ALPHA))
        model.fit(X_tr, y_tr)

        val_pred = model.predict(X_va)
        fold_test_pred = model.predict(X_test)

        oof[va_idx] = val_pred
        test_pred += fold_test_pred / N_SPLITS

        score = mean_euclidean_distance(y_va, val_pred)
        scores.append(score)

        fold_elapsed = time.time() - fold_start
        total_elapsed = time.time() - total_start
        avg_fold = total_elapsed / fold
        remaining = avg_fold * (N_SPLITS - fold)

        log_message(
            f"[Ridge] fold={fold} done | "
            f"score={score:.6f} | "
            f"fold_time={fold_elapsed:.1f}s | "
            f"elapsed={total_elapsed/60:.1f}m | "
            f"eta={remaining/60:.1f}m"
        )

        save_pickle(model, MODEL_DIR / f"ridge_fold{fold}.joblib")

    log_message("Ridge training finished")
    return oof, test_pred, scores


def main():
    total_start = time.time()
    fix_seed(SEED)

    log_message("Loading data")
    train_df, test_df, sample_sub, match_info_df = load_raw_data()
    train_df, test_df, match_info_df = basic_preprocess(train_df, test_df, match_info_df)

    log_message("Building episode datasets")
    train_episode_df = build_train_episode_dataset(train_df)
    test_episode_df = build_test_episode_dataset(test_df)

    log_message("Merging match info")
    train_model_df = add_match_features(train_episode_df, match_info_df)
    test_model_df = add_match_features(test_episode_df, match_info_df)

    log_message("Aligning train/test features")
    train_features, test_features, y = align_train_test_features(train_model_df, test_model_df)
    numeric_cols, categorical_cols = infer_feature_types(train_features)

    for col in categorical_cols:
        train_features[col] = train_features[col].fillna("MISSING").astype(str)
        test_features[col] = test_features[col].fillna("MISSING").astype(str)

    for col in numeric_cols:
        train_features[col] = pd.to_numeric(train_features[col], errors="coerce")
        test_features[col] = pd.to_numeric(test_features[col], errors="coerce")

    print("train_features shape:", train_features.shape)
    print("test_features shape:", test_features.shape)
    print("target shape:", y.shape)
    print("numeric cols:", len(numeric_cols))
    print("categorical cols:", len(categorical_cols))

    log_message("Starting CatBoost")
    cat_oof, cat_test_pred, cat_scores = train_catboost_cv(
        train_features, y, test_features, categorical_cols
    )

    log_message("Starting OHE preprocessing")
    preprocessor = build_ohe_pipeline(numeric_cols, categorical_cols)
    X_train_enc = preprocessor.fit_transform(train_features)
    X_test_enc = preprocessor.transform(test_features)

    if hasattr(X_train_enc, "toarray"):
        X_train_enc = X_train_enc.toarray()
        X_test_enc = X_test_enc.toarray()

    save_pickle(preprocessor, MODEL_DIR / "ohe_preprocessor.joblib")

    log_message("Starting ExtraTrees")
    et_oof, et_test_pred, et_scores = train_extratrees_cv(X_train_enc, y, X_test_enc)

    log_message("Starting Ridge")
    ridge_oof, ridge_test_pred, ridge_scores = train_ridge_cv(X_train_enc, y, X_test_enc)

    log_message("Saving outputs")
    score_df = pd.DataFrame({
        "model": ["catboost", "extratrees", "ridge"],
        "cv_mean_score": [np.mean(cat_scores), np.mean(et_scores), np.mean(ridge_scores)],
        "cv_std_score": [np.std(cat_scores), np.std(et_scores), np.std(ridge_scores)],
    }).sort_values("cv_mean_score")

    score_df.to_csv(REPORT_DIR / "cv_score_summary.csv", index=False)

    save_pred_csv(train_model_df["game_episode"], cat_oof, OOF_DIR / "oof_catboost.csv")
    save_pred_csv(train_model_df["game_episode"], et_oof, OOF_DIR / "oof_extratrees.csv")
    save_pred_csv(train_model_df["game_episode"], ridge_oof, OOF_DIR / "oof_ridge.csv")

    save_pred_csv(test_model_df["game_episode"], cat_test_pred, OOF_DIR / "test_pred_catboost.csv")
    save_pred_csv(test_model_df["game_episode"], et_test_pred, OOF_DIR / "test_pred_extratrees.csv")
    save_pred_csv(test_model_df["game_episode"], ridge_test_pred, OOF_DIR / "test_pred_ridge.csv")

    total_elapsed = time.time() - total_start
    print("\n✅ Training finished")
    print(score_df)
    log_message(f"Total training pipeline finished | total_time={total_elapsed/60:.1f}m")


if __name__ == "__main__":
    main()