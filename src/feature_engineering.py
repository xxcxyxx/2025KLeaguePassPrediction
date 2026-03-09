import re
from collections import Counter

import numpy as np
import pandas as pd


def tokenize_text(text):
    text = "" if pd.isna(text) else str(text).lower()
    tokens = re.split(r"[\s,\|\[\]\(\)\{\}:;>'\"/_-]+", text)
    return [t for t in tokens if t != ""]


def extract_numbers_from_text(text):
    text = "" if pd.isna(text) else str(text)
    nums = re.findall(r"-?\d+\.?\d*", text)
    return [float(n) for n in nums]


def path_text_features(path_text):
    text = "" if pd.isna(path_text) else str(path_text)
    tokens = tokenize_text(text)
    nums = extract_numbers_from_text(text)
    token_counter = Counter(tokens)

    return {
        "path_char_len": len(text),
        "path_token_len": len(tokens),
        "path_unique_token_len": len(set(tokens)),
        "path_num_count": len(nums),
        "path_num_mean": np.mean(nums) if len(nums) > 0 else np.nan,
        "path_num_std": np.std(nums) if len(nums) > 0 else np.nan,
        "path_num_min": np.min(nums) if len(nums) > 0 else np.nan,
        "path_num_max": np.max(nums) if len(nums) > 0 else np.nan,
        "path_contains_pass": int("pass" in token_counter),
        "path_contains_carry": int("carry" in token_counter),
        "path_contains_shot": int("shot" in token_counter),
        "path_contains_successful": int("successful" in token_counter),
        "path_contains_unsuccessful": int("unsuccessful" in token_counter),
        "path_pass_count": token_counter.get("pass", 0),
        "path_carry_count": token_counter.get("carry", 0),
        "path_shot_count": token_counter.get("shot", 0),
        "path_successful_count": token_counter.get("successful", 0),
        "path_unsuccessful_count": token_counter.get("unsuccessful", 0),
    }


def build_train_episode_dataset(train_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sort_cols = ["game_id", "period_id", "episode_id", "time_seconds", "action_id"]

    for game_episode, grp in train_df.sort_values(sort_cols).groupby("game_episode"):
        grp = grp.reset_index(drop=True)

        if len(grp) < 2:
            continue

        prefix = grp.iloc[:-1].copy()
        target_row = grp.iloc[-1].copy()

        path_tokens = []
        for _, r in prefix.iterrows():
            path_tokens.append(
                f"{r['type_name']} {r['result_name']} "
                f"{round(r['start_x'], 1)} {round(r['start_y'], 1)} "
                f"{int(r['player_id'])}"
            )
        path_text = " | ".join(path_tokens)

        prefix["delta_x"] = prefix["end_x"] - prefix["start_x"]
        prefix["delta_y"] = prefix["end_y"] - prefix["start_y"]
        prefix["move_dist"] = np.sqrt(prefix["delta_x"]**2 + prefix["delta_y"]**2)

        type_counts = prefix["type_name"].value_counts(dropna=False).to_dict()
        result_counts = prefix["result_name"].value_counts(dropna=False).to_dict()

        feat = {
            "game_episode": str(game_episode),
            "game_id": int(prefix["game_id"].iloc[0]),
            "period_id_first": int(prefix["period_id"].iloc[0]),
            "period_id_last": int(prefix["period_id"].iloc[-1]),
            "episode_id": int(prefix["episode_id"].iloc[0]),
            "is_home": int(prefix["is_home"].iloc[0]),
            "n_events": len(prefix),
            "n_unique_players": prefix["player_id"].nunique(),
            "n_unique_actions": prefix["action_id"].nunique(),
            "time_min": prefix["time_seconds"].min(),
            "time_max": prefix["time_seconds"].max(),
            "time_span": prefix["time_seconds"].max() - prefix["time_seconds"].min(),
            "start_x_mean": prefix["start_x"].mean(),
            "start_x_std": prefix["start_x"].std(),
            "start_x_min": prefix["start_x"].min(),
            "start_x_max": prefix["start_x"].max(),
            "start_y_mean": prefix["start_y"].mean(),
            "start_y_std": prefix["start_y"].std(),
            "start_y_min": prefix["start_y"].min(),
            "start_y_max": prefix["start_y"].max(),
            "last_start_x": prefix["start_x"].iloc[-1],
            "last_start_y": prefix["start_y"].iloc[-1],
            "last_player_id": int(prefix["player_id"].iloc[-1]),
            "last_team_id": int(prefix["team_id"].iloc[-1]),
            "last_type_name": str(prefix["type_name"].iloc[-1]),
            "last_result_name": str(prefix["result_name"].iloc[-1]),
            "type_pass_count": type_counts.get("Pass", 0),
            "type_carry_count": type_counts.get("Carry", 0),
            "type_shot_count": type_counts.get("Shot", 0),
            "result_successful_count": result_counts.get("Successful", 0),
            "result_unsuccessful_count": result_counts.get("Unsuccessful", 0),
            "move_dist_mean": prefix["move_dist"].mean(),
            "move_dist_std": prefix["move_dist"].std(),
            "delta_x_mean": prefix["delta_x"].mean(),
            "delta_y_mean": prefix["delta_y"].mean(),
            "path_text": path_text,
            "target_type_name": str(target_row["type_name"]),
            "target_result_name": str(target_row["result_name"]),
            "end_x": float(target_row["end_x"]),
            "end_y": float(target_row["end_y"]),
        }

        feat["pass_ratio"] = feat["type_pass_count"] / feat["n_events"] if feat["n_events"] > 0 else 0
        feat["carry_ratio"] = feat["type_carry_count"] / feat["n_events"] if feat["n_events"] > 0 else 0
        feat["success_ratio"] = feat["result_successful_count"] / feat["n_events"] if feat["n_events"] > 0 else 0

        feat.update(path_text_features(path_text))
        rows.append(feat)

    return pd.DataFrame(rows)


def build_test_episode_dataset(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in test_df.iterrows():
        feat = {
            "game_episode": str(r["game_episode"]),
            "game_id": int(r["game_id"]),
            "path_text": str(r["path"]),
        }
        feat.update(path_text_features(r["path"]))
        rows.append(feat)

    return pd.DataFrame(rows)


def align_train_test_features(train_model_df: pd.DataFrame, test_model_df: pd.DataFrame):
    y = train_model_df[["end_x", "end_y"]].copy()

    train_features = train_model_df.drop(columns=["end_x", "end_y"], errors="ignore").copy()
    test_features = test_model_df.copy()

    for c in ["target_type_name", "target_result_name"]:
        if c in train_features.columns:
            train_features = train_features.drop(columns=[c])

    all_cols = sorted(set(train_features.columns).union(set(test_features.columns)))

    for c in all_cols:
        if c not in train_features.columns:
            train_features[c] = np.nan
        if c not in test_features.columns:
            test_features[c] = np.nan

    train_features = train_features[all_cols].copy()
    test_features = test_features[all_cols].copy()

    return train_features, test_features, y


def infer_feature_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols