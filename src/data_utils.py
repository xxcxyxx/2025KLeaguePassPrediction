import pandas as pd

from src.config import MATCH_INFO_PATH, SAMPLE_SUB_PATH, TEST_PATH, TRAIN_PATH


def load_raw_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    match_info_df = pd.read_csv(MATCH_INFO_PATH)

    return train_df, test_df, sample_sub, match_info_df


def basic_preprocess(train_df, test_df, match_info_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    match_info_df = match_info_df.copy()

    train_df["is_home"] = train_df["is_home"].astype(int)
    train_df["type_name"] = train_df["type_name"].astype(str).fillna("MISSING")
    train_df["result_name"] = train_df["result_name"].astype(str).fillna("MISSING")
    train_df["game_episode"] = train_df["game_episode"].astype(str)

    test_df["game_episode"] = test_df["game_episode"].astype(str)
    test_df["path"] = test_df["path"].astype(str).fillna("")

    match_info_df["game_date"] = pd.to_datetime(match_info_df["game_date"], errors="coerce")

    return train_df, test_df, match_info_df


def add_match_features(df: pd.DataFrame, match_info_df: pd.DataFrame) -> pd.DataFrame:
    match_use = match_info_df.copy()

    match_use["game_year"] = match_use["game_date"].dt.year
    match_use["game_month"] = match_use["game_date"].dt.month
    match_use["game_dayofweek"] = match_use["game_date"].dt.dayofweek

    keep_cols = [
        "game_id",
        "season_id",
        "competition_id",
        "game_day",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "competition_name",
        "country_name",
        "season_name",
        "home_team_name",
        "away_team_name",
        "game_year",
        "game_month",
        "game_dayofweek",
    ]

    return df.merge(match_use[keep_cols], on="game_id", how="left")