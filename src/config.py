from pathlib import Path

BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data" / "raw"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
MATCH_INFO_PATH = DATA_DIR / "match_info.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
OOF_DIR = OUTPUT_DIR / "oof"
REPORT_DIR = OUTPUT_DIR / "reports"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

for d in [FIGURE_DIR, MODEL_DIR, OOF_DIR, REPORT_DIR, SUBMISSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["end_x", "end_y"]

SEED = 42
N_SPLITS = 3
OPTUNA_N_TRIALS = 80

CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 400,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "subsample": 0.8,
    "random_seed": SEED,
    "verbose": 100,
    "allow_writing_files": False
}

EXTRATREES_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": SEED,
    "n_jobs": -1
}

RIDGE_ALPHA = 3.0