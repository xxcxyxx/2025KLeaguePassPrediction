import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import OOF_DIR, REPORT_DIR, SUBMISSION_DIR
from src.data_utils import load_raw_data
from src.utils import load_json


def main():
    _, _, sample_sub, _ = load_raw_data()
    weights = load_json(REPORT_DIR / "best_ensemble_weights.json")

    cat_test = pd.read_csv(OOF_DIR / "test_pred_catboost.csv")
    et_test = pd.read_csv(OOF_DIR / "test_pred_extratrees.csv")
    ridge_test = pd.read_csv(OOF_DIR / "test_pred_ridge.csv")

    final = sample_sub.copy()

    final["end_x"] = (
        cat_test["end_x"] * weights["catboost"]
        + et_test["end_x"] * weights["extratrees"]
        + ridge_test["end_x"] * weights["ridge"]
    )

    final["end_y"] = (
        cat_test["end_y"] * weights["catboost"]
        + et_test["end_y"] * weights["extratrees"]
        + ridge_test["end_y"] * weights["ridge"]
    )

    final["end_x"] = final["end_x"].clip(0, 105)
    final["end_y"] = final["end_y"].clip(0, 68)

    out_path = SUBMISSION_DIR / "final_submission.csv"
    final.to_csv(out_path, index=False)

    print(f"✅ Submission saved to: {out_path}")
    print("✅ Used weights:", weights)


if __name__ == "__main__":
    main()