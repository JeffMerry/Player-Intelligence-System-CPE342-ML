"""Task 1 prediction: average the fold models on test.csv and write submissions/task1.csv.

Run from the project root after train.py:  python Task1/predict.py

ID mapping: test.csv uses player_id while sample_submission.csv uses ANS00001...
Every task's test.csv has the same row count as the sample (25,889) but different
player_ids, so rows are matched by position (test row i -> sample row i).
Confirm this with the instructor or the Kaggle page before the final submission.
"""
import joblib
import numpy as np
import pandas as pd

from common import DATA_DIR, MODEL_DIR, SAMPLE_SUB, SUB_PATH, ensemble_proba


def main():
    bundle = joblib.load(MODEL_DIR / "ensemble.joblib")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(SAMPLE_SUB)
    assert len(test) == len(sample), f"row mismatch: test {len(test)} vs sample {len(sample)}"
    assert list(test.columns[1:]) == bundle["features"], "feature columns differ from training"

    X = test[bundle["features"]]
    proba = np.mean([ensemble_proba(m, X) for m in bundle["models"]], axis=0)
    pred = (proba >= bundle["threshold"]).astype(int)

    out = pd.DataFrame({"id": sample["id"], "task1": pred})
    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUB_PATH, index=False)
    print(f"threshold {bundle['threshold']:.3f}; predicted cheater rate {pred.mean():.3f}")
    print(f"wrote {len(out)} rows -> {SUB_PATH}")


if __name__ == "__main__":
    main()
