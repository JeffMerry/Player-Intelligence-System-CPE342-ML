"""Shared paths and helpers for Task 1 (Anti-Cheat Pre-Filter)."""
from pathlib import Path

import numpy as np
from sklearn.metrics import fbeta_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "Task1"
SAMPLE_SUB = ROOT / "data" / "raw" / "sample_submission.csv"
MODEL_DIR = ROOT / "models" / "Task1"
SUB_PATH = ROOT / "submissions" / "task1.csv"
REPORT_PATH = ROOT / "reports" / "experiments" / "task1_cv.json"

ID_COL = "player_id"
TARGET = "is_cheater"
SEED = 42                # CV split seed
SEEDS = [42, 7, 2026]    # model seeds averaged inside each fold
N_FOLDS = 5


def ensemble_proba(models, X):
    """Average of the seed-averaged LightGBM and seed-averaged XGBoost probabilities."""
    per_type = [np.mean([m.predict_proba(X)[:, 1] for m in models[k]], axis=0) for k in ("lgbm", "xgb")]
    return np.mean(per_type, axis=0)


def feature_columns(df):
    return [c for c in df.columns if c not in (ID_COL, TARGET)]


def macro_f2(y_true, y_pred):
    """Competition metric: F2 averaged over both classes (Kaggle: "F2 Score (Macro)")."""
    return fbeta_score(y_true, y_pred, beta=2, average="macro")


def best_f2_threshold(y_true, proba, grid=np.arange(0.05, 0.95, 0.005)):
    """Return (best_macro_f2, best_threshold) over a threshold grid."""
    scores = [macro_f2(y_true, (proba >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(scores[i]), float(grid[i])
