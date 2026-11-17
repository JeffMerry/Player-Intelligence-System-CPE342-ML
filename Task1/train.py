"""Task 1 training: 5-fold CV of a seed-averaged LightGBM + XGBoost ensemble, macro-F2 threshold.

Run from the project root:  python Task1/train.py

- Drops train rows whose label is missing (unlabeled).
- Missing feature values are left as NaN (both libraries handle them natively;
  EDA showed missingness is unrelated to the target).
- Each fold trains LightGBM and XGBoost with several seeds; probabilities are averaged.
- The decision threshold is chosen on out-of-fold probabilities to maximise macro F2
  (the Kaggle metric is "F2 Score (Macro)").
- Saves the fold models (predict.py averages them on test.csv), the OOF probabilities
  and the LightGBM feature importance.
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import (DATA_DIR, MODEL_DIR, N_FOLDS, REPORT_PATH, SEED, SEEDS, TARGET,
                    best_f2_threshold, ensemble_proba, feature_columns, macro_f2)

LGB_PARAMS = dict(n_estimators=1300, learning_rate=0.02, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, verbose=-1)
XGB_PARAMS = dict(n_estimators=1000, learning_rate=0.02, max_depth=6, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, tree_method="hist")


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    n_raw = len(train)
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    features = feature_columns(train)
    X, y = train[features], train[TARGET].astype(int).values
    print(f"train rows: {n_raw} raw, {len(train)} labeled, cheater rate {y.mean():.3f}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))
    oof = {"lgbm": np.zeros(len(y)), "xgb": np.zeros(len(y))}
    fold_models, importance = [], np.zeros(len(features))
    for fold, (tr_idx, va_idx) in enumerate(folds):
        Xtr, ytr, Xva = X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx]
        models = {"lgbm": [lgb.LGBMClassifier(**LGB_PARAMS, random_state=s).fit(Xtr, ytr) for s in SEEDS],
                  "xgb": [xgb.XGBClassifier(**XGB_PARAMS, random_state=s).fit(Xtr, ytr) for s in SEEDS]}
        for name, ms in models.items():
            oof[name][va_idx] = np.mean([m.predict_proba(Xva)[:, 1] for m in ms], axis=0)
        importance += np.mean([m.booster_.feature_importance("gain") for m in models["lgbm"]], axis=0)
        fold_models.append(models)
        print(f"fold {fold}: AUC {roc_auc_score(y[va_idx], ensemble_proba(models, Xva)):.4f}", flush=True)

    oof["ensemble"] = (oof["lgbm"] + oof["xgb"]) / 2
    results = {}
    for name, p in oof.items():
        f2, thr = best_f2_threshold(y, p)
        results[name] = {"auc": roc_auc_score(y, p), "macro_f2": f2, "threshold": thr}
        print(f"{name:9s} OOF AUC {results[name]['auc']:.4f}  macro F2 {f2:.4f} @ thr {thr:.3f}")

    thr = results["ensemble"]["threshold"]
    pred = (oof["ensemble"] >= thr).astype(int)
    results["ensemble"].update(
        precision=precision_score(y, pred), recall=recall_score(y, pred),
        f2_class1_only=fbeta_score(y, pred, beta=2),
        predicted_positive_rate=float(pred.mean()),
        fold_macro_f2=[macro_f2(y[va], pred[va]) for _, va in folds])
    r = results["ensemble"]
    print(f"ensemble @ {thr:.3f}: precision {r['precision']:.4f} recall {r['recall']:.4f} "
          f"fold macro F2 {np.round(r['fold_macro_f2'], 4).tolist()}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fold_models, "features": features, "threshold": thr},
                MODEL_DIR / "ensemble.joblib")
    np.savez(MODEL_DIR / "oof.npz", y=y, **oof)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    (pd.Series(importance / N_FOLDS, index=features, name="lgbm_gain").sort_values(ascending=False)
     .to_csv(REPORT_PATH.with_name("task1_feature_importance.csv")))
    REPORT_PATH.write_text(json.dumps({
        "cv_seed": SEED, "model_seeds": SEEDS, "n_folds": N_FOLDS, "n_labeled": int(len(y)),
        "lgb_params": LGB_PARAMS, "xgb_params": XGB_PARAMS,
        "versions": {"lightgbm": lgb.__version__, "xgboost": xgb.__version__},
        "results": results}, indent=2))
    print(f"saved model -> {MODEL_DIR / 'ensemble.joblib'}\nsaved report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
