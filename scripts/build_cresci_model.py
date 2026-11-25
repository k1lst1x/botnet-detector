"""Fit and package the Cresci-2017 account model (profile + behaviour) as a review-queue artifact.

Same table, population and XGBoost parameters as scripts.cresci_account_experiment, wrapped with
a median imputer because ModelService expects imputer + XGBoost. The wrapped configuration is
therefore re-evaluated here (5-fold CV and leave-one-campaign-out), then fitted on all accounts.
The artifact is for research: Threads quality is not validated and it is not the active model.

Run:  .venv/bin/python -m scripts.build_cresci_model
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from engine.experiment import save_json
from engine.features import PORTABLE_FEATURES, PROFILE_V2_FEATURES, PROFILE_VERSION
from engine.model import ModelService
from engine.shard_index import file_hash
from scripts.cresci_account_experiment import MIN_POSTS, PARAMS, SEED, scores
from scripts.prepare_cresci import OUT

OUTPUT = Path("media/models/cresci-account-v1")
FEATURES = PROFILE_V2_FEATURES + PORTABLE_FEATURES
THRESHOLD = 0.5


def fit(x, y):
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("xgboost", xgb.XGBClassifier(**PARAMS))]).fit(x, y)


def main():
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} already exists")
    source = OUT / "account_features.parquet"
    t = pd.read_parquet(source)
    t = t[t.observed_posts >= MIN_POSTS].reset_index(drop=True)
    x, y, group = t[FEATURES].to_numpy(dtype=float), t["label"].to_numpy(), t["group"].to_numpy()

    oof = np.zeros(len(t))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, group):
        oof[te] = fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    g_train, g_test = train_test_split(np.where(group == "genuine")[0], test_size=0.3, random_state=SEED)
    loco = {}
    for held in sorted(set(group) - {"genuine"}):
        tr = np.r_[g_train, np.where((group != "genuine") & (group != held))[0]]
        te = np.r_[g_test, np.where(group == held)[0]]
        loco[held] = scores(y[te], fit(x[tr], y[tr]).predict_proba(x[te])[:, 1], THRESHOLD)

    OUTPUT.mkdir(parents=True)
    joblib.dump({
        "model": fit(x, y),
        "features": FEATURES,
        "feature_version": PROFILE_VERSION,
        "feature_set": "portable",
        "model_version": "cresci-account-v1",
        "threshold": THRESHOLD,
        "required_observed_features": PORTABLE_FEATURES,
        "minimum_observed_ratio": 0.75,
        "minimum_posts": MIN_POSTS,
        "decision_policy": "review_queue",
        "calibrated": False,
        "threads_quality_validated": False,
    }, OUTPUT / "model.joblib")
    status = ModelService(OUTPUT / "model.joblib").status()
    if status["status"] != "ready":
        raise ValueError(f"ModelService rejected the artifact: {status}")
    report = {
        "source": str(source), "source_sha256": file_hash(source),
        "model_sha256": file_hash(OUTPUT / "model.joblib"),
        "population": {"min_posts": MIN_POSTS, "by_group": t.group.value_counts().to_dict()},
        "params": PARAMS, "threshold": THRESHOLD,
        "cv5": scores(y, oof, THRESHOLD), "loco": loco,
        "independent_evaluation": False, "threads_quality_validated": False,
    }
    save_json(OUTPUT / "report.json", report)
    print(json.dumps({"cv5": report["cv5"], "loco": {k: round(v["roc_auc"], 4) for k, v in loco.items()},
                      "sha256": report["model_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
