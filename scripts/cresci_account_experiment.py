"""Account level on Cresci-2017: profile vs behaviour vs both, and campaign transfer.

Protocols (fixed XGBoost hyper-parameters, no tuning on test):
 1. cv5      - stratified 5-fold cross-validation over accounts;
 2. loco     - leave-one-campaign-out: the bot group under test is never seen in
               training; genuine accounts are split 70/30 once;
 3. twibot   - the TwiBot-22 profile model (profile-review-v2) applied unchanged.

Population: accounts with at least MIN_POSTS tweets, so all feature sets are
compared on the same accounts. Requires scripts.cresci_features.

Run:  .venv/bin/python -m scripts.cresci_account_experiment
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from engine.features import PORTABLE_FEATURES, PROFILE_V2_FEATURES
from scripts.prepare_cresci import OUT

REPORT = Path("media/models/experiments/cresci-account-v1")
PROFILE_MODEL = Path("media/models/review-profile-v2/model.joblib")
MIN_POSTS = 20
SEED = 42
SETS = {
    "profile": PROFILE_V2_FEATURES,
    "behaviour": PORTABLE_FEATURES,
    "profile+behaviour": PROFILE_V2_FEATURES + PORTABLE_FEATURES,
}
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, tree_method="hist", random_state=SEED, n_jobs=-1)


def scores(y, s, threshold=0.5) -> dict:
    pred = (s >= threshold).astype(int)
    return {"n": int(len(y)), "bots": int(y.sum()), "roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)), "f1": float(f1_score(y, pred)),
            "recall_bot": float(pred[y == 1].mean()),
            "false_positive_rate": float(pred[y == 0].mean())}


def cv5(t: pd.DataFrame, cols: list[str]) -> dict:
    y = t["label"].to_numpy()
    oof = np.zeros(len(t))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(t, t["group"]):
        m = xgb.XGBClassifier(**PARAMS).fit(t.iloc[tr][cols], y[tr])
        oof[te] = m.predict_proba(t.iloc[te][cols])[:, 1]
    return scores(y, oof)


def loco(t: pd.DataFrame, cols: list[str]) -> dict:
    genuine = t[t.group == "genuine"]
    g_train, g_test = train_test_split(genuine, test_size=0.3, random_state=SEED)
    out = {}
    for held in sorted(set(t.group) - {"genuine"}):
        train = pd.concat([g_train, t[(t.group != "genuine") & (t.group != held)]])
        test = pd.concat([g_test, t[t.group == held]])
        m = xgb.XGBClassifier(**PARAMS).fit(train[cols], train["label"])
        out[held] = scores(test["label"].to_numpy(), m.predict_proba(test[cols])[:, 1])
    return out


def twibot_transfer(t: pd.DataFrame) -> dict:
    bundle = joblib.load(PROFILE_MODEL)
    s = bundle["model"].predict_proba(t[bundle["features"]].to_numpy(dtype=float))[:, 1]
    y = t["label"].to_numpy()
    result = {"all": scores(y, s, bundle["threshold"])}
    for g in sorted(set(t.group) - {"genuine"}):
        mask = t.group.isin(["genuine", g]).to_numpy()
        result[g] = scores(y[mask], s[mask], bundle["threshold"])
    return result


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(OUT / "account_features.parquet")
    t = table[table.observed_posts >= MIN_POSTS].reset_index(drop=True)
    report = {
        "population": {"min_posts": MIN_POSTS, "by_group": t.group.value_counts().to_dict()},
        "params": PARAMS,
        "cv5": {}, "loco": {},
    }
    for name, cols in SETS.items():
        report["cv5"][name] = cv5(t, cols)
        report["loco"][name] = loco(t, cols)
        print(name, "cv5 ROC", round(report["cv5"][name]["roc_auc"], 4), flush=True)
        for g, r in report["loco"][name].items():
            print(f"   loco {g:15s} ROC {r['roc_auc']:.4f} recall {r['recall_bot']:.3f} "
                  f"FPR {r['false_positive_rate']:.3f}", flush=True)
    report["twibot_profile_model_all_accounts"] = twibot_transfer(table.reset_index(drop=True))
    for g, r in report["twibot_profile_model_all_accounts"].items():
        print(f"twibot->cresci {g:15s} ROC {r['roc_auc']:.4f} recall {r['recall_bot']:.3f} "
              f"FPR {r['false_positive_rate']:.3f}")
    (REPORT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
