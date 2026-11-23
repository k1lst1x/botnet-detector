"""Overfitting, leakage and robustness audit of the profile-v2 model (no model replacement).

Checks:
 1. train / val / test gap of the packaged model;
 2. learning curve (10 %, 30 %, 100 % of train) with the selected hyper-parameters;
 3. seed stability (5 seeds, full train);
 4. bootstrap 95 % CI on test (1000 resamples);
 5. leakage: duplicate ids across splits, identical feature vectors across splits
    and their label agreement;
 6. robustness to fields Threads may not expose (feature set to missing → imputed).

Run:  .venv/bin/python -m scripts.audit_profile_v2
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from engine.features import PROFILE_V2_FEATURES
from scripts.train_full_profile import CACHE, SEED

MODEL = Path("media/models/review-profile-v2/model.joblib")
FULL = Path("media/models/experiments/full-profile-v3/report.json")
OUT = Path("media/models/review-profile-v2/audit.json")


def scores(y, s):
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def fit(X, y, Xv, yv, params, seed):
    m = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", eval_metric="aucpr", early_stopping_rounds=100,
        random_state=seed, n_jobs=-1, **params,
    )
    m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    return m


def main():
    names = PROFILE_V2_FEATURES
    t = pd.read_parquet(CACHE / "profiles.parquet", columns=["id", "split", "label", *names])
    parts = {s: t[t.split == s] for s in ("train", "val", "test")}
    X = {s: d[names].to_numpy() for s, d in parts.items()}
    y = {s: d["label"].to_numpy() for s, d in parts.items()}
    bundle = joblib.load(MODEL)
    pipe = bundle["model"]
    report = {}

    report["gap"] = {s: scores(y[s], pipe.predict_proba(X[s])[:, 1]) for s in X}
    print("gap", report["gap"], flush=True)

    sel = json.loads(FULL.read_text())["sets"]["portable"]["selected"]
    params = {"max_depth": sel["max_depth"], "min_child_weight": sel["min_child_weight"]}
    rng = np.random.default_rng(SEED)
    curve = []
    for frac in (0.1, 0.3, 1.0):
        idx = rng.permutation(len(y["train"]))[: int(frac * len(y["train"]))]
        m = fit(X["train"][idx], y["train"][idx], X["val"], y["val"], params, SEED)
        curve.append({"train_fraction": frac, "n": len(idx), "best_iteration": int(m.best_iteration),
                      "train": scores(y["train"][idx], m.predict_proba(X["train"][idx])[:, 1]),
                      "val": scores(y["val"], m.predict_proba(X["val"])[:, 1])})
        print("curve", curve[-1], flush=True)
    report["learning_curve"] = curve

    seeds = []
    for seed in range(5):
        m = fit(X["train"], y["train"], X["val"], y["val"], params, seed)
        seeds.append({"seed": seed, **scores(y["val"], m.predict_proba(X["val"])[:, 1])})
    report["seed_stability_val"] = {
        "runs": seeds,
        "roc_auc_std": float(np.std([r["roc_auc"] for r in seeds])),
        "pr_auc_std": float(np.std([r["pr_auc"] for r in seeds])),
    }
    print("seeds", report["seed_stability_val"], flush=True)

    s_test = pipe.predict_proba(X["test"])[:, 1]
    boot = {"roc_auc": [], "pr_auc": []}
    n = len(y["test"])
    for _ in range(1000):
        i = rng.integers(0, n, n)
        boot["roc_auc"].append(roc_auc_score(y["test"][i], s_test[i]))
        boot["pr_auc"].append(average_precision_score(y["test"][i], s_test[i]))
    report["test_bootstrap_95ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                     for k, v in boot.items()}
    print("ci", report["test_bootstrap_95ci"], flush=True)

    ids = {s: set(d["id"]) for s, d in parts.items()}
    rounded = t[names].round(6).astype(str).agg("|".join, axis=1)
    t = t.assign(key=rounded)
    train_keys = t[t.split == "train"].groupby("key")["label"].mean()
    leak = {}
    for s in ("val", "test"):
        d = t[t.split == s]
        hit = d[d.key.isin(train_keys.index)]
        agree = ((train_keys.loc[hit.key].to_numpy() >= 0.5).astype(int) == hit.label.to_numpy()).mean() if len(hit) else None
        leak[s] = {"id_overlap_with_train": len(ids[s] & ids["train"]),
                   "rows_with_identical_train_vector": int(len(hit)),
                   "share": float(len(hit) / len(d)),
                   "majority_label_agreement": None if agree is None else float(agree),
                   "bot_share_among_duplicates": float(hit.label.mean()) if len(hit) else None}
    leak["val_test_id_overlap"] = len(ids["val"] & ids["test"])
    report["leakage"] = leak
    print("leak", leak, flush=True)

    base = scores(y["test"], s_test)
    groups = {
        "avatar_default": ["avatar_default"],
        "avatar (both)": ["avatar_present", "avatar_default"],
        "display name": ["name_length", "name_digit_ratio", "name_non_alnum_ratio", "name_username_overlap"],
        "biography": ["bio_present", "bio_length", "bio_url_count", "bio_mention_count",
                      "bio_hashtag_count", "bio_digit_ratio"],
    }
    robust = {"all_fields": base}
    for label, cols in groups.items():
        Xm = X["test"].copy()
        for c in cols:
            Xm[:, names.index(c)] = np.nan
        robust[f"without {label}"] = scores(y["test"], pipe.predict_proba(Xm)[:, 1])
    report["missing_field_robustness_test"] = robust
    print("robust", robust, flush=True)

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
