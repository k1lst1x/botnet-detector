"""Compare classifiers on the full TwiBot-22 profile table (same protocol as train_full_profile).

Logistic regression, random forest and CatBoost are trained from scratch on the
``portable`` and ``twitter`` feature sets. Hyper-parameters are chosen on val
(PR-AUC); the test split is read once, after all configurations are fixed.
XGBoost results are taken from the full-profile-v2 report (same table, same split).

Run:  .venv/bin/python -m scripts.compare_algorithms [--output DIR]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from scripts.train_full_profile import (
    FEATURE_SETS, SEED, SHARD7_VAL, best_f1_threshold, build_table, evaluate,
)

XGB_REPORT = Path("media/models/experiments/full-profile-v3/report.json")
SETS = ("portable", "twitter")


def logreg(C):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log1p),  # every feature is non-negative and heavy-tailed
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000),
    )


def forest(leaf):
    return RandomForestClassifier(
        n_estimators=200, min_samples_leaf=leaf, max_features="sqrt",
        n_jobs=-1, random_state=SEED,
    )


ALGORITHMS = {
    "logreg": [({"C": c}, lambda c=c: logreg(c)) for c in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [({"min_samples_leaf": m}, lambda m=m: forest(m)) for m in (10, 30, 100)],
    "catboost": [({"depth": d}, None) for d in (4, 6, 8)],
}


def fit_one(algo, params, factory, Xtr, ytr, Xva, yva):
    if algo == "catboost":
        model = CatBoostClassifier(
            iterations=3000, learning_rate=0.05, depth=params["depth"],
            eval_metric="PRAUC", od_type="Iter", od_wait=100, random_seed=SEED,
            thread_count=-1, verbose=False, allow_writing_files=False,
        )
        model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        params = {**params, "best_iteration": int(model.get_best_iteration())}
    else:
        model = factory()
        model.fit(Xtr, ytr)
    return model, params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="media/models/experiments/algorithm-comparison-v2")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    t0 = time.time()
    table = build_table()
    data = {s: table[table["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    shard7_ids = set(pd.read_parquet(SHARD7_VAL, columns=["external_user_id"])["external_user_id"])
    shard7 = data["val"]["id"].isin(shard7_ids).to_numpy()
    ytr, yva, yte = (data[s]["label"].to_numpy() for s in ("train", "val", "test"))

    report = {"protocol": "grid on val (PR-AUC); test read once after selection", "results": {}}
    chosen = {}
    for fset in SETS:
        cols = FEATURE_SETS[fset]
        Xtr, Xva = data["train"][cols], data["val"][cols]
        for algo, grid in ALGORITHMS.items():
            trials, best = [], None
            for params, factory in grid:
                start = time.time()
                model, params = fit_one(algo, params, factory, Xtr, ytr, Xva, yva)
                score = model.predict_proba(Xva)[:, 1]
                trial = {**params,
                         "val_pr_auc": float(average_precision_score(yva, score)),
                         "val_roc_auc": float(roc_auc_score(yva, score)),
                         "seconds": round(time.time() - start, 1)}
                trials.append(trial)
                print(f"{fset} {algo} {params} -> PR-AUC {trial['val_pr_auc']:.4f} "
                      f"ROC-AUC {trial['val_roc_auc']:.4f} ({trial['seconds']}s)", flush=True)
                if best is None or trial["val_pr_auc"] > best[0]["val_pr_auc"]:
                    best = (trial, model, score)
            trial, model, score = best
            threshold = best_f1_threshold(yva, score)
            path = out / f"{fset}-{algo}.joblib"
            joblib.dump({"model": model, "features": cols, "threshold": threshold}, path)
            chosen[(fset, algo)] = path
            report["results"].setdefault(fset, {})[algo] = {
                "grid": trials, "selected": trial, "threshold_from_val": threshold,
                "val": evaluate(yva, score, threshold),
                "val_shard7_subset": evaluate(yva[shard7], score[shard7], threshold),
            }
            del model, best

    # Single read of test, after every configuration above is frozen.
    for (fset, algo), path in chosen.items():
        bundle = joblib.load(path)
        score = bundle["model"].predict_proba(data["test"][bundle["features"]])[:, 1]
        report["results"][fset][algo]["test"] = evaluate(yte, score, bundle["threshold"])

    xgb = json.loads(XGB_REPORT.read_text())
    for fset in SETS:
        src = xgb["sets"][fset]
        report["results"][fset]["xgboost"] = {
            k: src[k] for k in ("grid", "selected", "threshold_from_val", "val", "val_shard7_subset", "test")
        }
        report["results"][fset]["xgboost"]["source"] = str(XGB_REPORT)

    report["seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n{'set':9s} {'algorithm':14s} {'valROC':>7s} {'valPR':>7s} {'testROC':>8s} {'testPR':>7s} {'testF1':>7s}")
    for fset, algos in report["results"].items():
        for algo, r in algos.items():
            print(f"{fset:9s} {algo:14s} {r['val']['roc_auc']:7.4f} {r['val']['pr_auc']:7.4f} "
                  f"{r['test']['roc_auc']:8.4f} {r['test']['pr_auc']:7.4f} {r['test']['f1_bot']:7.4f}")


if __name__ == "__main__":
    main()
