"""Package the full-TwiBot-22 portable XGBoost as a review-queue model; no re-fitting of trees.

The trained XGBClassifier is wrapped with a median imputer fitted on train, so the
artifact matches ModelService (imputer + XGBoost pipeline). Unknown Threads fields
(e.g. default avatar) are imputed instead of following an unseen missing branch.
The threshold is the highest-recall point with precision >= target on val.

Run:  .venv/bin/python -m scripts.build_profile_v2_model [--precision 0.7]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from engine.experiment import choose_thresholds, save_json
from engine.features import PROFILE_V2_FEATURES, PROFILE_VERSION
from engine.model import ModelService, metrics
from engine.shard_index import file_hash
from scripts.train_full_profile import CACHE, queue_metrics

SOURCE = Path("media/models/experiments/full-profile-v2")
REQUIRED = ["followers_log1p", "bio_present", "username_length", "name_length"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=float, default=0.7)
    parser.add_argument("--output", default="media/models/review-profile-v2")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} already exists")

    source = joblib.load(SOURCE / "portable.joblib")
    names = source["features"]
    if names != PROFILE_V2_FEATURES:
        raise ValueError("Artifact features differ from engine profile-v2")
    table = pd.read_parquet(CACHE / "profiles.parquet", columns=["split", "label", *names])
    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]

    # The system scores plain arrays in `names` order; drop DataFrame column names.
    source["model"].get_booster().feature_names = None
    if hasattr(source["model"], "feature_names_in_"):
        del source["model"].feature_names_in_
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(train[names].to_numpy())
    pipe = Pipeline([("imputer", imputer), ("xgboost", source["model"])])
    y = val["label"].to_numpy()
    scores = pipe.predict_proba(val[names].to_numpy())[:, 1]
    direct = source["model"].predict_proba(val[names].to_numpy())[:, 1]
    if not np.allclose(scores, direct):
        raise ValueError("Imputer changed predictions on fully observed rows")
    target = choose_thresholds(y, scores, args.precision, 100)["precision_target"]
    if target is None:
        raise ValueError("Precision target unreachable on val")

    bundle = {
        "model": pipe,
        "features": names,
        "feature_version": PROFILE_VERSION,
        "feature_set": "profile",
        "model_version": "profile-review-v2",
        "threshold": target["threshold"],
        "required_observed_features": REQUIRED,
        "minimum_observed_ratio": 0.75,
        "minimum_posts": 0,
        "decision_policy": "review_queue",
        "calibrated": False,
        "threads_quality_validated": False,
    }
    out.mkdir(parents=True)
    joblib.dump(bundle, out / "model.joblib")
    if ModelService(out / "model.joblib").status()["status"] != "ready":
        raise ValueError("ModelService rejected the artifact")
    save_json(out / "report.json", {
        "source": str(SOURCE / "portable.joblib"),
        "source_sha256": file_hash(SOURCE / "portable.joblib"),
        "model_sha256": file_hash(out / "model.joblib"),
        "training": "TwiBot-22 full official train split (700000 profiles), XGBoost, see docs/reports/2026-11-23-full-profile-comparison.md",
        "policy": f"manual review queue; threshold = max recall with val precision >= {args.precision}",
        "validation": metrics(y, scores, target["threshold"]),
        "selection": target,
        "queue_fraction_val": target["predicted_positive"] / len(y),
        "queue": queue_metrics(y, scores),
        "test_metrics_source": "media/models/experiments/full-profile-v2/report.json (sets.portable.test)",
        "independent_evaluation": False,
        "threads_quality_validated": False,
    })
    print(json.dumps({k: target[k] for k in target}, indent=2))
    print("fraction of val in queue:", round(target["predicted_positive"] / len(y), 4))


if __name__ == "__main__":
    main()
