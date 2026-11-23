"""Train profile models on the full TwiBot-22 user set (700k/200k/100k official split).

Three feature sets are compared with identical training protocol:

* ``baseline5``  – the five profile features of the active review model;
* ``portable``   – profile fields that the Threads API also exposes
                   (followers, biography, avatar, verified, name, username);
* ``twitter``    – ``portable`` plus Twitter-only fields (following, tweet and
                   list counts, account age, location, link, protected, pinned).

Hyper-parameters and early stopping use only train/val. The test split is read
once, after the configuration of every feature set has been fixed.

Run:  .venv/bin/python -m scripts.train_full_profile [--output DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import ijson
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from engine.features import PROFILE_V2_FEATURES, profile_text_features

RAW = Path("data/raw/twibot22")
CACHE = Path("data/processed/twibot22-profile-full-v2")
SHARD7_VAL = Path("data/processed/twibot22-shard7-training-v1/val.parquet")
ACTIVE_QUEUE_FRACTION = 2520 / 29917  # capacity of the active review queue
SEED = 42

BASELINE5 = [
    "followers_log1p", "following_log1p", "followers_following_ratio",
    "bio_present", "avatar_present",
]
PORTABLE = PROFILE_V2_FEATURES
TWITTER_ONLY = [
    "following_log1p", "followers_following_ratio", "tweet_count_log1p",
    "listed_count_log1p", "account_age_days", "tweets_per_day",
    "followers_per_day", "following_per_day", "listed_per_follower",
    "has_location", "has_url", "protected", "has_pinned_tweet",
]
FEATURE_SETS = {
    "baseline5": BASELINE5,
    "portable": PORTABLE,
    "twitter": PORTABLE + TWITTER_ONLY,
}
GRID = [
    {"max_depth": d, "min_child_weight": w}
    for d in (4, 6, 8)
    for w in (1, 10)
]

def twibot_profile_fields(user: dict) -> dict:
    """Map a raw TwiBot-22 user to the AccountSnapshot profile fields."""
    image = user.get("profile_image_url") or ""
    return {
        "biography": user.get("description") or "",
        "display_name": user.get("name") or "",
        "username": user.get("username") or "",
        "has_avatar": bool(image),
        "avatar_default": "default_profile_images" in image,
        "is_verified": bool(user.get("verified")),
    }


def profile_features(user: dict, reference: datetime) -> dict:
    metrics = user.get("public_metrics") or {}
    followers = float(metrics.get("followers_count") or 0)
    following = float(metrics.get("following_count") or 0)
    tweets = float(metrics.get("tweet_count") or 0)
    listed = float(metrics.get("listed_count") or 0)
    fields = twibot_profile_fields(user)
    created = user.get("created_at")
    age = None
    if created:
        age = max((reference - datetime.fromisoformat(created)).total_seconds() / 86400, 1.0)
    row = profile_text_features(
        fields["biography"], fields["display_name"], fields["username"],
        fields["avatar_default"], fields["is_verified"],
    )
    row.update({
        "followers_log1p": math.log1p(followers),
        "following_log1p": math.log1p(following),
        "followers_following_ratio": followers / (following + 1),
        "bio_present": float(bool(unicodedata.normalize("NFKC", fields["biography"]).strip())),
        "avatar_present": float(fields["has_avatar"]),
        "tweet_count_log1p": math.log1p(tweets),
        "listed_count_log1p": math.log1p(listed),
        "account_age_days": age if age is not None else np.nan,
        "tweets_per_day": tweets / age if age else np.nan,
        "followers_per_day": followers / age if age else np.nan,
        "following_per_day": following / age if age else np.nan,
        "listed_per_follower": listed / (followers + 1),
        "has_location": float(bool((user.get("location") or "").strip())),
        "has_url": float(bool((user.get("url") or "").strip())),
        "protected": float(bool(user.get("protected"))),
        "has_pinned_tweet": float(bool(user.get("pinned_tweet_id"))),
    })
    return row


def _read_csv(path: Path, column: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {row["id"]: row[column] for row in csv.DictReader(fh)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build_table() -> pd.DataFrame:
    cache = CACHE / "profiles.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    labels = _read_csv(RAW / "label.csv", "label")
    splits = _read_csv(RAW / "split.csv", "split")
    with (RAW / "user.json").open("rb") as fh:
        created = [
            datetime.fromisoformat(v)
            for v in ijson.items(fh, "item.created_at")
            if v
        ]
    reference = max(created)  # latest account creation = collection snapshot proxy
    del created
    rows = []
    with (RAW / "user.json").open("rb") as fh:
        for user in ijson.items(fh, "item"):
            uid = user["id"]
            row = profile_features(user, reference)
            row["id"] = uid
            row["label"] = 1 if labels[uid] == "bot" else 0
            row["split"] = splits[uid]
            rows.append(row)
    table = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache, index=False)
    (CACHE / "manifest.json").write_text(json.dumps({
        "rows": len(table),
        "counts": table["split"].value_counts().to_dict(),
        "reference_date": reference.isoformat(),
        "inputs_sha256": {n: _sha256(RAW / n) for n in ("user.json", "label.csv", "split.csv")},
        "features": sorted(c for c in table.columns if c not in {"id", "label", "split"}),
    }, indent=2, ensure_ascii=False))
    return table


def queue_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    order = np.argsort(-score, kind="stable")
    ranked = y[order]
    cum = np.cumsum(ranked)
    n, positives = len(y), int(y.sum())
    result = {}
    for frac in (0.01, 0.02, 0.05, ACTIVE_QUEUE_FRACTION, 0.10, 0.20):
        k = max(1, int(round(frac * n)))
        result[f"top_{frac:.4f}"] = {
            "k": k,
            "precision": float(cum[k - 1] / k),
            "recall": float(cum[k - 1] / positives),
        }
    precision = cum / np.arange(1, n + 1)
    ok = np.where((precision >= 0.85) & (np.arange(1, n + 1) >= 100))[0]
    k85 = int(ok.max() + 1) if len(ok) else None
    result["largest_queue_precision_0.85"] = None if k85 is None else {
        "k": k85, "fraction": k85 / n, "recall": float(cum[k85 - 1] / positives),
    }
    return result


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    return {
        "n": int(len(y)),
        "bot_share": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "threshold": float(threshold),
        "f1_bot": float(f1_score(y, pred)),
        "precision_bot": float(tp / max(pred.sum(), 1)),
        "recall_bot": float(tp / max(y.sum(), 1)),
        "accuracy": float((pred == y).mean()),
        "queue": queue_metrics(y, score),
    }


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    ranked, s = y[order], score[order]
    tp = np.cumsum(ranked)
    k = np.arange(1, len(y) + 1)
    f1 = 2 * tp / (k + y.sum())
    return float(s[int(np.argmax(f1))])


def train_set(name: str, cols: list[str], data: dict, out: Path) -> dict:
    Xtr, ytr = data["train"][cols], data["train"]["label"].to_numpy()
    Xva, yva = data["val"][cols], data["val"]["label"].to_numpy()
    trials = []
    best = None
    for params in GRID:
        model = xgb.XGBClassifier(
            n_estimators=3000, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
            early_stopping_rounds=100, random_state=SEED, n_jobs=-1, **params,
        )
        start = time.time()
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        score = model.predict_proba(Xva)[:, 1]
        trial = {
            **params,
            "best_iteration": int(model.best_iteration),
            "val_pr_auc": float(average_precision_score(yva, score)),
            "val_roc_auc": float(roc_auc_score(yva, score)),
            "seconds": round(time.time() - start, 1),
        }
        trials.append(trial)
        print(f"  {name} {params} -> PR-AUC {trial['val_pr_auc']:.4f} "
              f"ROC-AUC {trial['val_roc_auc']:.4f} ({trial['seconds']}s)", flush=True)
        if best is None or trial["val_pr_auc"] > best[0]["val_pr_auc"]:
            best = (trial, model, score)
    trial, model, val_score = best
    threshold = best_f1_threshold(yva, val_score)
    joblib.dump({"model": model, "features": cols, "threshold": threshold}, out / f"{name}.joblib")
    importance = model.get_booster().get_score(importance_type="gain")
    return {
        "features": cols,
        "grid": trials,
        "selected": trial,
        "threshold_from_val": threshold,
        "val": evaluate(yva, val_score, threshold),
        "importance_gain": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="media/models/experiments/full-profile-v2")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    t0 = time.time()
    table = build_table()
    print(f"table {table.shape} in {time.time() - t0:.0f}s", flush=True)
    data = {s: table[table["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    shard7_ids = set(pd.read_parquet(SHARD7_VAL, columns=["external_user_id"])["external_user_id"])
    shard7_mask = data["val"]["id"].isin(shard7_ids).to_numpy()

    report = {
        "protocol": "official TwiBot-22 split; grid + early stopping on val; test read once after selection",
        "counts": {s: {"n": len(d), "bots": int(d["label"].sum())} for s, d in data.items()},
        "sets": {},
    }
    for name, cols in FEATURE_SETS.items():
        report["sets"][name] = train_set(name, cols, data, out)
        yva = data["val"]["label"].to_numpy()
        bundle = joblib.load(out / f"{name}.joblib")
        val_score = bundle["model"].predict_proba(data["val"][cols])[:, 1]
        report["sets"][name]["val_shard7_subset"] = evaluate(
            yva[shard7_mask], val_score[shard7_mask], bundle["threshold"])

    # Single read of the test split, after every configuration is frozen.
    yte = data["test"]["label"].to_numpy()
    for name in FEATURE_SETS:
        bundle = joblib.load(out / f"{name}.joblib")
        score = bundle["model"].predict_proba(data["test"][bundle["features"]])[:, 1]
        report["sets"][name]["test"] = evaluate(yte, score, bundle["threshold"])

    report["versions"] = {"xgboost": xgb.__version__, "pandas": pd.__version__}
    report["seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for name, r in report["sets"].items():
        print(f"{name:10s} val ROC {r['val']['roc_auc']:.4f} PR {r['val']['pr_auc']:.4f} | "
              f"test ROC {r['test']['roc_auc']:.4f} PR {r['test']['pr_auc']:.4f} "
              f"F1 {r['test']['f1_bot']:.4f}")


if __name__ == "__main__":
    main()
