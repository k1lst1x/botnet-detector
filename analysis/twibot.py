"""TwiBot-22 profile-only baseline with the official train/valid/test split."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline

REFERENCE_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc)
FEATURES = [
    "followers_count",
    "following_count",
    "tweet_count",
    "listed_count",
    "followers_following_ratio",
    "account_age_days",
    "username_length",
    "name_length",
    "description_length",
    "has_description",
    "has_location",
    "verified",
    "protected",
    "has_profile_image",
]
SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "dev": "valid", "test": "test"}
LABELS = {"human": 0, "0": 0, "bot": 1, "1": 1}


class TwiBotFormatError(ValueError):
    """The dataset does not follow the official TwiBot-22 metadata schema."""


def _read_mapping(path: Path, value_column: str) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"id", value_column}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise TwiBotFormatError(f"{path.name}: expected columns id,{value_column}")
        result = {}
        for row in reader:
            key = str(row["id"]).strip()
            if key in result:
                raise TwiBotFormatError(f"{path.name}: duplicate id {key}")
            result[key] = str(row[value_column]).strip().lower()
    return result


def validate_dataset(root: Path) -> tuple[dict[str, int], dict[str, str]]:
    root = Path(root)
    required = [root / "user.json", root / "label.csv", root / "split.csv"]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise TwiBotFormatError(f"missing TwiBot-22 files: {', '.join(missing)}")
    raw_labels = _read_mapping(root / "label.csv", "label")
    raw_splits = _read_mapping(root / "split.csv", "split")
    bad_labels = sorted(set(raw_labels.values()) - set(LABELS))
    bad_splits = sorted(set(raw_splits.values()) - set(SPLIT_ALIASES))
    if bad_labels:
        raise TwiBotFormatError(f"unknown labels: {', '.join(bad_labels)}")
    if bad_splits:
        raise TwiBotFormatError(f"unknown splits: {', '.join(bad_splits)}")
    labels = {key: LABELS[value] for key, value in raw_labels.items()}
    splits = {key: SPLIT_ALIASES[value] for key, value in raw_splits.items()}
    labelled_without_split = set(labels) - set(splits)
    if labelled_without_split:
        raise TwiBotFormatError(f"{len(labelled_without_split)} labelled users have no split")
    return labels, splits


def iter_users(path: Path):
    """Stream the top-level JSON array when ijson is installed."""
    try:
        import ijson
    except ImportError:
        with path.open(encoding="utf-8") as fh:
            yield from json.load(fh)
        return
    with path.open("rb") as fh:
        yield from ijson.items(fh, "item")


def _number(value):
    if value is None or value == "":
        return np.nan
    try:
        number = float(value)
        return number if math.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _flag(value):
    if value is None or value == "":
        return np.nan
    if isinstance(value, bool):
        return float(value)
    return float(str(value).strip().lower() in {"1", "true", "yes"})


def _created_at(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%a %b %d %H:%M:%S %z %Y")
        except ValueError:
            return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def profile_features(user: dict) -> dict[str, float]:
    metrics = user.get("public_metrics") or {}
    followers = _number(metrics.get("followers_count"))
    following = _number(metrics.get("following_count"))
    created = _created_at(user.get("created_at"))
    age = max((REFERENCE_DATE - created).days, 0) if created else np.nan
    username = user.get("username")
    name = user.get("name")
    description = user.get("description")
    image = user.get("profile_image_url")
    return {
        "followers_count": followers,
        "following_count": following,
        "tweet_count": _number(metrics.get("tweet_count")),
        "listed_count": _number(metrics.get("listed_count")),
        "followers_following_ratio": followers / (following + 1) if not np.isnan(followers) and not np.isnan(following) else np.nan,
        "account_age_days": float(age),
        "username_length": float(len(username)) if username is not None else np.nan,
        "name_length": float(len(name)) if name is not None else np.nan,
        "description_length": float(len(description)) if description is not None else np.nan,
        "has_description": float(bool(description)) if description is not None else np.nan,
        "has_location": float(bool(user.get("location"))) if user.get("location") is not None else np.nan,
        "verified": _flag(user.get("verified")),
        "protected": _flag(user.get("protected")),
        "has_profile_image": float(bool(image)) if image is not None else np.nan,
    }


def load_dataset(root: Path, max_per_split: int | None = None):
    labels, splits = validate_dataset(root)
    rows = {name: [] for name in ("train", "valid", "test")}
    seen = set()
    limits = Counter()
    for user in iter_users(Path(root) / "user.json"):
        user_id = str(user.get("id", "")).strip()
        if user_id not in labels or user_id not in splits:
            continue
        split = splits[user_id]
        if max_per_split and limits[split] >= max_per_split:
            continue
        rows[split].append({"id": user_id, "label": labels[user_id], "features": profile_features(user)})
        limits[split] += 1
        seen.add(user_id)
    missing_users = set(labels) - seen
    if missing_users and max_per_split is None:
        raise TwiBotFormatError(f"user.json is missing {len(missing_users)} labelled users")
    for split, items in rows.items():
        if not items:
            raise TwiBotFormatError(f"split {split} has no usable labelled users")
    return rows


def _matrix(items):
    return np.asarray([[item["features"].get(name, np.nan) for name in FEATURES] for item in items], dtype=float)


def _metrics(items, probabilities):
    y = np.asarray([item["label"] for item in items], dtype=int)
    pred = (probabilities >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "humans": int((y == 0).sum()),
        "bots": int((y == 1).sum()),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y, probabilities)), 4) if len(set(y)) > 1 else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def make_classifier(kind="auto", seed=42):
    if kind in {"auto", "xgboost"}:
        try:
            from xgboost import XGBClassifier

            return "xgboost", XGBClassifier(
                n_estimators=350, max_depth=5, learning_rate=0.05, subsample=0.9,
                colsample_bytree=0.9, eval_metric="logloss", random_state=seed, n_jobs=-1,
            )
        except Exception:
            if kind == "xgboost":
                raise
    return "sklearn-hist", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, random_state=seed)


def train_baseline(rows, kind="auto", seed=42):
    y_train = np.asarray([item["label"] for item in rows["train"]], dtype=int)
    if len(set(y_train)) < 2:
        raise TwiBotFormatError("train split must contain both human and bot labels")
    model_name, classifier = make_classifier(kind, seed)
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), classifier)
    model.fit(_matrix(rows["train"]), y_train)
    metrics = {
        "dataset": "TwiBot-22",
        "model": model_name,
        "features": FEATURES,
        "split_source": "official split.csv",
        "train": {"n": len(rows["train"]), "humans": int((y_train == 0).sum()), "bots": int((y_train == 1).sum())},
    }
    for split in ("valid", "test"):
        metrics[split] = _metrics(rows[split], model.predict_proba(_matrix(rows[split]))[:, 1])
    bundle = {"model": model, "features": FEATURES, "kind": model_name, "dataset": "TwiBot-22"}
    return bundle, metrics


def save_artifacts(bundle, metrics, output: Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
