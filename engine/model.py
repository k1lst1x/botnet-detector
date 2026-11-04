"""Explicit offline training; inference never fits or substitutes heuristics."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted

from .features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    PORTABLE_FEATURES,
    PROFILE_FEATURES,
)


class ModelUnavailable(RuntimeError):
    pass


def matrix(rows, names):
    return np.array(
        [
            [np.nan if row.get(n) is None else float(row[n]) for n in names]
            for row in rows
        ],
        dtype=float,
    )


class ModelService:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.bundle = None
        self.reason = "model_not_ready"
        if self.path.is_file():
            try:
                # Only operator-controlled artifacts are trusted; no upload-model endpoint.
                bundle = joblib.load(self.path)
                names = bundle["features"]
                if (
                    bundle["feature_version"] != FEATURE_VERSION
                    or not names
                    or not set(names) <= set(FEATURE_NAMES)
                ):
                    raise ValueError("Incompatible features")
                if len(set(names)) != len(names) or not 0 <= bundle["threshold"] <= 1:
                    raise ValueError("Invalid artifact")
                if (
                    not isinstance(bundle["model"], Pipeline)
                    or len(bundle["model"].steps) != 2
                ):
                    raise ValueError("Expected imputer and XGBoost pipeline")
                check_is_fitted(bundle["model"])
                if bundle["model"].n_features_in_ != len(names):
                    raise ValueError("Model feature shape mismatch")
                bundle["model"].steps[-1][1].get_booster()
                if not isinstance(bundle["model_version"], str) or bundle[
                    "feature_set"
                ] not in ("profile", "portable"):
                    raise ValueError("Invalid model metadata")
                required = bundle["required_observed_features"]
                if (
                    not required
                    or not set(required) <= set(names)
                    or any(n.endswith("__missing") for n in required)
                ):
                    raise ValueError("Invalid coverage requirement")
                self.bundle = bundle
            except Exception:
                self.reason = "model_incompatible"

    def status(self):
        if self.bundle is None:
            return {
                "status": self.reason,
                "model_version": None,
                "feature_version": FEATURE_VERSION,
            }
        return {
            "status": "ready",
            "model_version": self.bundle["model_version"],
            "feature_version": FEATURE_VERSION,
            "feature_set": self.bundle["feature_set"],
        }

    def predict(self, rows):
        if self.bundle is None:
            raise ModelUnavailable(self.reason)
        names = self.bundle["features"]
        if any(not set(names) <= row.keys() for row in rows):
            raise ValueError("Feature schema missing fields")
        return self.bundle["model"].predict_proba(matrix(rows, names))[:, 1].tolist()


def metrics(y, probabilities, threshold):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)
    pred = probabilities >= threshold
    return {
        "n": len(y),
        "threshold": float(threshold),
        "precision_bot": float(precision_score(y, pred, zero_division=0)),
        "recall_bot": float(recall_score(y, pred, zero_division=0)),
        "f1_bot": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(y, probabilities))
        if len(set(y)) == 2
        else None,
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(set(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def validate_prepared(path: Path):
    """Verify split boundaries and training eligibility without fitting anything."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest["feature_version"] != FEATURE_VERSION
        or manifest["feature_names"] != FEATURE_NAMES
    ):
        raise ValueError("Prepared feature schema mismatch")
    seen = set()
    report = {}
    for split in ("train", "val", "test"):
        count = 0
        classes = set()
        for batch in pq.ParquetFile(path / f"{split}.parquet").iter_batches():
            for row in batch.to_pylist():
                if row["split"] != split or row["label"] not in (0, 1):
                    raise ValueError("Invalid split or label")
                if row["external_user_id"] in seen:
                    raise ValueError("Duplicate/cross-split identity")
                seen.add(row["external_user_id"])
                count += 1
                classes.add(row["label"])
                for name in FEATURE_NAMES:
                    if row[name] is not None and not np.isfinite(row[name]):
                        raise ValueError("Nonfinite feature")
        if count != manifest["counts"][split]:
            raise ValueError("Manifest count mismatch")
        report[split] = {"rows": count, "classes": sorted(classes)}
    if sum(r["rows"] for r in report.values()) != manifest["rows"]:
        raise ValueError("Manifest total mismatch")
    return {
        "splits": report,
        "feature_version": FEATURE_VERSION,
        "behavior_available": manifest["behavior_available"],
    }


def train_model(
    prepared: Path, output: Path, feature_set="profile", seed=42, min_precision=0.9
):
    """Called only by `train`. Fit train; select threshold on val; never read test."""
    from xgboost import XGBClassifier

    if feature_set not in ("profile", "portable"):
        raise ValueError("Unknown feature set")
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0,1]")
    prepared, output = Path(prepared), Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite model")
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest["feature_version"] != FEATURE_VERSION:
        raise ValueError("Feature version mismatch")
    if feature_set == "portable" and not manifest["behavior_available"]:
        raise ValueError(
            "Portable model needs tweet data; profiles alone are insufficient"
        )
    names = PROFILE_FEATURES if feature_set == "profile" else PORTABLE_FEATURES
    names = names + [f"{name}__missing" for name in names]
    train = pq.read_table(prepared / "train.parquet").to_pylist()
    val = pq.read_table(prepared / "val.parquet").to_pylist()
    for rows, split in ((train, "train"), (val, "val")):
        if (
            not rows
            or {r["label"] for r in rows} != {0, 1}
            or any(r["split"] != split for r in rows)
        ):
            raise ValueError("Each training/validation split requires both classes")
        ids = [r["external_user_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate identity")
    if {r["external_user_id"] for r in train} & {r["external_user_id"] for r in val}:
        raise ValueError("Training/validation leakage")
    observed = [
        n
        for n in names
        if not n.endswith("__missing") and any(r.get(n) is not None for r in train)
    ]
    if not observed:
        raise ValueError("No observed training features")
    pipe = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4,
        ),
    )
    y = [r["label"] for r in train]
    pipe.fit(matrix(train, names), y)
    probabilities = pipe.predict_proba(matrix(val, names))[:, 1]
    yval = np.array([r["label"] for r in val])
    candidates = []
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = probabilities >= threshold
        precision = precision_score(yval, pred, zero_division=0)
        recall = recall_score(yval, pred, zero_division=0)
        if precision >= min_precision:
            candidates.append((recall, float(threshold)))
    threshold = max(candidates)[1] if candidates else 1.0
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = {
        "model": pipe,
        "features": names,
        "feature_version": FEATURE_VERSION,
        "feature_set": feature_set,
        "model_version": version,
        "threshold": threshold,
        "seed": seed,
        "calibrated": False,
        "required_observed_features": observed,
        "minimum_observed_ratio": 0.5,
        "validation_metrics": metrics(yval, probabilities, threshold),
        "min_precision_target_met": bool(candidates),
        "training_rows": len(train),
        "validation_rows": len(val),
        "prepared_manifest_sha256": hashlib.sha256(
            (prepared / "manifest.json").read_bytes()
        ).hexdigest(),
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scikit-learn", "joblib")
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    output.with_suffix(".metrics.json").write_text(
        json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=2) + "\n"
    )
    return bundle["validation_metrics"]


def evaluate_model(prepared: Path, model_path: Path):
    service = ModelService(model_path)
    if service.bundle is None:
        raise ModelUnavailable(service.reason)
    rows = pq.read_table(Path(prepared) / "test.parquet").to_pylist()
    if not rows or any(r["split"] != "test" for r in rows):
        raise ValueError("Invalid test split")
    return metrics(
        [r["label"] for r in rows], service.predict(rows), service.bundle["threshold"]
    )
