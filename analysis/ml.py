"""Account-level models: supervised booster + Isolation Forest + explanations."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def make_classifier(kind="auto", seed=42):
    """XGBoost -> CatBoost -> sklearn GradientBoosting fallback."""
    if kind in ("auto", "xgboost"):
        try:
            from xgboost import XGBClassifier

            return "xgboost", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                            subsample=0.9, colsample_bytree=0.8, eval_metric="logloss",
                                            random_state=seed)
        except Exception:  # optional native dependency may be installed but unloadable
            if kind == "xgboost":
                raise
    if kind in ("auto", "catboost"):
        try:
            from catboost import CatBoostClassifier

            return "catboost", CatBoostClassifier(iterations=400, depth=5, verbose=False, random_seed=seed)
        except Exception:  # optional native dependency may be installed but unloadable
            if kind == "catboost":
                raise
    return "sklearn-gb", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)


def to_matrix(rows, feature_names):
    return np.array([[float(r.get(f, np.nan)) if r.get(f) is not None else np.nan for f in feature_names] for r in rows], dtype=float)


def evaluate(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4) if len(set(y_true)) > 1 else None,
        "n_test": int(len(y_true)),
    }


def train(rows, labels, feature_names, kind="auto", seed=42, test_size=0.3):
    """Train with an account-level stratified split. Returns (bundle, metrics)."""
    X = to_matrix(rows, feature_names)
    y = np.array(labels, dtype=int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    name, clf = make_classifier(kind, seed)
    pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), clf)
    pipe.fit(Xtr, ytr)
    metrics = evaluate(yte, pipe.predict_proba(Xte)[:, 1])
    # refit on all data for production use
    pipe.fit(X, y)
    return {"model": pipe, "features": feature_names, "kind": name}, metrics


def predict(bundle, rows):
    X = to_matrix(rows, bundle["features"])
    return bundle["model"].predict_proba(X)[:, 1]


def anomaly_scores(rows, feature_names, seed=42):
    """Isolation Forest, scaled to [0,1] where 1 = most anomalous."""
    X = to_matrix(rows, feature_names)
    if len(X) < 10:
        return np.zeros(len(X))
    X = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit(X)
    s = -iso.score_samples(X)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def heuristic_proba(row):
    """Fallback score when no labelled data / model is available."""
    z = 0.0
    z += 1.5 * min(row.get("posts_per_day", 0) / 30, 1)
    cv = row.get("interval_cv")
    if cv is not None and not np.isnan(cv):
        z += 1.2 * max(0, 1 - cv)
    z += 1.0 * (row.get("duplicate_ratio") or 0)
    z += 1.0 * (row.get("url_ratio") or 0)
    z += 1.5 * (row.get("sync_score") or 0)
    z += 0.8 * (row.get("night_activity_ratio") or 0)
    return float(1 / (1 + np.exp(-(z - 2.5) * 1.5)))


def explain(row, population, feature_names, top=5):
    """Robust z-score vs population median -> most deviating features."""
    X = to_matrix(population, feature_names)
    med = np.nanmedian(X, axis=0)
    q75, q25 = np.nanpercentile(X, 75, axis=0), np.nanpercentile(X, 25, axis=0)
    iqr = np.where((q75 - q25) > 1e-9, q75 - q25, np.nanstd(X, axis=0) + 1e-9)
    x = to_matrix([row], feature_names)[0]
    z = (x - med) / iqr
    items = []
    for f, v, zz, m in zip(feature_names, x, z, med):
        if np.isnan(zz):
            continue
        items.append({"feature": f, "value": round(float(v), 4), "median": round(float(m), 4), "z": round(float(zz), 2)})
    items.sort(key=lambda d: -abs(d["z"]))
    return items[:top]


def save(bundle, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load(path: Path):
    return joblib.load(path) if path.exists() else None
