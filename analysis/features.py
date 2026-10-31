"""Behavioural & content features per account.

Input: ``posts`` – list of dicts with keys
    account_id, ts (unix seconds), text, is_reply, is_repost, reply_to,
    urls, hashtags, mentions
Output: {account_id: {feature: value}}

Features with no data are returned as NaN and tracked in ``<name>__missing``
mask — model code imputes them (see slide 05: "отсутствующие данные отмечаются маской").
"""
from collections import defaultdict
import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import clean_text

BEHAVIOR_FEATURES = [
    "posts_per_day",
    "avg_post_interval",
    "interval_cv",            # регулярность (low = robotic)
    "night_activity_ratio",
    "hour_entropy",           # активность по времени суток
    "max_active_streak_h",    # продолжительность непрерывной активности
    "repost_ratio",
    "reply_ratio",
    "url_ratio",
    "hashtag_ratio",
    "duplicate_ratio",        # повторяемость контента
    "self_text_similarity",   # схожесть собственных текстов
    "text_diversity",         # разнообразие
    "interaction_rate",       # упоминания/ответы на пост
    "burstiness",
    "max_posts_per_hour",     # резкие всплески
]

NETWORK_FEATURES = [
    "degree",
    "in_degree",
    "out_degree",
    "clustering",
    "avg_common_neighbors",
    "reciprocity",
    "pagerank",
    "repeat_interaction_ratio",
    "coord_degree",           # число рёбер координации
    "coord_strength",         # сумма весов W
    "sync_score",             # синхронность с другими
    "community_size",
    "community_density",
    "edge_stability",         # устойчивость связей во времени
]

ALL_FEATURES = BEHAVIOR_FEATURES + NETWORK_FEATURES

NIGHT_HOURS = {0, 1, 2, 3, 4, 5}  # local time
TZ_OFFSET_H = 5  # Asia/Almaty UTC+5


def _entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    ps = [c / total for c in counts if c]
    return -sum(p * math.log2(p) for p in ps) / math.log2(24)


def _streak_hours(ts_sorted, gap=3600):
    """Longest run where consecutive posts are < gap apart (hours)."""
    if len(ts_sorted) < 2:
        return 0.0
    best = cur_start = ts_sorted[0]
    best = 0.0
    prev = ts_sorted[0]
    for t in ts_sorted[1:]:
        if t - prev > gap:
            cur_start = t
        best = max(best, t - cur_start)
        prev = t
    return best / 3600


def build_tfidf(posts):
    texts = [clean_text(p["text"]) for p in posts]
    if not any(texts):
        return None, None
    vec = TfidfVectorizer(min_df=1, max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    try:
        X = vec.fit_transform(texts)
    except ValueError:  # empty vocabulary
        return None, None
    return vec, X


def behavior_features(posts, X=None):
    by_acc = defaultdict(list)
    for i, p in enumerate(posts):
        by_acc[p["account_id"]].append(i)

    out = {}
    for acc, idx in by_acc.items():
        ps = [posts[i] for i in idx]
        n = len(ps)
        ts = sorted(p["ts"] for p in ps)
        span_days = max((ts[-1] - ts[0]) / 86400, 1.0)
        intervals = np.diff(ts) if n > 1 else np.array([])
        f = {}
        f["posts_per_day"] = n / span_days
        if len(intervals):
            mu, sd = float(intervals.mean()), float(intervals.std())
            f["avg_post_interval"] = mu / 60
            f["interval_cv"] = sd / mu if mu > 0 else 0.0
            f["burstiness"] = (sd - mu) / (sd + mu) if (sd + mu) > 0 else 0.0
        else:
            f["avg_post_interval"] = float("nan")
            f["interval_cv"] = float("nan")
            f["burstiness"] = float("nan")
        hours = [int(((t / 3600) + TZ_OFFSET_H) % 24) for t in ts]
        f["night_activity_ratio"] = sum(h in NIGHT_HOURS for h in hours) / n
        f["hour_entropy"] = _entropy([hours.count(h) for h in range(24)])
        f["max_active_streak_h"] = _streak_hours(ts)
        f["repost_ratio"] = sum(bool(p.get("is_repost")) for p in ps) / n
        f["reply_ratio"] = sum(bool(p.get("is_reply")) for p in ps) / n
        f["url_ratio"] = sum(bool(p.get("urls")) for p in ps) / n
        f["hashtag_ratio"] = sum(bool(p.get("hashtags")) for p in ps) / n
        norm = [clean_text(p["text"]) for p in ps]
        f["duplicate_ratio"] = 1 - len(set(norm)) / n
        tokens = " ".join(norm).split()
        f["text_diversity"] = len(set(tokens)) / len(tokens) if tokens else float("nan")
        f["interaction_rate"] = sum(len(p.get("mentions") or []) + bool(p.get("reply_to")) for p in ps) / n
        buckets = defaultdict(int)
        for t in ts:
            buckets[int(t // 3600)] += 1
        f["max_posts_per_hour"] = max(buckets.values())
        if X is not None and n > 1:
            sub = X[idx[:60]]
            sim = (sub @ sub.T).toarray()
            m = sim.shape[0]
            f["self_text_similarity"] = float((sim.sum() - np.trace(sim)) / (m * (m - 1)))
        else:
            f["self_text_similarity"] = float("nan")
        out[acc] = f
    return out
