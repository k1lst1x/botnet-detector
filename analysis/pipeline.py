"""End-to-end analysis (slide 07): features -> ML P(bot) -> weighted graph ->
Louvain/DBSCAN -> group scoring -> explanations. Persists results via Django ORM."""
from collections import Counter
from dataclasses import asdict

import numpy as np
from django.conf import settings
from django.db import transaction
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from core.models import Account, AnalysisRun, Community, Edge, Post

from . import ml
from .features import ALL_FEATURES, BEHAVIOR_FEATURES, behavior_features, build_tfidf
from .graph import GraphParams, community_stats, coordination_edges, detect_communities, interaction_graph, network_features

MODEL_PATH = lambda: settings.MODEL_DIR / "account_model.joblib"  # noqa: E731


def load_posts():
    rows = []
    for p in Post.objects.all().values("account_id", "created_at", "text", "is_reply", "is_repost",
                                         "reply_to", "urls", "hashtags", "mentions"):
        p["ts"] = p.pop("created_at").timestamp()
        p["reply_to"] = (p["reply_to"] or "").lower()
        rows.append(p)
    rows.sort(key=lambda r: r["ts"])
    return rows


def compute_features(params: GraphParams):
    accounts = list(Account.objects.all())
    ids = [a.id for a in accounts]
    handle_to_id = {a.handle.lower(): a.id for a in accounts}
    posts = load_posts()
    _, X = build_tfidf(posts) if posts else (None, None)
    beh = behavior_features(posts, X)
    G, times = interaction_graph(posts, handle_to_id)
    edges = coordination_edges(posts, X, params)
    net, C = network_features(ids, G, times, edges, posts)
    feats = {}
    for a in accounts:
        f = {k: float("nan") for k in BEHAVIOR_FEATURES}
        f.update(beh.get(a.id, {}))
        f.update(net.get(a.id, {}))
        f["followers"] = a.followers
        f["following"] = a.following
        f["ff_ratio"] = a.followers / (a.following + 1)
        feats[a.id] = f
    return accounts, feats, edges, C, posts


def train_and_evaluate(accounts, feats, kind="auto", seed=42):
    """Experiments A (behaviour) vs B (behaviour + network). Saves model B."""
    lab = [a for a in accounts if a.label in ("bot", "human")]
    y = [1 if a.label == "bot" else 0 for a in lab]
    if len(lab) < 20 or min(Counter(y).values()) < 5 or len(set(y)) < 2:
        return None, {}
    rows = [feats[a.id] for a in lab]
    base = BEHAVIOR_FEATURES + ["followers", "following", "ff_ratio"]
    _, m_a = ml.train(rows, y, base, kind, seed)
    bundle, m_b = ml.train(rows, y, base + [f for f in ALL_FEATURES if f not in BEHAVIOR_FEATURES], kind, seed)
    ml.save(bundle, MODEL_PATH())
    return bundle, {"A_baseline": m_a, "B_extended": m_b, "model": bundle["kind"]}


def run_analysis(params: GraphParams | None = None, retrain=False, model_kind="auto"):
    params = params or GraphParams()
    accounts, feats, edges, C, posts = compute_features(params)
    if not accounts:
        raise ValueError("Нет данных: импортируйте CSV или сгенерируйте синтетику.")
    metrics = {}
    bundle = None if retrain else ml.load(MODEL_PATH())
    if bundle is None:
        bundle, metrics = train_and_evaluate(accounts, feats, model_kind, params.seed)
    rows = [feats[a.id] for a in accounts]
    proba = ml.predict(bundle, rows) if bundle else np.array([ml.heuristic_proba(r) for r in rows])
    anomaly = ml.anomaly_scores(rows, BEHAVIOR_FEATURES, params.seed)
    louvain, dbscan = detect_communities(C, params)
    by_id = {a.id: a for a in accounts}
    p_by = {a.id: float(p) for a, p in zip(accounts, proba)}

    with transaction.atomic():
        run = AnalysisRun.objects.create(params=asdict(params), model_kind=bundle["kind"] if bundle else "heuristic")
        Account.objects.update(community=None, dbscan_cluster=None)
        comm_of, comm_score = {}, {}
        for label, members in enumerate(louvain):
            density, mean_w = community_stats(members, C)
            mean_p = float(np.mean([p_by[m] for m in members]))
            score = round(0.45 * min(mean_w / max(params.threshold * 2, 1e-9), 1) + 0.25 * density + 0.30 * mean_p, 4)
            url_c, tag_c = Counter(), Counter()
            mset = set(members)
            for p in posts:
                if p["account_id"] in mset:
                    url_c.update(set(p["urls"] or []))
                    tag_c.update(set(p["hashtags"] or []))
            comps = [e[3] for e in edges if e[0] in mset and e[1] in mset]
            top_signals = {k: round(float(np.mean([c[k] for c in comps])), 3) for k in comps[0]} if comps else {}
            com = Community.objects.create(run=run, method="louvain", label=label, size=len(members), density=round(density, 4),
                                           mean_weight=round(mean_w, 4), mean_bot_proba=round(mean_p, 4), coordination_score=score,
                                           top_signals=top_signals, shared_urls=[u for u, _ in url_c.most_common(5)],
                                           shared_hashtags=[t for t, _ in tag_c.most_common(5)])
            for m in members:
                comm_of[m] = com
                comm_score[m] = score

        for i, a in enumerate(accounts):
            f = feats[a.id]
            com = comm_of.get(a.id)
            f["community_size"] = com.size if com else 0
            f["community_density"] = com.density if com else 0.0
            a.bot_proba = round(p_by[a.id], 4)
            a.anomaly_score = round(float(anomaly[i]), 4)
            a.risk = round(0.5 * a.bot_proba + 0.2 * a.anomaly_score + 0.3 * comm_score.get(a.id, 0.0), 4)
            a.community = com
            a.dbscan_cluster = dbscan.get(a.id)
            a.features = {k: (None if isinstance(v, float) and np.isnan(v) else round(float(v), 5)) for k, v in f.items()}
            a.explanation = ml.explain(f, rows, ALL_FEATURES)
        Account.objects.bulk_update(accounts, ["bot_proba", "anomaly_score", "risk", "community", "dbscan_cluster",
                                               "features", "explanation"], batch_size=500)
        Edge.objects.exclude(run=run).delete()
        Edge.objects.bulk_create([Edge(run=run, a_id=a, b_id=b, weight=w, components=c) for a, b, w, c in edges], batch_size=1000)

        # group-level evaluation (experiment C) when ground truth exists
        truth = [a for a in accounts if a.true_group]
        if truth:
            gt = [a.true_group or f"solo_{a.id}" for a in accounts]
            pred = [f"c{comm_of[a.id].label}" if a.id in comm_of else f"solo_{a.id}" for a in accounts]
            db = [f"d{dbscan[a.id]}" if dbscan.get(a.id, -1) >= 0 else f"solo_{a.id}" for a in accounts]
            flagged = {a.id for a in accounts if a.id in comm_of and comm_of[a.id].coordination_score >= 0.5}
            true_coord = {a.id for a in truth}
            tp = len(flagged & true_coord)
            metrics["C_groups"] = {
                "louvain_ARI": round(adjusted_rand_score(gt, pred), 4),
                "louvain_NMI": round(normalized_mutual_info_score(gt, pred), 4),
                "dbscan_ARI": round(adjusted_rand_score(gt, db), 4),
                "dbscan_NMI": round(normalized_mutual_info_score(gt, db), 4),
                "group_precision": round(tp / len(flagged), 4) if flagged else 0.0,
                "group_recall": round(tp / len(true_coord), 4),
            }
        labelled = [a for a in accounts if a.label in ("bot", "human")]
        if labelled and len({a.label for a in labelled}) == 2:
            y = np.array([a.label == "bot" for a in labelled], dtype=int)
            metrics["risk_in_sample"] = ml.evaluate(y, np.array([a.risk for a in labelled]))
        run.metrics = metrics
        run.n_accounts = len(accounts)
        run.n_edges = len(edges)
        run.n_communities = len(louvain)
        run.save()
    return run
