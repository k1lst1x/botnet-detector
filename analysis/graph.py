"""Coordination graph, interaction graph and community detection.

W(A,B) = w1*TextSim + w2*TimeSim + w3*URLSim + w4*HashtagSim + w5*InteractionSim
"""
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

from .text import jaccard

DEFAULT_WEIGHTS = {"text": 0.30, "time": 0.25, "url": 0.20, "hashtag": 0.10, "interaction": 0.15}


@dataclass
class GraphParams:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    threshold: float = 0.30          # min W to create an edge
    window_s: int = 600              # Δt for TimeSim
    text_candidate_min: float = 0.35 # candidate pair if centroid cosine >= this
    max_bucket_size: int = 60
    set_shrink: float = 2.0          # Jaccard damping for small sets        # skip giant buckets (viral URLs / prime time)
    dbscan_eps: float = 0.6
    dbscan_min_samples: int = 3
    louvain_resolution: float = 1.0
    seed: int = 42

    def normalized_weights(self):
        s = sum(max(v, 0) for v in self.weights.values()) or 1.0
        return {k: max(v, 0) / s for k, v in self.weights.items()}


def account_profiles(posts, X, window_s):
    prof = defaultdict(lambda: {"urls": set(), "tags": set(), "targets": set(), "buckets": set(), "rows": []})
    for i, p in enumerate(posts):
        pr = prof[p["account_id"]]
        pr["urls"].update(p.get("urls") or [])
        pr["tags"].update(p.get("hashtags") or [])
        pr["targets"].update(p.get("mentions") or [])
        if p.get("reply_to"):
            pr["targets"].add(p["reply_to"])
        pr["buckets"].add(int(p["ts"] // window_s))
        pr["rows"].append(i)
    ids = list(prof.keys())
    centroids = None
    if X is not None:
        rows = [np.asarray(X[prof[a]["rows"]].mean(axis=0)).ravel() for a in ids]
        centroids = normalize(np.vstack(rows))
    return ids, prof, centroids


def time_sim(ba, bb):
    if not ba or not bb:
        return 0.0
    small, big = (ba, bb) if len(ba) <= len(bb) else (bb, ba)
    hits = sum(1 for b in small if b in big or (b - 1) in big or (b + 1) in big)
    return hits / len(small)


def candidate_pairs(ids, prof, centroids, params):
    pos = {a: i for i, a in enumerate(ids)}
    cands = set()
    for key in ("urls", "tags", "buckets"):
        inv = defaultdict(list)
        for a in ids:
            for v in prof[a][key]:
                inv[v].append(a)
        for members in inv.values():
            if 1 < len(members) <= params.max_bucket_size:
                for a, b in combinations(members, 2):
                    cands.add((a, b) if pos[a] < pos[b] else (b, a))
    if centroids is not None and len(ids) > 1:
        S = centroids @ centroids.T
        ii, jj = np.where(np.triu(S, 1) >= params.text_candidate_min)
        for i, j in zip(ii, jj):
            cands.add((ids[i], ids[j]))
    return cands, pos


def coordination_edges(posts, X, params: GraphParams):
    """Returns list of (a, b, weight, components)."""
    ids, prof, centroids = account_profiles(posts, X, params.window_s)
    if len(ids) < 2:
        return []
    cands, pos = candidate_pairs(ids, prof, centroids, params)
    w = params.normalized_weights()
    edges = []
    for a, b in cands:
        pa, pb = prof[a], prof[b]
        comp = {
            "text": float(max(0.0, centroids[pos[a]] @ centroids[pos[b]])) if centroids is not None else 0.0,
            "time": time_sim(pa["buckets"], pb["buckets"]),
            "url": jaccard(pa["urls"], pb["urls"], params.set_shrink),
            "hashtag": jaccard(pa["tags"], pb["tags"], params.set_shrink),
            "interaction": jaccard(pa["targets"], pb["targets"], params.set_shrink),
        }
        weight = sum(w[k] * comp[k] for k in w)
        if weight >= params.threshold:
            edges.append((a, b, round(weight, 4), {k: round(v, 3) for k, v in comp.items()}))
    return edges


def interaction_graph(posts, handle_to_id):
    """Directed graph of replies / mentions between known accounts."""
    G = nx.DiGraph()
    G.add_nodes_from(handle_to_id.values())
    times = defaultdict(list)
    for p in posts:
        targets = set(p.get("mentions") or [])
        if p.get("reply_to"):
            targets.add(p["reply_to"])
        for h in targets:
            t = handle_to_id.get(h)
            if t is None or t == p["account_id"]:
                continue
            if G.has_edge(p["account_id"], t):
                G[p["account_id"]][t]["count"] += 1
            else:
                G.add_edge(p["account_id"], t, count=1)
            times[(p["account_id"], t)].append(p["ts"])
    return G, times


def network_features(account_ids, G, inter_times, coord_edges, posts):
    """Per-account network features from interaction + coordination graphs."""
    U = G.to_undirected()
    pr = nx.pagerank(G, weight="count") if G.number_of_edges() else {a: 1 / max(len(account_ids), 1) for a in account_ids}
    clust = nx.clustering(U)
    C = nx.Graph()
    C.add_nodes_from(account_ids)
    for a, b, wgt, comp in coord_edges:
        C.add_edge(a, b, weight=wgt, time=comp["time"])

    # edge stability: does the pair co-occur in both halves of the observation period?
    if posts:
        ts = [p["ts"] for p in posts]
        mid = (min(ts) + max(ts)) / 2
    else:
        mid = 0
    halves = defaultdict(lambda: [set(), set()])
    for p in posts:
        halves[p["account_id"]][0 if p["ts"] < mid else 1].add(int(p["ts"] // 600))

    out = {}
    for a in account_ids:
        f = {}
        f["in_degree"] = G.in_degree(a) if a in G else 0
        f["out_degree"] = G.out_degree(a) if a in G else 0
        f["degree"] = U.degree(a) if a in U else 0
        f["clustering"] = clust.get(a, 0.0)
        nbrs = set(U[a]) if a in U else set()
        if nbrs:
            f["avg_common_neighbors"] = float(np.mean([len(nbrs & set(U[n])) for n in nbrs]))
            succ = set(G.successors(a))
            f["reciprocity"] = sum(1 for s in succ if G.has_edge(s, a)) / len(succ) if succ else 0.0
            counts = [d["count"] for _, _, d in G.out_edges(a, data=True)]
            f["repeat_interaction_ratio"] = sum(c > 1 for c in counts) / len(counts) if counts else 0.0
        else:
            f["avg_common_neighbors"] = f["reciprocity"] = f["repeat_interaction_ratio"] = 0.0
        f["pagerank"] = pr.get(a, 0.0) * len(account_ids)
        f["coord_degree"] = C.degree(a)
        f["coord_strength"] = C.degree(a, weight="weight")
        tvals = [d["time"] for _, _, d in C.edges(a, data=True)]
        f["sync_score"] = float(np.mean(tvals)) if tvals else 0.0
        stable = 0
        for n in C[a]:
            h1 = bool(halves[a][0]) and bool(halves[n][0]) and time_sim(halves[a][0], halves[n][0]) > 0.2
            h2 = bool(halves[a][1]) and bool(halves[n][1]) and time_sim(halves[a][1], halves[n][1]) > 0.2
            stable += h1 and h2
        f["edge_stability"] = stable / C.degree(a) if C.degree(a) else 0.0
        out[a] = f
    return out, C


def detect_communities(C: nx.Graph, params: GraphParams, min_size=3):
    """Louvain on the weighted coordination graph + DBSCAN on 1-W distances."""
    H = C.subgraph([n for n in C if C.degree(n) > 0]).copy()
    louvain = []
    if H.number_of_edges():
        comms = nx.community.louvain_communities(H, weight="weight", resolution=params.louvain_resolution, seed=params.seed)
        louvain = [sorted(c) for c in comms if len(c) >= min_size]

    dbscan = {}
    nodes = list(H.nodes)
    if len(nodes) >= params.dbscan_min_samples:
        idx = {n: i for i, n in enumerate(nodes)}
        D = np.ones((len(nodes), len(nodes)))
        np.fill_diagonal(D, 0)
        for a, b, d in H.edges(data=True):
            D[idx[a], idx[b]] = D[idx[b], idx[a]] = 1 - d["weight"]
        labels = DBSCAN(eps=params.dbscan_eps, min_samples=params.dbscan_min_samples, metric="precomputed").fit_predict(D)
        dbscan = {n: int(l) for n, l in zip(nodes, labels)}
    return louvain, dbscan


def community_stats(members, C: nx.Graph):
    S = C.subgraph(members)
    n = len(members)
    density = nx.density(S) if n > 1 else 0.0
    ws = [d["weight"] for _, _, d in S.edges(data=True)]
    return density, (float(np.mean(ws)) if ws else 0.0)
