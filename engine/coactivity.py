"""Group level: co-activity similarity network and coordinated communities.

Accounts are described by the objects they act on (retweeted posts and authors,
replied posts, mentions, hashtags, near-duplicate texts). Two accounts are similar
when they share rare objects: TF-IDF weighting over accounts, cosine similarity.
The k-nearest-neighbour graph is split into communities (Louvain), and each
community is scored by how tightly its members are tied.

Similarity is evidence of coordination, not a bot probability: fans of one event
also share objects. Community scores are therefore combined with account scores
downstream and reviewed manually.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import sparse

_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")
_RT = re.compile(r"^rt\s+@\w+:?\s*")
_DIGITS = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")
TEXT_MIN_CHARS = 20
OBJECT_KINDS = ("rt", "rtu", "re", "m", "h", "txt")


def text_key(text: str | None) -> str | None:
    """Hash of a tweet text with links, mentions, digits and case removed."""
    if not text:
        return None
    t = _RT.sub("", text.casefold())
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _DIGITS.sub("0", t)
    t = _SPACE.sub(" ", t).strip()
    if len(t) < TEXT_MIN_CHARS:
        return None
    return hashlib.blake2b(t.encode(), digest_size=8).hexdigest()


def post_objects(*, retweet_of_status=None, retweet_of_user=None, reply_to_status=None,
                 mentions=(), hashtags=(), text=None, kinds=OBJECT_KINDS) -> list[str]:
    """Typed object tokens of one post, e.g. ``rt:123`` or ``h:election``."""
    out = []
    if "rt" in kinds and retweet_of_status:
        out.append(f"rt:{retweet_of_status}")
    if "rtu" in kinds and retweet_of_user:
        out.append(f"rtu:{retweet_of_user}")
    if "re" in kinds and reply_to_status:
        out.append(f"re:{reply_to_status}")
    if "m" in kinds:
        out.extend(f"m:{m}" for m in (() if mentions is None else mentions)
                   if m != retweet_of_user)
    if "h" in kinds:
        out.extend(f"h:{h}" for h in (() if hashtags is None else hashtags))
    if "txt" in kinds and not retweet_of_status and not retweet_of_user:
        key = text_key(text)
        if key:
            out.append(f"txt:{key}")
    return out


@dataclass
class Similarity:
    accounts: list[str]
    matrix: sparse.csr_matrix  # cosine similarity, zero diagonal
    objects_kept: int


def similarity(objects_by_account: dict[str, list[str]], min_accounts: int = 2,
               max_account_share: float = 0.5) -> Similarity:
    """TF-IDF cosine similarity of accounts over shared objects.

    Objects used by fewer than ``min_accounts`` accounts carry no link; objects
    used by more than ``max_account_share`` of all accounts are background.
    """
    accounts = sorted(objects_by_account)
    n = len(accounts)
    counts = defaultdict(lambda: defaultdict(int))
    df = defaultdict(int)
    for i, a in enumerate(accounts):
        seen = set()
        for o in objects_by_account[a]:
            counts[i][o] += 1
            if o not in seen:
                df[o] += 1
                seen.add(o)
    vocab = {o: j for j, o in enumerate(
        o for o, d in df.items() if min_accounts <= d <= max(min_accounts, max_account_share * n))}
    rows, cols, vals = [], [], []
    for i, objs in counts.items():
        for o, c in objs.items():
            j = vocab.get(o)
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append(np.log1p(c) * np.log(n / df[o]))
    a = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float64)
    norms = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    a = sparse.diags(1 / norms) @ a
    s = (a @ a.T).tocsr()
    s.setdiag(0)
    s.eliminate_zeros()
    return Similarity(accounts, s, len(vocab))


def coordination_score(sim: Similarity, k: int = 10) -> np.ndarray:
    """Mean similarity to the k most similar other accounts (unsupervised)."""
    out = np.zeros(len(sim.accounts))
    for i in range(len(sim.accounts)):
        row = sim.matrix.getrow(i).data
        if row.size:
            top = np.sort(row)[-k:]
            out[i] = top.sum() / k
    return out


def knn_graph(sim: Similarity, k: int = 10, min_similarity: float = 0.2) -> nx.Graph:
    """Undirected graph keeping, for every account, its k strongest links above a floor."""
    g = nx.Graph()
    g.add_nodes_from(sim.accounts)
    m = sim.matrix
    for i in range(m.shape[0]):
        start, end = m.indptr[i], m.indptr[i + 1]
        idx, val = m.indices[start:end], m.data[start:end]
        keep = val >= min_similarity
        idx, val = idx[keep], val[keep]
        if idx.size > k:
            top = np.argpartition(-val, k)[:k]
            idx, val = idx[top], val[top]
        for j, w in zip(idx, val):
            g.add_edge(sim.accounts[i], sim.accounts[j], weight=float(w))
    return g


def communities(g: nx.Graph, seed: int = 42, resolution: float = 1.0,
                method: str = "louvain") -> list[set[str]]:
    """Communities of the non-isolated part of the graph, largest first.

    ``louvain`` (networkx), ``leiden`` (leidenalg, RB-configuration quality;
    falls back to Louvain when leidenalg is not installed) or ``components``
    (connected components of the k-NN graph).
    """
    core = g.subgraph(n for n, d in g.degree() if d > 0)
    if method == "components":
        found = [set(c) for c in nx.connected_components(core)]
    elif method == "leiden" and _leiden_available():
        import igraph as ig
        import leidenalg

        nodes = list(core.nodes())
        index = {n: i for i, n in enumerate(nodes)}
        edges = [(index[a], index[b]) for a, b in core.edges()]
        graph = ig.Graph(n=len(nodes), edges=edges)
        graph.es["weight"] = [core[a][b]["weight"] for a, b in core.edges()]
        part = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition, weights="weight",
            resolution_parameter=resolution, seed=seed)
        found = [{nodes[i] for i in members} for members in part]
    else:
        found = nx.community.louvain_communities(core, weight="weight", seed=seed,
                                                 resolution=resolution)
    return sorted(found, key=len, reverse=True)


def _leiden_available() -> bool:
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError:
        return False
    return True


def community_scores(g: nx.Graph, groups: list[set[str]], min_size: int = 3) -> list[dict]:
    """Size, weighted density and mean tie strength of every community."""
    out = []
    for cid, members in enumerate(groups):
        if len(members) < min_size:
            continue
        sub = g.subgraph(members)
        weights = [d["weight"] for _, _, d in sub.edges(data=True)]
        possible = len(members) * (len(members) - 1) / 2
        out.append({
            "community": cid,
            "size": len(members),
            "edges": len(weights),
            "density": len(weights) / possible if possible else 0.0,
            "mean_weight": float(np.mean(weights)) if weights else 0.0,
            # tie strength x cohesion; high for many strongly linked members
            "score": float(np.mean(weights)) * (len(weights) / possible) if weights else 0.0,
        })
    return out


# --- system entry point -------------------------------------------------------------

_HASHTAG = re.compile(r"#(\w+)")
GROUP_FLAG_MIN_SIZE = 5
GROUP_FLAG_MIN_WEIGHT = 0.2


def dataset_objects(data) -> dict[str, list[str]]:
    """Object tokens of every account in a Dataset: targets, URLs, hashtags, texts."""
    from urllib.parse import urlsplit

    out: dict[str, list[str]] = {f"{a.platform}:{a.external_user_id}": [] for a in data.accounts}
    for post in data.posts:
        objs = out.setdefault(f"{post.platform}:{post.author_id}", [])
        objs.extend(f"m:{t.casefold()}" for t in post.targets or ())
        urls = post.urls if post.urls is not None else _URL.findall(post.text or "")
        for url in urls:
            try:
                parts = urlsplit(url)
            except ValueError:
                continue
            if parts.netloc:
                objs.append(f"u:{parts.netloc.lower()}{parts.path.rstrip('/')}")
        text = post.text or ""
        objs.extend(f"h:{h.casefold()}" for h in _HASHTAG.findall(text))
        if not post.is_repost:
            key = text_key(text)
            if key:
                objs.append(f"txt:{key}")
    return out


def group_summary(data, k: int = 10,
                  min_similarity: float = 0.2, method: str = "louvain",
                  resolution: float = 0.5) -> dict:
    """Coordinated groups of a Dataset for the review queue.

    A group is flagged when it has >= GROUP_FLAG_MIN_SIZE members and a mean tie
    strength >= GROUP_FLAG_MIN_WEIGHT. The flag raises review priority; it does not
    change the account score (see docs/reports/2026-11-24-cresci-account-and-group.md).
    Louvain with resolution 0.5 was selected on synthetic campaigns only
    (scripts.cresci_partition_tuning).
    """
    objects = dataset_objects(data)
    sim = similarity(objects, max_account_share=0.8)
    coord = coordination_score(sim, k)
    g = knn_graph(sim, k, min_similarity)
    found = communities(g, resolution=resolution, method=method)
    scored = community_scores(g, found, min_size=2)
    groups, account_group = [], {}
    for rank, c in enumerate(sorted(scored, key=lambda c: (-c["size"], -c["mean_weight"]))):
        members = sorted(found[c["community"]])
        shared = Counter(o for m in members for o in set(objects[m]))
        group_id = f"g{rank + 1}"
        groups.append({
            "group_id": group_id,
            "size": c["size"],
            "density": round(c["density"], 4),
            "mean_weight": round(c["mean_weight"], 4),
            "flagged": c["size"] >= GROUP_FLAG_MIN_SIZE and c["mean_weight"] >= GROUP_FLAG_MIN_WEIGHT,
            "members": members,
            "shared_objects": [{"object": o, "accounts": n}
                               for o, n in shared.most_common(8) if n >= 2],
        })
        for m in members:
            account_group[m] = group_id
    return {
        "method": method if method != "leiden" or _leiden_available() else "louvain",
        "k": k,
        "min_similarity": min_similarity,
        "objects_kept": sim.objects_kept,
        "groups": groups,
        "account_group": account_group,
        "coordination_score": {a: round(float(s), 4) for a, s in zip(sim.accounts, coord)},
        "flag_rule": {"min_size": GROUP_FLAG_MIN_SIZE, "min_mean_weight": GROUP_FLAG_MIN_WEIGHT},
        "note": "coordination_is_not_bot_probability",
    }
