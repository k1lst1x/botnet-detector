"""Sparse observed coactivity graph; no follower graph or lexical similarity."""

from collections import defaultdict
from itertools import combinations
from urllib.parse import urlunsplit

from .features import URL_PATTERN, parsed_url


def coordination_summary(
    data, window_s=600, max_bucket_accounts=60, max_candidates=100000
):
    windows = defaultdict(set)
    urls = defaultdict(set)
    targets = defaultdict(set)
    for post in data.posts:
        key = f"{post.platform}:{post.author_id}"
        windows[key].add(int(post.created_at.timestamp() // window_s))
        for url in (
            post.urls if post.urls is not None else URL_PATTERN.findall(post.text or "")
        ):
            parts = parsed_url(url)
            if parts is not None:
                urls[key].add(
                    urlunsplit(
                        (
                            parts.scheme.lower(),
                            parts.netloc.lower(),
                            parts.path,
                            parts.query,
                            "",
                        )
                    )
                )
        targets[key].update(post.targets or [])
    candidates = set()
    truncated = False
    for profiles in (windows, urls, targets):
        index = defaultdict(list)
        for account, values in profiles.items():
            for value in values:
                index[value].append(account)
        for members in index.values():
            if not 1 < len(members) <= max_bucket_accounts:
                continue
            for pair in combinations(sorted(members), 2):
                if len(candidates) >= max_candidates:
                    truncated = True
                    break
                candidates.add(pair)
    edges = []

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a or b else 0.0

    for a, b in sorted(candidates):
        shared = len(windows[a] & windows[b])
        # Require repeated coactivity or a shared URL/target, not a single coincident post.
        if shared < 2 and not urls[a] & urls[b] and not targets[a] & targets[b]:
            continue
        edges.append(
            {
                "a": a,
                "b": b,
                "synchrony": shared / max(1, min(len(windows[a]), len(windows[b]))),
                "repeated_windows": shared,
                "shared_url_ratio": jaccard(urls[a], urls[b]),
                "shared_target_ratio": jaccard(targets[a], targets[b]),
            }
        )
    return {
        "edges": edges,
        "window_s": window_s,
        "truncated": truncated,
        "observation_complete": data.observation.complete,
        "note": "coordination_is_not_bot_probability",
    }
