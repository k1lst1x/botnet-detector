"""Shared CLI/worker inference entry point, without ORM or training imports."""

from collections import defaultdict

from .features import FEATURE_VERSION, extract_features, extract_profile_features


RECENT_POSTS = 3
RECENT_TEXT_CHARS = 280


def account_context(account, posts):
    """Human-readable profile and activity shown next to the score; not model input."""
    ordered = sorted(posts, key=lambda p: p.created_at)
    return {
        "profile": {
            "display_name": account.display_name,
            "biography": account.biography,
            "followers": account.followers,
            "following": account.following,
            "is_verified": account.is_verified,
        },
        "activity": {
            "first": ordered[0].created_at.isoformat() if ordered else None,
            "last": ordered[-1].created_at.isoformat() if ordered else None,
            "recent": [{"created_at": p.created_at.isoformat(), "text": p.text[:RECENT_TEXT_CHARS]}
                       for p in reversed(ordered) if p.text][:RECENT_POSTS],
        },
    }


def review_decision(score, threshold):
    return "priority_review" if score >= threshold else "standard_review"


def review_queue(decision, in_flagged_group):
    """Account flag OR coordinated-group flag; averaging the two scores hurt recall."""
    if decision == "priority_review" or in_flagged_group:
        return "priority"
    if decision == "standard_review":
        return "standard"
    return "not_scored"


def analyze_dataset(data, model):
    grouped = defaultdict(list)
    for post in data.posts:
        grouped[(post.platform, post.author_id)].append(post)
    results = []
    for account in data.accounts:
        posts = grouped[account.key]
        status = model.status()
        selected = posts
        observation = data.observation
        policy = model.bundle.get("sampling_policy") if status["status"] == "ready" else None
        if policy:
            from .sampling import sample_posts
            selected = sample_posts(posts, policy["max_posts"], policy["seed"])
            if len(selected) < len(posts):
                observation = observation.model_copy(update={"complete": False, "truncated": True})
        row = extract_features(account, selected, observation)
        row.update(extract_profile_features(account))
        status = model.status()
        warnings = []
        score = None
        decision = status["status"]
        if not data.observation.complete:
            warnings.append("partial_observation")
        if account.identity_source == "username_fallback":
            warnings.append("unstable_author_identity")
        explanation = []
        if status["status"] == "ready":
            required = model.bundle["required_observed_features"]
            observed = [n for n in required if row.get(n) is not None]
            if len(posts) < model.bundle.get("minimum_posts", 0) or len(observed) / max(1, len(required)) < model.bundle.get(
                "minimum_observed_ratio", 0.5
            ):
                decision = "insufficient_data"
            else:
                score = model.predict([row])[0]
                decision = ("needs_review" if model.bundle.get("decision_policy") == "review_only"
                            else "bot" if score >= model.bundle["threshold"] else "human")
                if model.bundle.get("decision_policy") == "review_queue":
                    decision = review_decision(score, model.bundle["threshold"])
                    warnings.append("uncalibrated_score")
                warnings.append(
                    "threads_quality_unverified"
                    if account.platform == "threads"
                    else "uncalibrated_score"
                )
                # Contributions from the actual booster, not population deviation.
                from xgboost import DMatrix
                from .model import matrix

                pipe = model.bundle["model"]
                X = pipe.steps[0][1].transform(matrix([row], model.bundle["features"]))
                contrib = (
                    pipe.steps[-1][1]
                    .get_booster()
                    .predict(DMatrix(X), pred_contribs=True)[0][:-1]
                )
                explanation = [
                    {
                        "feature": name,
                        "contribution_log_odds": float(value),
                        "value": row[name],
                    }
                    for name, value in sorted(
                        zip(model.bundle["features"], contrib),
                        key=lambda pair: abs(pair[1]),
                        reverse=True,
                    )[:5]
                ]
        coverage = {
            "posts": len(posts),
            "sampled_posts": len(selected),
            "complete": observation.complete,
            "truncated": observation.truncated,
            "available_features": sum(
                v is not None for n, v in row.items() if not n.endswith("__missing")
            ),
        }
        results.append(
            {
                "platform": account.platform,
                "external_user_id": account.external_user_id,
                "username": account.username,
                "score": score,
                "threshold": model.bundle["threshold"] if score is not None else None,
                **account_context(account, posts),
                "decision": decision,
                "model_version": status["model_version"],
                "feature_version": (model.bundle or {}).get("feature_version", FEATURE_VERSION),
                "coverage": coverage,
                "warnings": warnings,
                "features": row,
                "explanation": explanation,
                "observed_window": data.observation.model_dump(mode="json"),
            }
        )
    from .coactivity import group_summary
    from .coordination import coordination_summary

    groups = group_summary(data)
    flagged = {g["group_id"]: g for g in groups["groups"] if g["flagged"]}
    for row in results:
        key = f"{row['platform']}:{row['external_user_id']}"
        group_id = groups["account_group"].get(key)
        row["group"] = None if group_id is None else {
            "group_id": group_id,
            "flagged": group_id in flagged,
            "coordination_score": groups["coordination_score"].get(key),
        }
        row["queue"] = review_queue(row["decision"], group_id in flagged)
        if group_id in flagged:
            row["warnings"].append("coordinated_group")
    return {
        "results": results,
        "model": model.status(),
        "groups": {k: v for k, v in groups.items() if k not in ("account_group", "coordination_score")},
        "coordination": coordination_summary(data),
    }
