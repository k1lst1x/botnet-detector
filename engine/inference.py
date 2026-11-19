"""Shared CLI/worker inference entry point, without ORM or training imports."""

from collections import defaultdict

from .features import FEATURE_VERSION, extract_features


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
                "decision": decision,
                "model_version": status["model_version"],
                "feature_version": FEATURE_VERSION,
                "coverage": coverage,
                "warnings": warnings,
                "features": row,
                "explanation": explanation,
                "observed_window": data.observation.model_dump(mode="json"),
            }
        )
    from .coordination import coordination_summary

    return {
        "results": results,
        "model": model.status(),
        "coordination": coordination_summary(data),
    }
