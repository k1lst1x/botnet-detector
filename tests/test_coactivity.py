import numpy as np

from engine.coactivity import (community_scores, communities, coordination_score, knn_graph,
                               post_objects, similarity, text_key)


def test_text_key_ignores_links_mentions_digits_case():
    a = text_key("Buy NOW at http://t.co/abc @bob only 99 dollars today!!")
    b = text_key("buy now at https://t.co/xyz @alice only 15 dollars today!!")
    assert a is not None and a == b
    assert text_key("short") is None


def test_post_objects_types_and_retweet_text_skipped():
    objs = post_objects(retweet_of_status="1", retweet_of_user="bob", mentions=["bob", "ann"],
                        hashtags=["vote"], text="RT @bob: a long enough retweeted message here")
    assert objs == ["rt:1", "rtu:bob", "m:ann", "h:vote"]


def _campaign_world():
    rng = np.random.default_rng(0)
    world = {f"h{i}": [f"h:topic{rng.integers(0, 400)}" for _ in range(20)] for i in range(60)}
    for i in range(8):  # coordinated: share a small pool of retweeted posts
        world[f"b{i}"] = [f"rt:c{j}" for j in rng.integers(0, 5, 15)] + ["h:topic1"]
    return world


def test_campaign_members_have_highest_coordination_and_one_community():
    sim = similarity(_campaign_world())
    score = dict(zip(sim.accounts, coordination_score(sim, k=5)))
    bots = [a for a in sim.accounts if a.startswith("b")]
    humans = [a for a in sim.accounts if a.startswith("h")]
    assert min(score[b] for b in bots) > max(score[h] for h in humans)
    g = knn_graph(sim, k=5, min_similarity=0.2)
    comms = communities(g)
    assert set(bots) in comms
    scored = community_scores(g, comms)
    top = max(scored, key=lambda c: c["score"])
    assert top["size"] == 8 and top["density"] > 0.7  # k=5 neighbours of 7 possible


def test_background_objects_are_dropped():
    world = {f"a{i}": ["h:everyone", f"h:own{i}"] for i in range(10)}
    sim = similarity(world, max_account_share=0.5)
    assert sim.objects_kept == 0 and sim.matrix.nnz == 0
