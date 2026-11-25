import numpy as np
import pytest

from engine.coactivity import (community_scores, communities, coordination_score, knn_graph,
                               post_objects, similarity, system_post_objects, text_key)


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


def test_max_channels_finds_one_shared_kind_hidden_by_other_activity():
    world = {f"a{i}": [f"txt:same{j}" for j in range(3)] + [f"h:own{i % 2}_{j}" for j in range(40)]
             for i in range(4)}
    world |= {f"x{i}": [f"h:bg{i}", "h:bgshared"] for i in range(20)}
    joint = similarity(world, max_account_share=0.9)
    best = similarity(world, max_account_share=0.9, channels="max")
    i, j = joint.accounts.index("a0"), joint.accounts.index("a1")
    assert joint.matrix[i, j] < 0.2 and best.matrix[i, j] > 0.99


def test_min_shared_drops_pairs_with_one_common_object():
    world = {"a": ["h:x"], "b": ["h:x"], "c": ["h:y", "h:z"], "d": ["h:y", "h:z"], "e": ["h:q"]}
    sim = similarity(world)
    assert knn_graph(sim, min_shared=1).has_edge("a", "b")
    g = knn_graph(sim, min_shared=2)
    assert not g.has_edge("a", "b") and g.has_edge("c", "d")


def test_truncated_links_are_not_objects():
    assert system_post_objects("x", urls=["http://t.co…", "http://shop.example/p/"]) == ["u:shop.example/p"]


def test_missing_strings_from_parquet_become_none(tmp_path):
    pytest.importorskip("pandas")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from scripts.prepare_cresci import frame

    pq.write_table(pa.table({"reply": pa.array(["1", None]), "n": [1, 2]}), tmp_path / "t.parquet")
    assert frame(pq.read_table(tmp_path / "t.parquet"))["reply"].tolist() == ["1", None]
