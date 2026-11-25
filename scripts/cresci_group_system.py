"""Group level re-evaluated in the configuration the system actually runs.

Differences from scripts.cresci_group_experiment / cresci_partition_tuning:
 - objects are engine.coactivity.system_post_objects (targets, URLs, hashtags, texts), the
   background limit is 0.8, as in group_summary;
 - campaigns are injected as POSTS, so the features of the members are recomputed and the
   account model really sees the changed accounts;
 - genuine accounts are split in two halves: the flag rule is selected on the calibration half
   (synthetic campaigns + its false positives), everything is reported on the other half.
Real Cresci campaign labels are never used for selection.

Run:  .venv/bin/python -m scripts.cresci_group_system
"""
from __future__ import annotations

import itertools
import json
import time
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.model_selection import train_test_split

from engine.coactivity import (community_scores, communities, coordination_score, knn_graph,
                               similarity, system_post_objects)
from engine.contracts import Observation, Post
from engine.features import PORTABLE_FEATURES, PROFILE_V2_FEATURES, extract_features, extract_profile_features
from scripts.cresci_account_experiment import MIN_POSTS, PARAMS
from scripts.cresci_features import account
from scripts.cresci_group_experiment import GROUPS, MIN_POSTS as MIN_WINDOW_POSTS, WINDOW, ranking
from scripts.prepare_cresci import OUT, frame

REPORT = Path("media/models/experiments/cresci-group-system-v1")
CACHE = OUT / "window_features.parquet"
SEED, K, BACKGROUND, LAST = 42, 10, 0.8, 200
THRESHOLD = 0.5
COLS = PROFILE_V2_FEATURES + PORTABLE_FEATURES
KINDS = ("copypaste", "link", "mention", "hashtag")
SIZES, INTENSITIES, POOL, SEEDS = (10, 30), (0.02, 0.05, 0.1, 0.2), 20, (0, 1, 2)
FPR_BUDGETS = (0.01, 0.02, 0.05)
CURRENT = {"channels": "joint", "min_similarity": 0.2, "min_shared": 1, "resolution": 0.5, "min_size": 5,
           "min_weight": 0.2, "min_cohesion": 0.0}
GRAPHS = [dict(zip(("channels", "min_similarity", "min_shared", "resolution"), v))
          for v in itertools.product(("joint", "max"), (0.1, 0.2, 0.3, 0.4), (1, 2, 3, 5), (0.5, 1.0))]
RULES = [dict(zip(("min_size", "min_weight", "min_cohesion"), v))
         for v in itertools.product((3, 5, 8), (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), (0.0, 0.3, 0.5, 0.7, 0.9))]


def load():
    cols = ["tweet_id", "user_id", "group", "created_at", "text", "reply_to_status", "reply_to_user",
            "retweet_of_status", "retweet_of_user", "mentions", "urls"]
    t = frame(pq.read_table(OUT / "tweets.parquet", columns=cols, filters=[("group", "in", GROUPS)]))
    t = t[(t.created_at >= WINDOW[0]) & (t.created_at < WINDOW[1])].drop_duplicates("tweet_id")
    n = t.groupby("user_id").size()
    t = t[t.user_id.isin(n[n >= MIN_WINDOW_POSTS].index)].sort_values(["user_id", "created_at"])
    group_of = t.groupby("user_id")["group"].first()
    records, objects = {}, {}
    for r in t.itertuples(index=False):
        targets = list(r.mentions) if r.mentions is not None else []
        if r.reply_to_user:
            targets.append(f"id:{r.reply_to_user}")
        repost = bool(r.retweet_of_status or r.retweet_of_user)
        rec = (r.created_at.to_pydatetime(), r.text, targets,
               list(r.urls) if r.urls is not None else [], repost, bool(r.reply_to_status))
        records.setdefault(r.user_id, []).append(rec)
        objects.setdefault(r.user_id, []).extend(system_post_objects(r.text, targets, rec[3], repost))
    return records, objects, group_of


def features(uid, snapshot, records) -> dict:
    last = sorted(records, key=lambda r: r[0])[-LAST:]
    posts = [Post(platform="twitter", external_post_id=str(i), author_id=uid, created_at=c, text=x,
                  targets=m, urls=u, is_repost=rp, is_reply=rep)
             for i, (c, x, m, u, rp, rep) in enumerate(last)]
    start, end = posts[0].created_at, posts[-1].created_at
    row = extract_features(snapshot, posts, Observation(start=start, end=end, source="cresci2017_window",
                                                        complete=start < end))
    row.update(extract_profile_features(snapshot))
    return {c: row.get(c) for c in COLS} | {"observed_posts": len(posts)}


def window_features(records, snapshots) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    rows = joblib.Parallel(6)(joblib.delayed(features)(u, snapshots[u], records[u][-LAST:]) for u in records)
    f = pd.DataFrame(rows, index=list(records)).astype(float)
    f.to_parquet(CACHE)
    return f


def campaign_posts(kind, n, rng, times) -> list[tuple]:
    """n campaign posts of one member; only the named kind of object is shared."""
    words = np.array(["morning", "really", "people", "never", "coffee", "weather", "music", "think", "home",
                      "great", "watching", "tonight", "school", "friends", "game", "maybe", "story", "again"])
    out = []
    for j, when in zip(rng.integers(0, POOL, n), rng.choice(times, n)):
        text, targets, urls = " ".join(rng.choice(words, 7)) + f" {''.join(rng.choice(list('abcdefgh'), 6))}", [], []
        if kind == "copypaste":
            text = f"everyone should read this message variant {'abcdefghijklmnopqrst'[j]} and share it now"
        elif kind == "link":
            urls = [f"https://campaign.example/p{j}"]
            text += " " + urls[0]
        elif kind == "mention":
            targets = [f"campaign_target_{j}"]
            text = f"@campaign_target_{j} {text}"
        else:
            text += f" #campaigntag{j}"
        out.append((when.to_pydatetime() + timedelta(seconds=int(rng.integers(0, 1800))), text, targets, urls,
                    False, False))
    return out


def inject(records, objects, members, kind, intensity, rng):
    """Changed objects of the whole world and changed records of the members."""
    times = pd.date_range("2014-08-01", periods=60, freq="D", tz="UTC") + pd.Timedelta(hours=14)
    world, changed = dict(objects), {}
    for a in members:
        new = campaign_posts(kind, max(1, round(intensity * len(records[a]))), rng, times)
        changed[a] = records[a] + new
        world[a] = objects[a] + [o for c, x, m, u, rp, _ in new for o in system_post_objects(x, m, u, rp)]
    return world, changed


def scenarios():
    return list(itertools.product(KINDS, SIZES, INTENSITIES, SEEDS))


def sims_of(world):
    return {c: similarity(world, max_account_share=BACKGROUND, channels=c) for c in ("joint", "max")}


def graph_stats(sims, graph):
    g = knn_graph(sims[graph["channels"]], K, graph["min_similarity"], graph["min_shared"])
    comms = communities(g, SEED, graph["resolution"])
    return [(s, comms[s["community"]]) for s in community_scores(g, comms, 2, K)]


def flagged(found, rule) -> set[str]:
    out = set()
    for s, members in found:
        if (s["size"] >= rule["min_size"] and s["mean_weight"] >= rule["min_weight"]
                and s["cohesion"] >= rule["min_cohesion"]):
            out |= members
    return out


def calibrate_one(records, objects, scenario):
    kind, size, intensity, seed = scenario
    rng = np.random.default_rng([seed, size, int(intensity * 100), KINDS.index(kind)])
    members = set(rng.choice(sorted(objects), size, replace=False))
    world, _ = inject(records, objects, sorted(members), kind, intensity, rng)
    sims = sims_of(world)
    others = len(objects) - size
    out = np.empty((len(GRAPHS), len(RULES), 2), dtype=np.float32)
    for gi, graph in enumerate(GRAPHS):
        found = graph_stats(sims, graph)
        for ri, rule in enumerate(RULES):
            f = flagged(found, rule)
            out[gi, ri] = len(f & members) / size, len(f - members) / others
    return out


def evaluate_one(records, objects, snapshots, model, base_score, configs, scenario):
    kind, size, intensity, seed = scenario
    rng = np.random.default_rng([100 + seed, size, int(intensity * 100), KINDS.index(kind)])
    members = sorted(rng.choice(sorted(objects), size, replace=False))
    world, changed = inject(records, objects, members, kind, intensity, rng)
    sims = sims_of(world)
    f = pd.DataFrame([features(a, snapshots[a], changed[a]) for a in members], index=members).astype(float)
    new = model.predict_proba(f[COLS])[:, 1]
    acc = set(f.index[(new >= THRESHOLD) & (f.observed_posts >= MIN_POSTS).to_numpy()])
    rest = set(objects) - set(members)
    base_flag = {a for a in objects if base_score.get(a, 0) >= THRESHOLD}
    row = {"kind": kind, "size": size, "intensity": intensity, "seed": seed,
           "account_recall_before": len(base_flag & set(members)) / size,
           "account_recall_after": len(acc) / size, "account_fpr": len(base_flag & rest) / len(rest)}
    for name, c in configs.items():
        fl = flagged(graph_stats(sims, c), c)
        row[f"{name}_recall"] = len(fl & set(members)) / size
        row[f"{name}_fpr"] = len(fl & rest) / len(rest)
        row[f"{name}_or_recall"] = len((fl | acc) & set(members)) / size
        row[f"{name}_or_fpr"] = len((fl | base_flag) & rest) / len(rest)
    return row


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    records, objects, group_of = load()
    users = pd.read_parquet(OUT / "users.parquet").drop_duplicates("user_id")
    snapshots = {u.user_id: account(u) for u in users.itertuples(index=False) if u.user_id in records}
    f = window_features(records, snapshots)
    print(f"{len(records)} accounts, features {time.time() - t0:.0f}s", flush=True)
    genuine = sorted(group_of.index[group_of == "genuine"])
    cal, test = train_test_split(genuine, test_size=0.5, random_state=SEED)
    # The account model is trained on the calibration half as its human class.
    table = pd.read_parquet(OUT / "account_features_v2.parquet").set_index("user_id")
    extra = table[(table.observed_posts >= MIN_POSTS) & table.group.isin(["fake_followers", "traditional_1"])]
    enough = f.observed_posts >= MIN_POSTS
    social = {c: [a for a in group_of.index[group_of == c] if enough[a]] for c in GROUPS[1:]}
    human = [a for a in cal if enough[a]]

    def fit(held=None):
        bots = [a for c, v in social.items() if c != held for a in v]
        x = pd.concat([f.loc[human + bots, COLS], extra[COLS]])
        return xgb.XGBClassifier(**PARAMS).fit(x, np.r_[np.zeros(len(human)), np.ones(len(bots) + len(extra))])

    report = {"window": WINDOW, "accounts": group_of.value_counts().to_dict(), "k": K,
              "max_account_share": BACKGROUND, "genuine_calibration": len(cal), "genuine_test": len(test),
              "account_threshold": THRESHOLD, "current_rule": CURRENT, "scenario_count": len(scenarios())}

    sub = lambda ids, d: {a: d[a] for a in ids}  # noqa: E731
    cal_r, cal_o = sub(cal, records), sub(cal, objects)
    grid = np.mean(joblib.Parallel(5)(joblib.delayed(calibrate_one)(cal_r, cal_o, s) for s in scenarios()), axis=0)
    print(f"calibration {time.time() - t0:.0f}s", flush=True)
    configs, report["selection"] = {"current": CURRENT}, {}
    for budget in FPR_BUDGETS:
        ok = np.where(grid[:, :, 1] <= budget, grid[:, :, 0], -1)
        gi, ri = np.unravel_index(int(ok.argmax()), ok.shape)
        c = GRAPHS[gi] | RULES[ri]
        configs[f"fpr{int(budget * 100)}"] = c
        report["selection"][f"fpr{int(budget * 100)}"] = {
            "rule": c, "calibration_recall": float(grid[gi, ri, 0]), "calibration_fpr": float(grid[gi, ri, 1])}
    gi = GRAPHS.index({k: CURRENT[k] for k in GRAPHS[0]})
    ri = RULES.index({k: CURRENT[k] for k in RULES[0]})
    report["selection"]["current"] = {"rule": CURRENT, "calibration_recall": float(grid[gi, ri, 0]),
                                      "calibration_fpr": float(grid[gi, ri, 1])}
    print(json.dumps(report["selection"], indent=1), flush=True)

    model = fit()
    te = [a for a in test if enough[a]]
    base = dict(zip(te, model.predict_proba(f.loc[te, COLS])[:, 1]))
    te_r, te_o = sub(test, records), sub(test, objects)
    rows = joblib.Parallel(5)(joblib.delayed(evaluate_one)(
        te_r, te_o, sub(test, snapshots), model, base, configs, s) for s in scenarios())
    rows = pd.DataFrame(rows)
    report["synthetic_test"] = {
        "mean": rows.drop(columns=["kind", "size", "intensity", "seed"]).mean().to_dict(),
        "by_intensity": rows.drop(columns=["kind", "seed", "size"]).groupby("intensity").mean().reset_index().to_dict("records"),
        "by_kind": rows.drop(columns=["seed", "size", "intensity"]).groupby("kind").mean().reset_index().to_dict("records"),
        "by_size": rows.drop(columns=["seed", "kind", "intensity"]).groupby("size").mean().reset_index().to_dict("records"),
    }
    rows.to_csv(REPORT / "synthetic_test.csv", index=False)
    print(f"synthetic {time.time() - t0:.0f}s", flush=True)

    # Real campaigns: the world is the test half of genuine accounts and all social spambots.
    world = {a: o for a, o in objects.items() if a not in set(cal)}
    sims = sims_of(world)
    sim = sims["joint"]
    ids = np.array(sim.accounts)
    grp = group_of.loc[ids].to_numpy()
    y = (grp != "genuine").astype(int)
    report["coordination_score"] = {}
    for channels, s in sims.items():
        coord = coordination_score(s, K)
        report["coordination_score"][channels] = {"all": ranking(y, coord)} | {
            c: ranking(y[np.isin(grp, ["genuine", c])], coord[np.isin(grp, ["genuine", c])]) for c in GROUPS[1:]}
    report["cresci"] = {}
    for name, c in configs.items():
        fl = np.isin(ids, list(flagged(graph_stats(sims, c), c)))
        res = {"precision": float(y[fl].mean()) if fl.any() else None, "recall": float(fl[y == 1].mean()),
               "fpr": float(fl[y == 0].mean()), "genuine_flagged": int(fl[y == 0].sum()),
               "recall_by_campaign": {g: float(fl[grp == g].mean()) for g in GROUPS[1:]}, "loco": {}}
        report["cresci"][name] = res
    pos = {a: i for i, a in enumerate(ids)}
    for held in GROUPS[1:]:
        m = fit(held)
        members = [a for a in group_of.index[group_of == held]]

        def acc_flag(a_ids):
            s = m.predict_proba(f.loc[a_ids, COLS])[:, 1]
            return (s >= THRESHOLD) & enough[a_ids].to_numpy()

        ah, ag = acc_flag(members), acc_flag(test)
        for name, c in configs.items():
            found = report["cresci"][name].setdefault("_flags", None)
            if found is None:
                found = np.isin(ids, list(flagged(graph_stats(sims, c), c)))
                report["cresci"][name]["_flags"] = found
            gh, gg = found[[pos[a] for a in members]], found[[pos[a] for a in test]]
            report["cresci"][name]["loco"][held] = {
                "account_recall": float(ah.mean()), "account_fpr": float(ag.mean()),
                "or_recall": float((ah | gh).mean()), "or_fpr": float((ag | gg).mean())}
    for r in report["cresci"].values():
        r.pop("_flags")
    report["seconds"] = round(time.time() - t0, 1)
    (REPORT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("synthetic_test", "coordination_score", "cresci")}, indent=1))
    print("done", report["seconds"])


if __name__ == "__main__":
    main()
