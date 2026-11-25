"""Group level on Cresci-2017: co-activity network, communities, synthetic campaigns.

All groups are restricted to one common period (WINDOW) so that time alone does
not separate them. fake_followers (2009-2013) and traditional_1 (2009-2010) have no
tweets in that period and are excluded from the group experiments.

Experiments:
 1. unsupervised coordination score (mean similarity to k nearest accounts) vs label;
 2. Louvain communities vs campaigns: ARI/NMI, purity/recall per campaign,
    precision/recall of accounts in flagged communities;
 3. ablation by object kind;
 4. leave-one-campaign-out: account model (never saw the campaign) alone vs
    account model + coordination score;
 5. synthetic campaigns injected into genuine accounts: size x intensity x pool.

Run:  .venv/bin/python -m scripts.cresci_group_experiment
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import (adjusted_rand_score, average_precision_score,
                             normalized_mutual_info_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from engine.coactivity import (OBJECT_KINDS, community_scores, communities,
                               coordination_score, knn_graph, post_objects, similarity)
from engine.features import PORTABLE_FEATURES, PROFILE_V2_FEATURES
from scripts.cresci_account_experiment import PARAMS
from scripts.prepare_cresci import OUT, frame

REPORT = Path("media/models/experiments/cresci-group-v2")
WINDOW = ("2014-06-01", "2014-11-01")
GROUPS = ["genuine", "social_1", "social_2", "social_3"]
MIN_POSTS = 10
K = 10
MIN_SIM = 0.2
SEED = 42


def load() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    cols = ["user_id", "group", "created_at", "text", "reply_to_status",
            "retweet_of_status", "retweet_of_user", "mentions", "hashtags"]
    t = frame(pq.read_table(OUT / "tweets.parquet", columns=cols,
                            filters=[("group", "in", GROUPS)]))
    t = t[(t.created_at >= WINDOW[0]) & (t.created_at < WINDOW[1])]
    n = t.groupby("user_id").size()
    t = t[t.user_id.isin(n[n >= MIN_POSTS].index)]
    accounts = t.groupby("user_id")["group"].first()
    return t, accounts


def objects_for(t: pd.DataFrame, kinds=OBJECT_KINDS) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in t.itertuples(index=False):
        out.setdefault(r.user_id, []).extend(post_objects(
            retweet_of_status=r.retweet_of_status, retweet_of_user=r.retweet_of_user,
            reply_to_status=r.reply_to_status, mentions=r.mentions, hashtags=r.hashtags,
            text=r.text, kinds=kinds))
    return out


def ranking(y, s) -> dict:
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def community_eval(sim, groups_of: pd.Series, flag_thresholds=(0.1, 0.2, 0.3, 0.5)) -> dict:
    g = knn_graph(sim, K, MIN_SIM)
    comms = communities(g, SEED)
    scored = {c["community"]: c for c in community_scores(g, comms)}
    member = {a: i for i, c in enumerate(comms) for a in c}
    ids = sim.accounts
    pred = np.array([member.get(a, -1 - i) for i, a in enumerate(ids)])  # isolated = singleton
    truth = groups_of.loc[ids].to_numpy()
    bots = truth != "genuine"
    result = {
        "graph": {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                  "isolated": int(sum(1 for _, d in g.degree() if d == 0)),
                  "communities_size_ge3": len(scored)},
        "bots_only": {"ari": float(adjusted_rand_score(truth[bots], pred[bots])),
                      "nmi": float(normalized_mutual_info_score(truth[bots], pred[bots]))},
        "campaigns": {},
        "flagged": {},
        "top_communities": [],
    }
    for camp in sorted(set(truth[bots])):
        in_camp = truth == camp
        best = Counter(pred[in_camp]).most_common(1)[0][0]
        in_best = pred == best
        result["campaigns"][camp] = {
            "size": int(in_camp.sum()),
            "best_community_recall": float((in_best & in_camp).sum() / in_camp.sum()),
            "best_community_purity": float((in_best & in_camp).sum() / in_best.sum()),
            "communities_touched": int(len(set(pred[in_camp]))),
        }
    for thr in flag_thresholds:
        flagged_comms = {c for c, s in scored.items() if s["mean_weight"] >= thr and s["size"] >= 5}
        flagged = np.isin(pred, list(flagged_comms))
        tp = int((flagged & bots).sum())
        result["flagged"][f"mean_weight>={thr}"] = {
            "communities": len(flagged_comms), "accounts": int(flagged.sum()),
            "precision": tp / max(int(flagged.sum()), 1), "recall": tp / int(bots.sum()),
            "genuine_flagged": int((flagged & ~bots).sum()),
        }
    for c in sorted(scored.values(), key=lambda c: -c["size"])[:10]:
        members = [ids[i] for i in np.where(pred == c["community"])[0]]
        c = dict(c, composition=dict(Counter(groups_of.loc[members])))
        result["top_communities"].append(c)
    return result


def loco_combined(sim, coord: np.ndarray, groups_of: pd.Series) -> dict:
    feats = pd.read_parquet(OUT / "account_features_v2.parquet").set_index("user_id")
    cols = PROFILE_V2_FEATURES + PORTABLE_FEATURES
    ids = np.array(sim.accounts)
    f = feats.loc[ids]
    y = (groups_of.loc[ids] != "genuine").astype(int).to_numpy()
    grp = groups_of.loc[ids].to_numpy()
    genuine_idx = np.where(grp == "genuine")[0]
    g_train, g_test = train_test_split(genuine_idx, test_size=0.3, random_state=SEED)
    # Other campaigns from the full Cresci table (all periods) are allowed in training.
    others_all = feats[(feats.observed_posts >= 20) & (feats.group != "genuine")]
    out = {}
    for held in ["social_1", "social_2", "social_3"]:
        train_x = pd.concat([f.iloc[g_train][cols], others_all[others_all.group != held][cols]])
        train_y = np.r_[np.zeros(len(g_train)), np.ones((others_all.group != held).sum())]
        m = xgb.XGBClassifier(**PARAMS).fit(train_x, train_y)
        test = np.r_[g_test, np.where(grp == held)[0]]
        s_acc = m.predict_proba(f.iloc[test][cols])[:, 1]
        s_coord = coord[test]
        combined = (rankdata(s_acc) + rankdata(s_coord)) / (2 * len(test))
        yt = y[test]
        out[held] = {
            "n": int(len(test)), "bots": int(yt.sum()),
            "account_model": ranking(yt, s_acc),
            "coordination_only": ranking(yt, s_coord),
            "combined_rank_mean": ranking(yt, combined),
            "combined_max": ranking(yt, np.maximum(rankdata(s_acc), rankdata(s_coord))),
        }
    return out


def inject(base: dict[str, list[str]], members: list[str], intensity: float, pool: int,
           rng: np.random.Generator) -> dict[str, list[str]]:
    """Add campaign actions: each member repeats objects from a shared pool."""
    out = {a: list(v) for a, v in base.items()}
    for a in members:
        n_new = max(1, int(round(intensity * len(base[a]))))
        out[a].extend(f"rt:campaign_{j}" for j in rng.integers(0, pool, n_new))
    return out


def synthetic(genuine_objects: dict[str, list[str]], seeds=(0, 1, 2)) -> list[dict]:
    rows = []
    accounts = sorted(genuine_objects)
    for size in (10, 30, 100):
        for intensity in (0.02, 0.05, 0.1, 0.2):
            for pool in (20, 200):
                aucs, recalls, purities = [], [], []
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    members = list(rng.choice(accounts, size, replace=False))
                    sim = similarity(inject(genuine_objects, members, intensity, pool, rng))
                    coord = coordination_score(sim, K)
                    y = np.isin(sim.accounts, members).astype(int)
                    aucs.append(roc_auc_score(y, coord))
                    g = knn_graph(sim, K, MIN_SIM)
                    comms = communities(g, SEED)
                    best = max(comms, key=lambda c: len(c & set(members)), default=set())
                    hit = len(best & set(members))
                    recalls.append(hit / size)
                    purities.append(hit / len(best) if best else 0.0)
                row = {"size": size, "intensity": intensity, "pool": pool,
                       "roc_auc": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
                       "best_community_recall": float(np.mean(recalls)),
                       "best_community_purity": float(np.mean(purities))}
                rows.append(row)
                print("  synthetic", row, flush=True)
    return rows


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tweets, groups_of = load()
    print(f"window {WINDOW}: {len(groups_of)} accounts, {len(tweets)} tweets", flush=True)
    report = {"window": WINDOW, "min_posts": MIN_POSTS, "k": K, "min_similarity": MIN_SIM,
              "accounts": groups_of.value_counts().to_dict(), "tweets": len(tweets)}

    objects = objects_for(tweets)
    sim = similarity(objects)
    coord = coordination_score(sim, K)
    y = (groups_of.loc[sim.accounts] != "genuine").astype(int).to_numpy()
    report["objects_kept"] = sim.objects_kept
    report["coordination_score"] = {"all": ranking(y, coord)}
    grp = groups_of.loc[sim.accounts].to_numpy()
    for camp in ["social_1", "social_2", "social_3"]:
        m = np.isin(grp, ["genuine", camp])
        report["coordination_score"][camp] = ranking(y[m], coord[m])
    print("coordination", report["coordination_score"], flush=True)

    report["communities"] = community_eval(sim, groups_of)
    print("communities", json.dumps({k: report["communities"][k] for k in
                                     ("graph", "bots_only", "campaigns", "flagged")}, indent=1), flush=True)

    report["ablation_without_kind"] = {}
    for kind in OBJECT_KINDS:
        s = similarity(objects_for(tweets, tuple(k for k in OBJECT_KINDS if k != kind)))
        c = coordination_score(s, K)
        yy = (groups_of.loc[s.accounts] != "genuine").astype(int).to_numpy()
        report["ablation_without_kind"][kind] = ranking(yy, c)
    for kind in OBJECT_KINDS:
        s = similarity(objects_for(tweets, (kind,)))
        c = coordination_score(s, K)
        yy = (groups_of.loc[s.accounts] != "genuine").astype(int).to_numpy()
        report.setdefault("only_kind", {})[kind] = ranking(yy, c)
    print("ablation", report["ablation_without_kind"], report["only_kind"], flush=True)

    report["loco_account_vs_combined"] = loco_combined(sim, coord, groups_of)
    print("loco", json.dumps(report["loco_account_vs_combined"], indent=1), flush=True)

    genuine = {a: v for a, v in objects.items() if groups_of.loc[a] == "genuine"}
    report["synthetic_injection"] = {"genuine_accounts": len(genuine), "rows": synthetic(genuine)}
    report["seconds"] = round(time.time() - t0, 1)
    (REPORT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("done", report["seconds"])


if __name__ == "__main__":
    main()
