"""Choose the community method on synthetic campaigns, then check it on Cresci campaigns.

Selection never looks at Cresci campaign labels: every method is scored on campaigns
injected into genuine accounts (known membership), and the best one by mean F1 of
the best-matching community is then evaluated on the real social spambot campaigns.

Also evaluates the combination rule "priority = account flag OR coordinated-group flag"
against the account model alone (leave-one-campaign-out and synthetic campaigns).

Run:  .venv/bin/python -m scripts.cresci_partition_tuning
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

from engine.coactivity import community_scores, communities, knn_graph, similarity
from engine.features import PORTABLE_FEATURES, PROFILE_V2_FEATURES
from scripts.cresci_account_experiment import PARAMS
from scripts.cresci_group_experiment import K, MIN_SIM, SEED, inject, load, objects_for
from scripts.prepare_cresci import OUT

REPORT = Path("media/models/experiments/cresci-partition-v1")
METHODS = (
    [("louvain", r) for r in (0.1, 0.3, 0.5, 1.0)]
    + [("leiden", r) for r in (0.01, 0.05, 0.1, 0.3, 1.0)]
    + [("components", None)]
)
FLAG_MIN_SIZE = 5
FLAG_MIN_WEIGHT = 0.2
ACCOUNT_COLS = PROFILE_V2_FEATURES + PORTABLE_FEATURES


def partition(sim, method, resolution):
    g = knn_graph(sim, K, MIN_SIM)
    comms = communities(g, SEED, resolution or 1.0, method)
    return g, comms


def flagged_accounts(g, comms) -> set[str]:
    """Members of communities with >= FLAG_MIN_SIZE accounts and mean tie >= FLAG_MIN_WEIGHT."""
    out = set()
    for c in community_scores(g, comms, FLAG_MIN_SIZE):
        if c["mean_weight"] >= FLAG_MIN_WEIGHT:
            out |= comms[c["community"]]
    return out


def best_f1(comms, members: set[str]) -> tuple[float, float, float]:
    best = max(comms, key=lambda c: len(c & members), default=set())
    hit = len(best & members)
    r, p = hit / len(members), (hit / len(best) if best else 0.0)
    return (2 * p * r / (p + r) if hit else 0.0), r, p


def synthetic_selection(genuine_objects) -> dict:
    accounts = sorted(genuine_objects)
    scenarios = [(s, i, p) for s in (10, 30, 100) for i in (0.05, 0.1) for p in (20, 200)]
    out = {}
    for method, res in METHODS:
        f1s = []
        for size, intensity, pool in scenarios:
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed)
                members = set(rng.choice(accounts, size, replace=False))
                sim = similarity(inject(genuine_objects, list(members), intensity, pool, rng))
                _, comms = partition(sim, method, res)
                f1s.append(best_f1(comms, members)[0])
        key = f"{method}" + (f"@{res}" if res is not None else "")
        out[key] = {"mean_best_f1": float(np.mean(f1s)), "min_best_f1": float(np.min(f1s))}
        print("  synthetic", key, out[key], flush=True)
    return out


def cresci_eval(sim, groups_of, method, res) -> dict:
    g, comms = partition(sim, method, res)
    member = {a: i for i, c in enumerate(comms) for a in c}
    ids = sim.accounts
    pred = np.array([member.get(a, -1 - i) for i, a in enumerate(ids)])
    truth = groups_of.loc[ids].to_numpy()
    bots = truth != "genuine"
    flagged = np.isin(ids, list(flagged_accounts(g, comms)))
    tp = int((flagged & bots).sum())
    camp = {}
    for c in sorted(set(truth[bots])):
        f1, r, p = best_f1(comms, set(np.array(ids)[truth == c]))
        camp[c] = {"best_f1": f1, "recall": r, "purity": p,
                   "communities_touched": int(len(set(pred[truth == c])))}
    return {
        "communities": len(comms),
        "ari_bots": float(adjusted_rand_score(truth[bots], pred[bots])),
        "nmi_bots": float(normalized_mutual_info_score(truth[bots], pred[bots])),
        "campaigns": camp,
        "flag": {"accounts": int(flagged.sum()), "precision": tp / max(int(flagged.sum()), 1),
                 "recall": tp / int(bots.sum()), "genuine_flagged": int((flagged & ~bots).sum())},
    }


def combination(sim, groups_of, method, res, genuine_objects) -> dict:
    feats = pd.read_parquet(OUT / "account_features.parquet").set_index("user_id")
    others = feats[(feats.observed_posts >= 20) & (feats.group != "genuine")]
    ids = np.array(sim.accounts)
    grp = groups_of.loc[ids].to_numpy()
    g_train, g_test = train_test_split(np.where(grp == "genuine")[0], test_size=0.3,
                                       random_state=SEED)
    g, comms = partition(sim, method, res)
    group_flag = np.isin(ids, list(flagged_accounts(g, comms)))
    out = {"loco": {}}

    def pr(flag, y):
        tp = int((flag & (y == 1)).sum())
        return {"precision": tp / max(int(flag.sum()), 1), "recall": tp / max(int(y.sum()), 1),
                "false_positive_rate": float(flag[y == 0].mean())}

    for held in ["social_1", "social_2", "social_3"]:
        tr = others[others.group != held]
        x = pd.concat([feats.loc[ids[g_train], ACCOUNT_COLS], tr[ACCOUNT_COLS]])
        y_tr = np.r_[np.zeros(len(g_train)), np.ones(len(tr))]
        m = xgb.XGBClassifier(**PARAMS).fit(x, y_tr)
        test = np.r_[g_test, np.where(grp == held)[0]]
        acc_flag = m.predict_proba(feats.loc[ids[test], ACCOUNT_COLS])[:, 1] >= 0.5
        y = (grp[test] != "genuine").astype(int)
        out["loco"][held] = {"account_only": pr(acc_flag, y), "group_only": pr(group_flag[test], y),
                             "account_or_group": pr(acc_flag | group_flag[test], y)}
        print("  loco", held, out["loco"][held], flush=True)

    # Synthetic: coordinated accounts that look human. The account model is trained on
    # train-genuine + all Cresci bots, members come from test-genuine accounts.
    m = xgb.XGBClassifier(**PARAMS).fit(
        pd.concat([feats.loc[ids[g_train], ACCOUNT_COLS], others[ACCOUNT_COLS]]),
        np.r_[np.zeros(len(g_train)), np.ones(len(others))])
    test_genuine = sorted(set(ids[g_test]) & set(genuine_objects))
    acc_score = dict(zip(test_genuine, m.predict_proba(feats.loc[test_genuine, ACCOUNT_COLS])[:, 1]))
    rows = []
    for size in (10, 30):
        for intensity in (0.05, 0.1, 0.2):
            rec_acc, rec_grp, fp = [], [], []
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed)
                members = set(rng.choice(test_genuine, size, replace=False))
                world = {a: genuine_objects[a] for a in test_genuine}
                s = similarity(inject(world, list(members), intensity, 20, rng))
                gg, cc = partition(s, method, res)
                fl = flagged_accounts(gg, cc)
                rec_acc.append(np.mean([acc_score[a] >= 0.5 for a in members]))
                rec_grp.append(len(fl & members) / size)
                fp.append(len(fl - members) / (len(test_genuine) - size))
            rows.append({"size": size, "intensity": intensity,
                         "recall_account_model": float(np.mean(rec_acc)),
                         "recall_group_flag": float(np.mean(rec_grp)),
                         "group_flag_fpr_other_genuine": float(np.mean(fp))})
            print("  synthetic-combo", rows[-1], flush=True)
    out["synthetic_humanlike_campaign"] = rows
    return out


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tweets, groups_of = load()
    objects = objects_for(tweets)
    genuine = {a: v for a, v in objects.items() if groups_of.loc[a] == "genuine"}
    report = {"k": K, "min_similarity": MIN_SIM,
              "flag_rule": {"min_size": FLAG_MIN_SIZE, "min_mean_weight": FLAG_MIN_WEIGHT}}
    report["selection_on_synthetic"] = synthetic_selection(genuine)
    chosen = max(report["selection_on_synthetic"],
                 key=lambda k: report["selection_on_synthetic"][k]["mean_best_f1"])
    report["chosen"] = chosen
    sim = similarity(objects)
    report["cresci_all_methods"] = {}
    for method, res in METHODS:
        key = f"{method}" + (f"@{res}" if res is not None else "")
        report["cresci_all_methods"][key] = cresci_eval(sim, groups_of, method, res)
        r = report["cresci_all_methods"][key]
        print(f"cresci {key:16s} ARI {r['ari_bots']:.3f} NMI {r['nmi_bots']:.3f} "
              f"flag P {r['flag']['precision']:.3f} R {r['flag']['recall']:.3f} "
              + " ".join(f"{c}:F1 {v['best_f1']:.2f}" for c, v in r["campaigns"].items()), flush=True)
    method, _, res = chosen.partition("@")
    report["combination"] = combination(sim, groups_of, method, float(res) if res else None, genuine)
    report["seconds"] = round(time.time() - t0, 1)
    (REPORT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("chosen", chosen, "done", report["seconds"])


if __name__ == "__main__":
    main()
