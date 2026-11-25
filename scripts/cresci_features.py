"""Account features for Cresci-2017 computed by the system engine (parity with inference).

Each account gets its last N tweets as a complete observation window
[first of the N, last of the N]. Profile fields follow AccountSnapshot.
Targets are @-mentions plus the replied-to user, as the engine expects.

Output: data/processed/cresci2017-v1/account_features_v2.parquet

Run:  .venv/bin/python -m scripts.cresci_features [--last 200]
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import pyarrow.parquet as pq

from engine.contracts import AccountSnapshot, Observation, Post
from engine.features import BASE_FEATURES, extract_features, extract_profile_features
from scripts.prepare_cresci import OUT, frame

TWEET_COLS = ["tweet_id", "user_id", "created_at", "text", "reply_to_status",
              "reply_to_user", "retweet_of_status", "retweet_of_user", "mentions", "urls"]


def last_tweets(n: int) -> pd.DataFrame:
    t = frame(pq.read_table(OUT / "tweets.parquet", columns=TWEET_COLS))
    t = t.drop_duplicates("tweet_id")
    t = t.sort_values(["user_id", "created_at"])
    return t.groupby("user_id", sort=False).tail(n)


def account(u) -> AccountSnapshot:
    def num(v):
        return None if pd.isna(v) else int(v)
    image = u.profile_image_url or ""
    return AccountSnapshot(
        platform="twitter", external_user_id=u.user_id, username=u.screen_name or None,
        display_name=u.name or None, biography=u.description,
        followers=num(u.followers_count), following=num(u.friends_count),
        has_avatar=bool(image), avatar_default=bool(u.default_profile_image),
        created_at=None if pd.isna(u.account_created_at) else u.account_created_at.to_pydatetime(),
        collected_at=None if pd.isna(u.crawled_at) else u.crawled_at.to_pydatetime(),
    )


def posts(uid: str, rows: pd.DataFrame) -> list[Post]:
    out = []
    for r in rows.itertuples(index=False):
        targets = list(r.mentions) if r.mentions is not None else []
        if r.reply_to_user:
            targets.append(f"id:{r.reply_to_user}")
        out.append(Post(
            platform="twitter", external_post_id=r.tweet_id, author_id=uid,
            created_at=r.created_at.to_pydatetime(), text=r.text,
            is_reply=r.reply_to_status is not None,
            is_repost=r.retweet_of_status is not None or r.retweet_of_user is not None,
            urls=list(r.urls) if r.urls is not None else [], targets=targets,
        ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=200)
    args = parser.parse_args()
    t0 = time.time()
    users = pd.read_parquet(OUT / "users.parquet").drop_duplicates("user_id")
    tweets = last_tweets(args.last)
    by_user = dict(tuple(tweets.groupby("user_id", sort=False)))
    print(f"loaded {len(users)} users, {len(tweets)} tweets in {time.time() - t0:.0f}s", flush=True)
    rows = []
    for i, u in enumerate(users.itertuples(index=False)):
        acc = account(u)
        mine = by_user.get(u.user_id)
        ps = posts(u.user_id, mine) if mine is not None else []
        if ps:
            obs = Observation(start=ps[0].created_at, end=ps[-1].created_at,
                              source="cresci2017_timeline", complete=ps[0].created_at < ps[-1].created_at)
        else:
            ref = acc.collected_at or acc.created_at
            obs = Observation(start=ref, end=ref, source="cresci2017_timeline") if ref else None
        row = extract_features(acc, ps, obs) if obs else dict.fromkeys(BASE_FEATURES)
        row.update(extract_profile_features(acc))
        row.update(user_id=u.user_id, label=int(u.label == "bot"), group=u.group,
                   observed_posts=len(ps))
        rows.append(row)
        if i % 2000 == 0:
            print(f"  {i} accounts ({time.time() - t0:.0f}s)", flush=True)
    table = pd.DataFrame(rows)
    num = [c for c in table.columns if c not in {"user_id", "group"}]
    table[num] = table[num].astype(float)
    table.to_parquet(OUT / "account_features_v2.parquet", index=False)
    print(table.groupby("group")[["label", "observed_posts"]].agg(["count", "mean"]))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
