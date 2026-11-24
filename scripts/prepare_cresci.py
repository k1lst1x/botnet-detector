"""Convert Cresci-2017 (Fake Project) CSV files to Parquet tables with group labels.

Input:  data/raw/cresci2017/datasets_full.csv/<group>.csv/{users,tweets}.csv
Output: data/processed/cresci2017-v1/{users,tweets}.parquet + manifest.json

Every group is one labelled population; social spambot groups are single
coordinated campaigns (Cresci et al., WWW'17) and serve as group ground truth.

Run:  .venv/bin/python -m scripts.prepare_cresci
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW = Path("data/raw/cresci2017/datasets_full.csv")
OUT = Path("data/processed/cresci2017-v1")
GROUPS = {
    "genuine_accounts": ("human", "genuine"),
    "social_spambots_1": ("bot", "social_1"),
    "social_spambots_2": ("bot", "social_2"),
    "social_spambots_3": ("bot", "social_3"),
    "traditional_spambots_1": ("bot", "traditional_1"),
    "traditional_spambots_2": ("bot", "traditional_2"),
    "traditional_spambots_3": ("bot", "traditional_3"),
    "traditional_spambots_4": ("bot", "traditional_4"),
    "fake_followers": ("bot", "fake_followers"),
}
USER_COLS = [
    "id", "name", "screen_name", "statuses_count", "followers_count", "friends_count",
    "favourites_count", "listed_count", "url", "lang", "location", "default_profile",
    "default_profile_image", "profile_image_url", "protected", "verified", "description",
    "created_at", "timestamp",
]
TWEET_COLS = [
    "id", "text", "source", "user_id", "in_reply_to_status_id", "in_reply_to_user_id",
    "retweeted_status_id", "num_hashtags", "num_urls", "num_mentions", "created_at",
]
RT = re.compile(r"^RT @(\w+)")
MENTION = re.compile(r"@(\w+)")
HASHTAG = re.compile(r"#(\w+)")
URL = re.compile(r"https?://\S+")
SOURCE = re.compile(r">([^<]*)<")
TWITTER_DATE = "%a %b %d %H:%M:%S %z %Y"

TWEET_SCHEMA = pa.schema([
    ("tweet_id", pa.string()), ("user_id", pa.string()), ("group", pa.string()),
    ("created_at", pa.timestamp("s", tz="UTC")), ("text", pa.string()), ("source", pa.string()),
    ("reply_to_status", pa.string()), ("reply_to_user", pa.string()),
    ("retweet_of_status", pa.string()), ("retweet_of_user", pa.string()),
    ("mentions", pa.list_(pa.string())), ("hashtags", pa.list_(pa.string())),
    ("urls", pa.list_(pa.string())),
])


def _id(series: pd.Series) -> pd.Series:
    """Twitter ids as strings; '0', 'NULL' and blanks mean 'not set'."""
    s = series.astype("string").str.strip()
    return s.where(~s.isin(["", "0", "NULL", "nan"]), None)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _dates(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, format=TWITTER_DATE, errors="coerce", utc=True)
    # traditional_spambots_1 stores epoch milliseconds with a Java "L" suffix.
    epoch = pd.to_numeric(series.str.rstrip("L"), errors="coerce")
    fallback = pd.to_datetime(epoch, unit="ms", errors="coerce", utc=True)
    return out.fillna(fallback)


def convert_tweets(group: str, writer: pq.ParquetWriter) -> dict:
    path = RAW / f"{group}.csv" / "tweets.csv"
    if not path.exists():
        return {"rows": 0, "bad_dates": 0}
    rows = bad = 0
    for chunk in pd.read_csv(path, dtype=str, usecols=lambda c: c in TWEET_COLS,
                             chunksize=500_000, keep_default_na=False,
                             on_bad_lines="warn", encoding="utf-8", encoding_errors="replace"):
        created = _dates(chunk["created_at"])
        bad += int(created.isna().sum())
        keep = created.notna() & (_id(chunk["id"]).notna()) & (_id(chunk["user_id"]).notna())
        chunk, created = chunk[keep], created[keep]
        text = chunk["text"].fillna("")
        rt_user = text.str.extract(RT, expand=False).str.lower()
        frame = pd.DataFrame({
            "tweet_id": _id(chunk["id"]),
            "user_id": _id(chunk["user_id"]),
            "group": GROUPS[group][1],
            "created_at": created.dt.floor("s"),
            "text": text,
            "source": chunk["source"].fillna("").str.extract(SOURCE, expand=False).fillna(""),
            "reply_to_status": _id(chunk["in_reply_to_status_id"]),
            "reply_to_user": _id(chunk["in_reply_to_user_id"]),
            "retweet_of_status": _id(chunk["retweeted_status_id"]),
            "retweet_of_user": rt_user.where(rt_user.notna(), None),
            "mentions": text.str.lower().str.findall(MENTION),
            "hashtags": text.str.lower().str.findall(HASHTAG),
            "urls": text.str.findall(URL),
        })
        writer.write_table(pa.Table.from_pandas(frame, schema=TWEET_SCHEMA, preserve_index=False))
        rows += len(frame)
        print(f"  {group}: {rows} tweets", flush=True)
    return {"rows": rows, "bad_dates": bad}


def load_users(group: str) -> pd.DataFrame:
    path = RAW / f"{group}.csv" / "users.csv"
    users = pd.read_csv(path, dtype=str, keep_default_na=False, encoding_errors="replace")
    for col in USER_COLS:
        if col not in users:
            users[col] = ""
    users = users[USER_COLS].copy()
    users["user_id"] = _id(users.pop("id"))
    users["label"] = GROUPS[group][0]
    users["group"] = GROUPS[group][1]
    users["account_created_at"] = _dates(users.pop("created_at"))
    users["crawled_at"] = pd.to_datetime(users.pop("timestamp"), errors="coerce", utc=True)
    for col in ("statuses_count", "followers_count", "friends_count", "favourites_count",
                "listed_count"):
        users[col] = pd.to_numeric(users[col], errors="coerce")
    for col in ("default_profile", "default_profile_image", "protected", "verified"):
        users[col] = users[col].map({"1": 1.0, "True": 1.0, "true": 1.0}).fillna(0.0)
    return users


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise SystemExit(f"{OUT} already prepared")
    OUT.mkdir(parents=True, exist_ok=True)
    users = pd.concat([load_users(g) for g in GROUPS], ignore_index=True)
    dup = users["user_id"].duplicated(keep=False)
    users.to_parquet(OUT / "users.parquet", index=False)

    stats = {}
    with pq.ParquetWriter(OUT / "tweets.parquet", TWEET_SCHEMA, compression="zstd") as writer:
        for group in GROUPS:
            stats[group] = convert_tweets(group, writer)

    tweets = pq.read_table(OUT / "tweets.parquet", columns=["user_id", "group", "created_at"]).to_pandas()
    period = tweets.groupby("group")["created_at"].quantile([0.01, 0.5, 0.99]).unstack()
    manifest = {
        "source": "https://botometer.osome.iu.edu/bot-repository/datasets/cresci-2017/cresci-2017.csv.zip",
        "archive_sha256": _sha256(Path("data/raw/cresci2017/cresci-2017.csv.zip")),
        "users": users.groupby("group").size().to_dict(),
        "duplicate_user_ids_across_groups": int(dup.sum()),
        "tweets": {GROUPS[g][1]: s for g, s in stats.items()},
        "users_with_tweets": tweets.groupby("group")["user_id"].nunique().to_dict(),
        "tweet_period_p01_p50_p99": {
            g: [str(v.date()) for v in row] for g, row in period.iterrows()
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
