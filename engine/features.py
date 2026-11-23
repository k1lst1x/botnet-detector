"""Numeric features; no learned vocabulary, language detector or local timezone."""

from collections import Counter
from datetime import timezone
import math
import re
import statistics
import unicodedata
from urllib.parse import urlsplit

from .contracts import AccountSnapshot, Observation, Post

FEATURE_VERSION = "portable-v1"
PROFILE_FEATURES = [
    "followers_log1p",
    "following_log1p",
    "followers_following_ratio",
    "bio_present",
    "avatar_present",
    "account_age_days",
]
PORTABLE_FEATURES = [
    "posts_per_day",
    "interval_median_s",
    "interval_p90_s",
    "interval_cv",
    "burstiness",
    "short_interval_ratio",
    "hour_entropy",
    "circadian_amplitude",
    "max_posts_per_hour",
    "active_day_ratio",
    "reply_ratio",
    "repost_ratio",
    "quote_ratio",
    "media_ratio",
    "duplicate_ratio",
    "text_length_median",
    "text_length_cv",
    "text_digit_ratio",
    "text_space_ratio",
    "text_punctuation_ratio",
    "url_ratio",
    "unique_domains",
    "domain_concentration",
    "unique_targets",
    "target_concentration",
]
BASE_FEATURES = PROFILE_FEATURES + PORTABLE_FEATURES
FEATURE_NAMES = BASE_FEATURES + [f"{name}__missing" for name in BASE_FEATURES]
URL_PATTERN = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)

# profile-v2: profile fields exposed by both Twitter and the Threads profile API.
# followers_log1p, bio_present and avatar_present are shared with portable-v1.
PROFILE_VERSION = "profile-v2"
PROFILE_EXTRA_FEATURES = [
    "bio_length",
    "bio_url_count",
    "bio_mention_count",
    "bio_hashtag_count",
    "bio_digit_ratio",
    "avatar_default",
    "verified",
    "name_length",
    "name_digit_ratio",
    "name_non_alnum_ratio",
    "username_length",
    "username_digit_ratio",
    "username_trailing_digits",
    "username_underscore_count",
    "name_username_overlap",
]
PROFILE_V2_FEATURES = [
    "followers_log1p", "bio_present", *PROFILE_EXTRA_FEATURES[:5],
    "avatar_present", *PROFILE_EXTRA_FEATURES[5:],
]
_BIO_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#\w+")


def _share(count, total):
    return count / total if total else 0.0


def name_username_overlap(name, username):
    """Share of username letters found, in order, in the display name."""
    a = re.sub(r"[^a-z]", "", name.lower())
    b = re.sub(r"[^a-z]", "", username.lower())
    if not a or not b:
        return 0.0
    i = 0
    for ch in a:
        if i < len(b) and b[i] == ch:
            i += 1
    return i / len(b)


def profile_text_features(biography, name, username, avatar_default, verified):
    """profile-v2 extras; None marks an unknown source field.

    The same function builds training rows (TwiBot-22) and inference rows.
    """
    f = dict.fromkeys(PROFILE_EXTRA_FEATURES)
    if biography is not None:
        bio = unicodedata.normalize("NFKC", biography)
        f["bio_length"] = float(len(bio))
        f["bio_url_count"] = float(len(_BIO_URL.findall(bio)))
        f["bio_mention_count"] = float(len(_MENTION.findall(bio)))
        f["bio_hashtag_count"] = float(len(_HASHTAG.findall(bio)))
        f["bio_digit_ratio"] = _share(sum(c.isdigit() for c in bio), len(bio))
    if avatar_default is not None:
        f["avatar_default"] = float(avatar_default)
    if verified is not None:
        f["verified"] = float(verified)
    if name is not None:
        name = unicodedata.normalize("NFKC", name)
        f["name_length"] = float(len(name))
        f["name_digit_ratio"] = _share(sum(c.isdigit() for c in name), len(name))
        f["name_non_alnum_ratio"] = _share(
            sum(not (c.isalnum() or c.isspace()) for c in name), len(name)
        )
    if username is not None:
        f["username_length"] = float(len(username))
        f["username_digit_ratio"] = _share(sum(c.isdigit() for c in username), len(username))
        f["username_trailing_digits"] = float(len(username) - len(username.rstrip("0123456789")))
        f["username_underscore_count"] = float(username.count("_"))
    if name is not None and username is not None:
        f["name_username_overlap"] = name_username_overlap(name, username)
    return f


def extract_profile_features(account: AccountSnapshot):
    return profile_text_features(
        account.biography,
        account.display_name,
        account.username,
        account.avatar_default,
        account.is_verified,
    )


def parsed_url(value):
    try:
        parts = urlsplit(value.rstrip(".,;!?)"))
        return parts if parts.hostname and parts.scheme in ("http", "https") else None
    except ValueError:
        return None


def _mean(values):
    return statistics.fmean(values) if values else None


def _concentration(items):
    counts = Counter(items)
    total = sum(counts.values())
    return sum((n / total) ** 2 for n in counts.values()) if total else None


def _cv(values):
    mean = _mean(values)
    return (
        statistics.pstdev(values) / mean
        if len(values) > 1 and mean and mean > 0
        else None
    )


def extract_features(
    account: AccountSnapshot, posts: list[Post], observation: Observation
):
    if any((p.platform, p.author_id) != account.key for p in posts):
        raise ValueError("Features require one account identity")
    f = dict.fromkeys(BASE_FEATURES)
    for attr in ("followers", "following"):
        value = getattr(account, attr)
        f[f"{attr}_log1p"] = math.log1p(value) if value is not None else None
    if account.followers is not None and account.following is not None:
        f["followers_following_ratio"] = account.followers / (account.following + 1)
    f["bio_present"] = (
        float(bool(account.biography.strip()))
        if account.biography is not None
        else None
    )
    f["avatar_present"] = (
        float(account.has_avatar) if account.has_avatar is not None else None
    )
    if (
        account.created_at
        and account.collected_at
        and account.collected_at >= account.created_at
    ):
        f["account_age_days"] = (
            account.collected_at - account.created_at
        ).total_seconds() / 86400
    if posts:
        for flag, name in [
            ("is_reply", "reply_ratio"),
            ("is_repost", "repost_ratio"),
            ("is_quote", "quote_ratio"),
            ("has_media", "media_ratio"),
        ]:
            f[name] = _mean(
                [float(getattr(p, flag)) for p in posts if getattr(p, flag) is not None]
            )
        texts = [
            unicodedata.normalize("NFKC", p.text) for p in posts if p.text is not None
        ]
        lengths = [len(t) for t in texts]
        f["text_length_median"] = statistics.median(lengths) if lengths else None
        f["text_length_cv"] = _cv(lengths)
        nonempty = [" ".join(t.casefold().split()) for t in texts if t.strip()]
        f["duplicate_ratio"] = (
            1 - len(set(nonempty)) / len(nonempty) if nonempty else None
        )
        chars = "".join(texts)
        if chars:
            f["text_digit_ratio"] = sum(c.isdigit() for c in chars) / len(chars)
            f["text_space_ratio"] = sum(c.isspace() for c in chars) / len(chars)
            f["text_punctuation_ratio"] = sum(
                unicodedata.category(c).startswith("P") for c in chars
            ) / len(chars)
        url_lists = [
            p.urls
            if p.urls is not None
            else URL_PATTERN.findall(p.text)
            if p.text is not None
            else None
            for p in posts
        ]
        known_urls = [urls for urls in url_lists if urls is not None]
        f["url_ratio"] = _mean([float(bool(urls)) for urls in known_urls])
        domains = [
            parts.hostname.lower()
            for urls in known_urls
            for url in urls
            if (parts := parsed_url(url)) is not None
        ]
        f["unique_domains"] = float(len(set(domains))) if known_urls else None
        f["domain_concentration"] = _concentration(domains)
        targets = [p.targets for p in posts if p.targets is not None]
        f["unique_targets"] = (
            float(len({t for values in targets for t in values})) if targets else None
        )
        f["target_concentration"] = _concentration(
            [t for values in targets for t in values]
        )
        if observation.complete and not observation.truncated:
            ts = sorted(p.created_at.timestamp() for p in posts)
            intervals = [b - a for a, b in zip(ts, ts[1:])]
            span = (observation.end - observation.start).total_seconds()
            f["posts_per_day"] = len(posts) / (span / 86400)
            f["max_posts_per_hour"] = float(
                max(Counter(int(t // 3600) for t in ts).values())
            )
            f["active_day_ratio"] = len(
                {p.created_at.astimezone(timezone.utc).date() for p in posts}
            ) / ((observation.end.date() - observation.start.date()).days + 1)
            hours = Counter(p.created_at.astimezone(timezone.utc).hour for p in posts)
            f["hour_entropy"] = -sum(
                (c / len(posts)) * math.log2(c / len(posts)) for c in hours.values()
            ) / math.log2(24)
            f["circadian_amplitude"] = abs(
                sum(
                    complex(
                        math.cos(2 * math.pi * h / 24), math.sin(2 * math.pi * h / 24)
                    )
                    * c
                    for h, c in hours.items()
                )
            ) / len(posts)
            if intervals:
                ordered = sorted(intervals)
                f["interval_median_s"] = statistics.median(intervals)
                f["interval_p90_s"] = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
                mu = statistics.fmean(intervals)
                sd = statistics.pstdev(intervals)
                f["interval_cv"] = sd / mu if mu > 0 else None
                f["burstiness"] = (sd - mu) / (sd + mu) if sd + mu > 0 else None
                f["short_interval_ratio"] = sum(t < 60 for t in intervals) / len(
                    intervals
                )
    f.update({f"{k}__missing": float(v is None) for k, v in list(f.items())})
    return f
