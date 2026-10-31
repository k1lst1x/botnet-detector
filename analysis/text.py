"""Text utilities: entity extraction and normalisation (ru / kk / en)."""
import re

URL_RE = re.compile(r"https?://[^\s]+", re.I)
HASHTAG_RE = re.compile(r"#([\w\u0400-\u04FF]+)", re.U)
MENTION_RE = re.compile(r"@([\w.]+)", re.U)
TOKEN_RE = re.compile(r"[\w\u0400-\u04FF]+", re.U)


def normalize_url(url: str) -> str:
    url = url.lower().rstrip(".,;:!?)»\"'")
    url = re.sub(r"^https?://(www\.)?", "", url)
    url = url.split("?")[0].split("#")[0].rstrip("/")
    return url


def extract_entities(text: str):
    text = text or ""
    urls = sorted({normalize_url(u) for u in URL_RE.findall(text)})
    hashtags = sorted({h.lower() for h in HASHTAG_RE.findall(text)})
    mentions = sorted({m.lower().rstrip(".") for m in MENTION_RE.findall(text)})
    return urls, hashtags, mentions


def clean_text(text: str) -> str:
    text = URL_RE.sub(" ", text or "")
    text = MENTION_RE.sub(" ", text)
    return " ".join(TOKEN_RE.findall(text.lower()))


def jaccard(a, b, shrink: float = 0.0) -> float:
    """Jaccard index. ``shrink`` > 0 damps similarity of tiny sets
    (one shared URL between two accounts with one URL each != coordination)."""
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / (len(a | b) + shrink)
