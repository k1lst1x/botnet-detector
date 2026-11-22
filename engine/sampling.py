"""The same stable bottom-k post sampling used by the prepared shard."""
import hashlib


def sample_posts(posts, limit=100, seed=42):
    if limit < 1:
        raise ValueError('Sampling limit must be positive')
    def key(post):
        pid = post.external_post_id
        if post.platform == 'twitter' and pid.isdigit():
            pid = 't' + pid
        return hashlib.blake2b(f'{seed}:{pid}'.encode(), digest_size=8).digest(), pid
    return sorted(posts, key=key)[:limit]
