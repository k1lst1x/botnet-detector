"""Additional numeric features for offline ablation; not a serving schema."""
import numpy as np


def enhance(X, names, observed_posts):
    col=lambda n:X[:,names.index(n)]
    followers=np.expm1(col('followers_log1p').astype(float))
    following=np.expm1(col('following_log1p').astype(float))
    urls=col('url_ratio')*observed_posts
    extra=np.column_stack([
        col('followers_log1p')-col('following_log1p'),
        np.log1p(followers+following),
        np.minimum(followers,following)/(np.maximum(followers,following)+1),
        col('unique_domains')/np.maximum(urls,1),
    ])
    new=['log_social_ratio','log_social_total','social_balance','domains_per_linked_post']
    return np.column_stack([X,extra]),names+new
