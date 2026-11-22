"""Merge observations conservatively; observed history is not complete history."""
from collections import defaultdict
from .contracts import Dataset, Observation


def merge_observations(datasets):
    datasets=list(datasets)
    if not datasets:
        raise ValueError('No observations supplied')
    accounts,posts={},{}
    snapshots={}
    duplicates=0
    for data in datasets:
        for account in data.accounts:
            snapshot_key=(account.key,account.collected_at)
            previous=snapshots.get(snapshot_key)
            if previous is not None and previous != account:
                raise ValueError("Conflicting account snapshot at identical collection time")
            snapshots[snapshot_key]=account
            old=accounts.get(account.key)
            if old is not None and old != account:
                if old.collected_at is None or account.collected_at is None or old.collected_at==account.collected_at:
                    raise ValueError('Conflicting account snapshot without unique collection time')
                if old.collected_at>account.collected_at:
                    continue
            accounts[account.key]=account
        for post in data.posts:
            key=post.platform,post.external_post_id
            if key in posts:
                if posts[key]!=post:
                    raise ValueError('Conflicting post; reconcile edited/enriched records explicitly')
                duplicates+=1
            posts[key]=post
    observation=Observation(start=min(d.observation.start for d in datasets),end=max(d.observation.end for d in datasets),source='merged_observations',complete=False,truncated=any(d.observation.truncated for d in datasets))
    merged=Dataset(accounts=sorted(accounts.values(),key=lambda a:a.key),posts=sorted(posts.values(),key=lambda p:(p.created_at,p.platform,p.external_post_id)),observation=observation)
    grouped=defaultdict(list)
    for p in merged.posts:grouped[(p.platform,p.author_id)].append(p)
    summaries=[]
    for account in merged.accounts:
        items=grouped[account.key]
        summaries.append({'platform':account.platform,'external_user_id':account.external_user_id,'posts':len(items),'first_observed_post':items[0].created_at.isoformat() if items else None,'last_observed_post':items[-1].created_at.isoformat() if items else None,'posts_with_known_targets':sum(p.targets is not None for p in items),'posts_with_explicit_urls':sum(p.urls is not None for p in items),'identity_source':account.identity_source})
    report={'inputs':len(datasets),'duplicate_posts':duplicates,'unique_posts':len(posts),'accounts':summaries,'source_observations':[d.observation.model_dump(mode='json') for d in datasets],'temporal_training_ready':False,'warnings':['merged coverage is not certified; missing intervals may exist','latest profile must not be used as a historical snapshot for all posts','username fallback identities may change or refer to different owners']}
    return merged,report
