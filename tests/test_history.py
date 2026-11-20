import pytest
from engine.history import merge_observations
from tests.test_engine_features import sample
from engine.contracts import Dataset


def dataset():
    a,p,o=sample(False)
    return Dataset(accounts=[a],posts=p,observation=o)


def test_duplicate_imports_do_not_inflate_posts_or_certify_coverage():
    d=dataset();merged,report=merge_observations([d,d])
    assert len(merged.posts)==len(d.posts)
    assert report['duplicate_posts']==len(d.posts)
    assert not merged.observation.complete
    assert report['temporal_training_ready'] is False


def test_conflicting_post_is_rejected():
    d=dataset();other=d.model_copy(deep=True);other.posts[0].text='changed'
    with pytest.raises(ValueError,match='post'):merge_observations([d,other])


def test_profile_conflict_without_collection_timestamp_is_rejected():
    d=dataset();other=d.model_copy(deep=True);other.accounts[0].followers=999
    d.accounts[0].collected_at=None;other.accounts[0].collected_at=None
    with pytest.raises(ValueError,match='snapshot'):merge_observations([d,other])


def test_latest_snapshot_wins_independent_of_input_order():
    from datetime import datetime, timezone, timedelta
    a=dataset();b=a.model_copy(deep=True)
    a.accounts[0].collected_at=datetime(2026,1,1,tzinfo=timezone.utc)
    b.accounts[0].collected_at=a.accounts[0].collected_at+timedelta(days=1)
    b.accounts[0].followers=999
    x,_=merge_observations([a,b]);y,_=merge_observations([b,a])
    assert x==y and x.accounts[0].followers==999


def test_disjoint_complete_windows_do_not_fill_gap():
    from datetime import timedelta
    a=dataset();a.observation.complete=True
    b=a.model_copy(deep=True);b.observation.start+=timedelta(days=10);b.observation.end+=timedelta(days=10)
    for p in b.posts:p.created_at+=timedelta(days=10);p.external_post_id+='new'
    result,_=merge_observations([a,b])
    assert not result.observation.complete


def test_old_snapshot_conflicts_rejected_after_newer_snapshot():
    from datetime import datetime,timezone,timedelta
    import itertools
    a=dataset();b=a.model_copy(deep=True);c=a.model_copy(deep=True)
    stamp=datetime(2026,1,1,tzinfo=timezone.utc)
    a.accounts[0].collected_at=stamp;b.accounts[0].collected_at=stamp
    b.accounts[0].followers=999
    c.accounts[0].collected_at=stamp+timedelta(days=1)
    for order in itertools.permutations([a,b,c]):
        with pytest.raises(ValueError,match='snapshot'):merge_observations(order)
