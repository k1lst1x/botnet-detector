from engine.inference import review_decision


def test_review_threshold_does_not_exonerate_low_scores():
    assert review_decision(.6,.6)=='priority_review'
    assert review_decision(.599,.6)=='standard_review'


def test_real_review_artifact_requires_complete_profile_and_scores_without_posts():
    from pathlib import Path
    import pytest
    from engine.model import ModelService
    from engine.contracts import Dataset
    from engine.inference import analyze_dataset
    from tests.test_engine_features import sample
    path=Path('media/models/review-mvp-v1/model.joblib')
    if not path.exists():pytest.skip('Local trained artifact unavailable')
    model=ModelService(path)
    a,_,obs=sample(False)
    missing=analyze_dataset(Dataset(accounts=[a],posts=[],observation=obs),model)['results'][0]
    assert missing['decision']=='insufficient_data'
    a=a.model_copy(update={'followers':100,'following':200,'biography':'test','has_avatar':True})
    result=analyze_dataset(Dataset(accounts=[a],posts=[],observation=obs),model)['results'][0]
    assert result['decision'] in ('priority_review','standard_review')
    assert result['score'] is not None and result['explanation']


def test_tabular_import_preserves_avatar_false_and_missing():
    import json
    from engine.datasets import read_import
    row={'author_id':'a','post_id':'p','created_at':'2026-01-01T00:00:00Z','followers':100,'following':200,'biography':'','has_avatar':False}
    assert read_import(json.dumps(row),'jsonl').accounts[0].has_avatar is False
    csv='author_id,post_id,created_at,followers,following,biography,has_avatar\na,p,2026-01-01T00:00:00Z,100,200,,true\n'
    assert read_import(csv,'csv').accounts[0].has_avatar is True
    row.pop('has_avatar')
    assert read_import(json.dumps(row),'jsonl').accounts[0].has_avatar is None
