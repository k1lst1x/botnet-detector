import json

import numpy as np
import pytest

from engine.experiment import choose_thresholds, load_array_split
from engine.inference import analyze_dataset
from engine.sampling import sample_posts
from tests.test_engine_features import sample


def test_thresholds_respect_support_and_unattainable_precision_is_null():
    y=np.array([0,0,1,1]);scores=np.array([.1,.8,.9,.7])
    result=choose_thresholds(y,scores,min_precision=.9,min_predictions=2)
    assert result['precision_target'] is None
    assert result['best_f1']['threshold']==.7
    assert result['best_f1']['recall_bot']==1
    assert result['best_f1']['precision_bot']==pytest.approx(2/3)


def test_tied_scores_are_all_included_in_threshold_support():
    result=choose_thresholds(np.array([0,1,1]),np.array([.1,.8,.8]),min_precision=.9,min_predictions=2)
    assert result['precision_target']['threshold']==.8
    assert result['precision_target']['predicted_positive']==2


def test_unknown_or_modified_array_is_rejected_before_loading(tmp_path):
    from engine.shard_index import file_hash
    np.save(tmp_path/'train.X.npy',np.array([[1.,0.],[2.,0.]],dtype=np.float32))
    np.save(tmp_path/'train.y.npy',np.array([0,1],dtype=np.uint8))
    manifest={'feature_names':['text_length_median','text_length_median__missing'],'splits':{'train':{'rows':2,'classes':{'human':1,'bot':1}}},'array_sha256':{p.name:file_hash(p) for p in tmp_path.glob('*.npy')}}
    X,y=load_array_split(tmp_path,manifest,'train')
    assert X.shape==(2,2) and y.tolist()==[0,1]
    with pytest.raises(ValueError,match='split'):
        load_array_split(tmp_path,manifest,'support')
    np.save(tmp_path/'train.y.npy',np.array([1,1],dtype=np.uint8))
    with pytest.raises(ValueError,match='hash'):
        load_array_split(tmp_path,manifest,'train')


def test_sampling_is_source_order_independent_and_does_not_mutate_posts():
    _,posts,_=sample()
    a=sample_posts(posts,limit=2,seed=42)
    b=sample_posts(list(reversed(posts)),limit=2,seed=42)
    assert [p.external_post_id for p in a]==[p.external_post_id for p in b]
    assert len(a)==2 and len(posts)==3


def test_candidate_model_refuses_too_few_posts_before_prediction():
    account,posts,obs=sample(False)
    from engine.contracts import Dataset
    class Candidate:
        bundle={'minimum_posts':20,'required_observed_features':['text_length_median'],'minimum_observed_ratio':.5}
        def status(self):return {'status':'ready','model_version':'candidate'}
        def predict(self,rows):raise AssertionError('too few posts must not score')
    result=analyze_dataset(Dataset(accounts=[account],posts=posts,observation=obs),Candidate())
    assert result['results'][0]['decision']=='insufficient_data'
    assert result['results'][0]['score'] is None


def test_sampling_matches_preparation_hash_recipe():
    import hashlib
    _,posts,_=sample()
    expected=sorted(posts,key=lambda p:(hashlib.blake2b(f'42:{p.external_post_id}'.encode(),digest_size=8).digest(),p.external_post_id))[:2]
    assert sample_posts(posts,2,42)==expected


def test_evaluation_rejects_changed_frozen_artifact_before_arrays(tmp_path):
    from engine.experiment import evaluate
    (tmp_path/'model.joblib').write_bytes(b'changed')
    (tmp_path/'frozen.json').write_text(json.dumps({'frozen_sha256':{'model.joblib':'incorrect'}}))
    with pytest.raises(ValueError,match='Frozen artifact'):
        evaluate(tmp_path)


def test_twitter_native_and_twibot_post_ids_select_same_subset():
    _,posts,_=sample()
    native=[posts[0].model_copy(update={'platform':'twitter','external_post_id':str(i)}) for i in range(150)]
    canonical=[p.model_copy(update={'external_post_id':'t'+p.external_post_id}) for p in native]
    assert ['t'+p.external_post_id for p in sample_posts(native)]==[p.external_post_id for p in sample_posts(canonical)]


def test_validation_only_experiment_rejects_test_evaluation(tmp_path):
    from engine.experiment import evaluate
    from engine.shard_index import file_hash
    (tmp_path/'plan.json').write_text(json.dumps({'evaluation_policy':'validation_only; previously viewed test is not a fresh holdout'}))
    (tmp_path/'frozen.json').write_text(json.dumps({'frozen_sha256':{'plan.json':file_hash(tmp_path/'plan.json')}}))
    with pytest.raises(ValueError,match='validation.only'):
        evaluate(tmp_path)
