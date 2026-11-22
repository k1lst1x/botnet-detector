import numpy as np
import pytest
from engine.shift_audit import summarize_shift, audit_arrays


def test_class_conditional_shift_distinguishes_class_mix_from_feature_shift():
    X=np.array([[0],[0],[10],[10]],dtype=float); y=np.array([0,0,1,1])
    V=np.array([[0],[10],[10],[10]],dtype=float); vy=np.array([0,1,1,1])
    r=summarize_shift(X,y,V,vy,['text_length_median'])
    assert r['train_positive_weight']==1
    assert r['features']['text_length_median']['all']['ks_statistic']==.25
    assert r['features']['text_length_median']['human']['ks_statistic']==0
    assert r['features']['text_length_median']['bot']['ks_statistic']==0


def test_missing_values_are_reported_and_absent_comparison_is_null():
    X=np.array([[np.nan],[np.nan]]);y=np.array([0,1])
    V=np.array([[2.],[np.nan]]);vy=np.array([0,1])
    r=summarize_shift(X,y,V,vy,['domain_concentration'])
    assert r['features']['domain_concentration']['all']['train']['missing_rate']==1
    assert r['features']['domain_concentration']['all']['ks_statistic'] is None
    assert r['features']['domain_concentration']['all']['train']['quantiles'] is None


def test_auditor_reads_only_train_and_validation(monkeypatch,tmp_path):
    import json
    import engine.shift_audit as module
    from engine.shard_index import file_hash
    (tmp_path/'manifest.json').write_text(json.dumps({'feature_names':['text_length_median']}))
    calls=[]
    def loader(directory,manifest,split):
        calls.append(split)
        assert split in ('train','val')
        return np.array([[1.],[2.]]),np.array([0,1])
    monkeypatch.setattr(module,'load_array_split',loader)
    audit_arrays(tmp_path,file_hash(tmp_path/'manifest.json'))
    assert calls==['train','val']
    with pytest.raises(ValueError,match='manifest'):
        audit_arrays(tmp_path,'wrong')
