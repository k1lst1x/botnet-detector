"""Package verified profile XGBoost for a manual review queue; no fitting."""
import json
from pathlib import Path
import joblib
import numpy as np
import pyarrow.parquet as pq
from engine.experiment import choose_thresholds, save_json
from engine.model import ModelService, metrics
from engine.features import FEATURE_VERSION
from engine.shard_index import file_hash


def main():
    source=Path('media/models/experiments/feature-model-comparison-v3')
    report=json.loads((source/'report.json').read_text())
    artifact=source/'profile-xgboost.joblib'
    assert file_hash(artifact)==report['results']['profile-xgboost']['artifact_sha256']
    path=Path('data/processed/twibot22-shard7-training-v1/val.parquet')
    assert file_hash(path)==report['source_sha256']['val']
    b=joblib.load(artifact);names=b['features']
    t=pq.read_table(path,columns=['label']+names)
    X=np.column_stack([t[n].to_numpy() for n in names]);y=t['label'].to_numpy()
    scores=b['model'].predict_proba(X)[:,1]
    target=choose_thresholds(y,scores,.6,100)['precision_target']
    if target is None:raise ValueError('Review target unavailable')
    bundle={'model':b['model'],'features':names,'feature_version':FEATURE_VERSION,'feature_set':'profile','model_version':'profile-review-mvp-v1','threshold':target['threshold'],'required_observed_features':[n for n in names if not n.endswith('__missing')],'minimum_observed_ratio':1.0,'minimum_posts':0,'decision_policy':'review_queue','calibrated':False,'threads_quality_validated':False}
    out=Path('media/models/review-mvp-v1');out.mkdir(parents=True,exist_ok=False)
    joblib.dump(bundle,out/'model.joblib')
    assert ModelService(out/'model.joblib').status()['status']=='ready'
    save_json(out/'report.json',{'source_sha256':file_hash(artifact),'model_sha256':file_hash(out/'model.joblib'),'policy':'manual review, all five profile fields required; no post minimum because model uses profile only','validation':metrics(y,scores,target['threshold']),'selection':target,'queue_fraction':target['predicted_positive']/len(y),'independent_evaluation':False,'threads_quality_validated':False})


if __name__=='__main__':main()
