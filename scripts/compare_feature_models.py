"""Fixed six-run research comparison. Only train and validation are read."""
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pyarrow.parquet as pq
import catboost
import sklearn
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from engine.experiment import choose_thresholds, save_json
from engine.model import metrics
from engine.shard_index import file_hash


def main():
    config=json.loads(Path('experiments/feature-model-comparison.json').read_text())
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    save_json(out/'plan.json',config)
    source=Path(config['prepared_directory'])
    if file_hash(source/'manifest.json')!=config['prepared_manifest_sha256']:
        raise ValueError('Manifest changed')
    manifest=json.loads((source/'manifest.json').read_text())
    all_base=config['profile']+config['posts']
    names=all_base+[n+'__missing' for n in all_base]
    data={};seen=set();coverage={};hashes={}
    for split in ('train','val'):
        path=source/f'{split}.parquet';hashes[split]=file_hash(path)
        if hashes[split]!=manifest['parquet_sha256'][path.name]:raise ValueError('Parquet changed')
        table=pq.read_table(path,columns=['external_user_id','split','label']+names)
        ids=table['external_user_id'].to_pylist()
        if len(set(ids))!=len(ids) or seen.intersection(ids):raise ValueError('Duplicate identity')
        seen.update(ids)
        if set(table['split'].to_pylist())!={split}:raise ValueError('Split mismatch')
        X=np.column_stack([table[n].to_numpy() for n in names]).astype(np.float32)
        y=table['label'].to_numpy()
        if set(np.unique(y))!={0,1} or np.isinf(X).any():raise ValueError('Invalid values')
        if not np.array_equal(X[:,len(all_base):],np.isnan(X[:,:len(all_base)]).astype(float)):raise ValueError('Invalid masks')
        data[split]=(X,y)
        coverage[split]={n:float(np.isfinite(X[:,i]).mean()) for i,n in enumerate(all_base)}
    results={}
    for subset in ('profile','posts','combined'):
        base=all_base if subset=='combined' else config[subset]
        features=base+[n+'__missing' for n in base]
        indices=[names.index(n) for n in features]
        X,y=data['train'];V,vy=data['val']
        for algorithm in ('xgboost','catboost'):
            started=time.monotonic()
            estimator=xgboost.XGBClassifier(**config[algorithm]) if algorithm=='xgboost' else catboost.CatBoostClassifier(**config[algorithm])
            pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),estimator)
            pipe.fit(X[:,indices],y)
            scores=pipe.predict_proba(V[:,indices])[:,1]
            thresholds=choose_thresholds(vy,scores,.9,100)
            key=f'{subset}-{algorithm}'
            artifact=out/f'{key}.joblib'
            joblib.dump({'model':pipe,'features':features,'research_only':True,'decision_policy':'review_only','thresholds':thresholds},artifact)
            loaded=joblib.load(artifact)
            np.testing.assert_allclose(loaded['model'].predict_proba(V[:10,indices])[:,1],scores[:10])
            results[key]={'selection':thresholds,'best_f1_metrics':metrics(vy,scores,thresholds['best_f1']['threshold']),'precision_target_metrics':metrics(vy,scores,thresholds['precision_target']['threshold']) if thresholds['precision_target'] else None,'seconds':time.monotonic()-started,'artifact_sha256':file_hash(artifact)}
            print(key,json.dumps(results[key]),flush=True)
            save_json(out/'results.json',results)
    report={'plan':config,'plan_sha256':file_hash(out/'plan.json'),'code_sha256':file_hash(Path(__file__)),'versions':{'catboost':catboost.__version__,'xgboost':xgboost.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__},'source_sha256':hashes,'coverage':coverage,'rows':{s:len(y) for s,(X,y) in data.items()},'results':results,'scope':'validation only; six exploratory comparisons, no independent confirmation; test not read'}
    save_json(out/'report.json',report)


if __name__=='__main__':main()
