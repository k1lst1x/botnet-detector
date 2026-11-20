"""Error audit and fixed-capacity validation ablation; no test reads."""
import json
from pathlib import Path
import hashlib
import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from engine.experiment import save_json
from engine.shard_index import file_hash
from engine.research_features import enhance
from engine.model import metrics


def queue(y,p,ids,k=2520):
    tie=np.array([hashlib.sha256(str(i).encode()).hexdigest() for i in ids])
    idx=np.lexsort((tie,-p))[:k]
    selected=np.zeros(len(y),dtype=bool);selected[idx]=True
    tp=int(y[selected].sum())
    return selected,{'capacity':k,'tp':tp,'fp':k-tp,'precision':tp/k,'recall':tp/int(y.sum())}


def main():
    root=Path('media/models/experiments/error-driven-v4');root.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('experiments/feature-model-comparison.json').read_text())
    save_json(root/'plan.json',{'parent':config,'variants':{'enhanced4':{'n_estimators':300,'max_depth':4},'enhanced6':{'n_estimators':500,'max_depth':6}},'capacity':2520,'scope':'validation only','design_sha256':file_hash(Path('docs/superpowers/plans/2026-11-20-error-driven-experiment.md'))})
    source=Path(config['prepared_directory']);assert file_hash(source/'manifest.json')==config['prepared_manifest_sha256']
    manifest=json.loads((source/'manifest.json').read_text());base=config['profile']+config['posts'];names=base+[n+'__missing' for n in base];data={}
    for split in ('train','val'):
        path=source/f'{split}.parquet';assert file_hash(path)==manifest['parquet_sha256'][path.name]
        t=pq.read_table(path,columns=names+['label','external_user_id','observed_posts'])
        data[split]=(np.column_stack([t[n].to_numpy() for n in names]),t['label'].to_numpy(),t['external_user_id'].to_numpy(),t['observed_posts'].to_numpy())
    X,y,ids,count=data['train'];V,vy,vids,vcount=data['val'];assert not set(ids)&set(vids)
    results={};predictions={};selections={}
    oldroot=Path('media/models/experiments/feature-model-comparison-v3');oldreport=json.loads((oldroot/'report.json').read_text())
    for variant in ('profile-xgboost','combined-xgboost'):
        path=oldroot/f'{variant}.joblib';assert file_hash(path)==oldreport['results'][variant]['artifact_sha256'];b=joblib.load(path)
        p=b['model'].predict_proba(V[:,[names.index(n) for n in b['features']]])[:,1]
        selections[variant],results[variant]=queue(vy,p,vids);predictions[variant]=p
    audit={};old=predictions['profile-xgboost']>=.407402902841568
    groups={'followers':np.expm1(V[:,names.index('followers_log1p')]),'following':np.expm1(V[:,names.index('following_log1p')]),'bio':V[:,names.index('bio_present')],'avatar':V[:,names.index('avatar_present')],'sample_posts':vcount}
    for name,values in groups.items():
        edges=[-np.inf,.5,np.inf] if name in ('bio','avatar') else [-np.inf,100,1000,10000,np.inf] if name in ('followers','following') else [-np.inf,30,60,100,np.inf]
        rows=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=(values>=lo)&(values<hi);n=int(mask.sum());hum=int(np.sum(mask&(vy==0)));bot=int(np.sum(mask&(vy==1)));fp=int(np.sum(mask&old&(vy==0)));fn=int(np.sum(mask&~old&(vy==1)))
            rows.append({'range':f'[{lo},{hi})','n':n,'humans':hum,'bots':bot,'fp':fp,'fn':fn,'fpr':fp/hum if hum else None,'fnr':fn/bot if bot else None})
        audit[name]=rows
    E,en=enhance(X,names,count);EV,_=enhance(V,names,vcount)
    for key,trees,depth in [('enhanced4',300,4),('enhanced6',500,6)]:
        params=config['xgboost']|{'n_estimators':trees,'max_depth':depth}
        pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),XGBClassifier(**params));pipe.fit(E,y)
        p=pipe.predict_proba(EV)[:,1];selections[key],results[key]=queue(vy,p,vids);predictions[key]=p
        joblib.dump({'model':pipe,'features':en,'research_only':True},root/f'{key}.joblib')
    rng=np.random.default_rng(42)
    for key,p in predictions.items():
        results[key]['ranking_metrics']=metrics(vy,p,.5)
        delta=[]
        for _ in range(1000):
            idx=rng.integers(0,len(vy),len(vy));a=selections[key][idx];b=selections['profile-xgboost'][idx]
            delta.append(float(vy[idx][a].mean()-vy[idx][b].mean()))
        results[key]['paired_bootstrap_precision_delta_95']=np.quantile(delta,[.025,.975]).tolist()
    report={'scope':'exploratory same-validation comparison, no test, no deployment','audit':audit,'results':results,'plan_sha256':file_hash(root/'plan.json'),'code_sha256':file_hash(Path(__file__)),'source_manifest_sha256':config['prepared_manifest_sha256'],'limitations':['derived features add representation, not new observations','no target metadata; temporal features unavailable','bootstrap conditions on chosen validation queues, not selection-adjusted or independent confirmation']}
    save_json(root/'report.json',report);save_json(Path('docs/reports/2026-11-20-error-driven.json'),report)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
