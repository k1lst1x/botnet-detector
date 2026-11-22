"""Recover metadata for identical selected train/val posts, then fit fixed candidates."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import ijson
import joblib
import numpy as np
import pyarrow.parquet as pq
from urllib.parse import urlsplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from engine.post_metadata import metadata
from engine.shard_index import HashReader,file_hash
from engine.features import URL_PATTERN
from engine.experiment import save_json
from engine.model import metrics
from scripts.error_driven_experiment import queue

ROOT=Path('media/models/experiments/recovered-metadata-v5')
SOURCE=Path('data/raw/twibot22-kaggle-shard7/tweet_7-004.json')
EXPECTED='5dd415d4fc8c6733f557657eaa3d8a3cfd215580786338bd93af766045d36036'
NEW=['entity_url_known_ratio','entity_target_known_ratio','entity_url_post_ratio','entity_domain_count','entity_domain_concentration','entity_url_repeat_ratio','entity_target_post_ratio','entity_target_count','entity_target_concentration']


def concentration(counter):
    n=sum(counter.values())
    return sum((v/n)**2 for v in counter.values()) if n else np.nan


def main():
    config=json.loads(Path('experiments/feature-model-comparison.json').read_text())
    prepared=Path(config['prepared_directory']);assert file_hash(prepared/'manifest.json')==config['prepared_manifest_sha256']
    manifest=json.loads((prepared/'manifest.json').read_text())
    names=config['profile']+config['posts'];names+= [n+'__missing' for n in names]
    tables=[]
    for split in ('train','val'):
        path=prepared/f'{split}.parquet';assert file_hash(path)==manifest['parquet_sha256'][path.name]
        tables.append(pq.read_table(path,columns=['external_user_id','label','observed_posts']+names))
    ids=[uid for table in tables for uid in table['external_user_id'].to_pylist()]
    assert len(set(ids))==len(ids)
    counts=np.concatenate([t['observed_posts'].to_numpy() for t in tables]);n=len(ids)
    identity={'prepared_manifest_sha256':config['prepared_manifest_sha256'],'features':NEW,'extractor_sha256':file_hash(Path('engine/post_metadata.py')),'selection':'bottom100 BLAKE2b seed42 canonical ID','index_state_sha256':file_hash(Path('data/processed/twibot22-shard7-index-v1/state.json'))}
    if ROOT.exists():
        cached=json.loads((ROOT/'plan.json').read_text())
        if cached.get('recovery_identity')!=identity:raise ValueError('Cached recovery identity changed')
    if not (ROOT/'metadata.npy').exists():
        ROOT.mkdir(parents=True,exist_ok=False)
        save_json(ROOT/'plan.json',{'recovery_identity':identity,'parent':config,'features':NEW,'capacity':2520,'source_sha256':EXPECTED,'design_sha256':file_hash(Path('docs/superpowers/plans/2026-11-20-recovered-metadata.md'))})
        mapping={}
        dbpath=Path('data/processed/twibot22-shard7-index-v1/tweets.sqlite3').resolve()
        with sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True) as db:
            for i,uid in enumerate(ids):
                posts=[row[0] for row in db.execute('SELECT id FROM tweets WHERE author=? ORDER BY sample_key,id LIMIT 100',(uid,))]
                assert len(posts)==counts[i]
                for pid in posts:mapping[pid]=i
        print(f'Selected {len(mapping)} posts for {n} accounts',flush=True)
        domains=[collections.Counter() for _ in ids];urls=[collections.Counter() for _ in ids];targets=[collections.Counter() for _ in ids]
        known=np.zeros((n,4),dtype=np.int64);audit=collections.Counter();start=time.monotonic()
        with SOURCE.open('rb') as f:
            reader=HashReader(f)
            for raw in ijson.items(reader,'item',use_float=True):
                audit['raw_records']+=1
                pid=str(raw['id']);pid='t'+pid if pid.isdigit() else pid
                i=mapping.pop(pid,None)
                if i is not None:
                    audit['selected_records']+=1
                    try:m=metadata(raw)
                    except (ValueError,TypeError):m={'urls':None,'targets':None};audit['invalid_metadata']+=1
                    if m['urls'] is not None:
                        known[i,0]+=1;known[i,1]+=bool(m['urls'])
                        audit['text_entity_count_disagreement']+=len(URL_PATTERN.findall(raw.get('text') or ''))!=len(m['urls'])
                        for u in m['urls']:
                            host=urlsplit(u).hostname.lower();domains[i][host]+=1
                            urls[i][hashlib.sha256(u.encode()).digest()]+=1
                    if m['targets'] is not None:
                        known[i,2]+=1;known[i,3]+=bool(m['targets'])
                        targets[i].update(m['targets'])
                if audit['raw_records']%500000==0:print(json.dumps(dict(audit)|{'seconds':round(time.monotonic()-start)}),flush=True)
            if reader.digest.hexdigest()!=EXPECTED:raise ValueError('Raw hash mismatch')
        if mapping:raise ValueError(f'Missing selected posts: {len(mapping)}')
        M=np.full((n,len(NEW)),np.nan)
        for i in range(n):
            uk,up,tk,tp=known[i];u=urls[i];d=domains[i];t=targets[i]
            M[i]=[uk/counts[i],tk/counts[i],up/uk if uk else np.nan,len(d) if uk else np.nan,concentration(d),1-len(u)/sum(u.values()) if u else np.nan,tp/tk if tk else np.nan,len(t) if tk else np.nan,concentration(t)]
        np.save(ROOT/'metadata.npy',M)
        save_json(ROOT/'extraction.json',{'audit':dict(audit),'rows':n,'train_rows':len(tables[0]),'ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'metadata_sha256':file_hash(ROOT/'metadata.npy'),'raw_sha256':EXPECTED})
    extraction=json.loads((ROOT/'extraction.json').read_text());assert file_hash(ROOT/'metadata.npy')==extraction['metadata_sha256'];assert hashlib.sha256('\n'.join(ids).encode()).hexdigest()==extraction['ids_sha256']
    if (ROOT/'results.json').exists():raise ValueError('Experiment already evaluated')
    M=np.load(ROOT/'metadata.npy');M=np.column_stack([M,np.isnan(M).astype(float)])
    A=np.concatenate([np.column_stack([t[name].to_numpy() for name in names]) for t in tables]);labels=np.concatenate([t['label'].to_numpy() for t in tables]);cut=len(tables[0]);vy=labels[cut:]
    results={};old=Path('media/models/experiments/feature-model-comparison-v3');b=joblib.load(old/'profile-xgboost.joblib');p=b['model'].predict_proba(A[cut:,[names.index(n) for n in b['features']]])[:,1]
    _,results['active_profile']=queue(vy,p,ids[cut:])
    for subset in ('profile','combined'):
        base=config['profile'] if subset=='profile' else config['profile']+config['posts'];features=base+[n+'__missing' for n in base];X=np.column_stack([A[:,[names.index(n) for n in features]],M])
        pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),XGBClassifier(**config['xgboost']));pipe.fit(X[:cut],labels[:cut]);p=pipe.predict_proba(X[cut:])[:,1]
        _,r=queue(vy,p,ids[cut:]);r['ranking_metrics']=metrics(vy,p,.5);results[subset+'_metadata']=r
        joblib.dump({'model':pipe,'features':features+NEW+[n+'__missing' for n in NEW],'research_only':True},ROOT/f'{subset}.joblib')
    save_json(ROOT/'results.json',results)
    save_json(Path('docs/reports/2026-11-20-recovered-metadata.json'),{'scope':'validation exploratory, no deployment','extraction':extraction,'results':results})
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
