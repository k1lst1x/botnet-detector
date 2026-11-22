"""Recover graph tokens, construct fixed-train-reference features, evaluate once."""
import json,hashlib,sqlite3,time
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit,urlunsplit
import ijson,joblib,numpy as np,pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from engine.shard_index import file_hash,HashReader
from engine.post_metadata import metadata
from engine.graph_features import reference_features
from engine.experiment import save_json
from engine.model import metrics
from scripts.error_driven_experiment import queue
from scripts.recover_metadata_experiment import SOURCE,EXPECTED

ROOT=Path('media/models/experiments/graph-v6')
CHANNELS=['url','target','url_time','target_time']


def token(value):return hashlib.sha256(value.encode()).digest()


def main():
    config=json.loads(Path('experiments/feature-model-comparison.json').read_text());prepared=Path(config['prepared_directory'])
    assert file_hash(prepared/'manifest.json')==config['prepared_manifest_sha256']
    manifest=json.loads((prepared/'manifest.json').read_text());base=config['profile'];names=base+[n+'__missing' for n in base];tables=[]
    for split in ('train','val'):
        p=prepared/f'{split}.parquet';assert file_hash(p)==manifest['parquet_sha256'][p.name]
        tables.append(pq.read_table(p,columns=names+['external_user_id','label','observed_posts']))
    ids=sum([t['external_user_id'].to_pylist() for t in tables],[]);assert len(set(ids))==len(ids)
    cut=len(tables[0]);counts=np.concatenate([t['observed_posts'].to_numpy() for t in tables]);id_hash=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    identity={'runner_sha256':file_hash(Path(__file__)),'ids_sha256':id_hash,'prepared_manifest':config['prepared_manifest_sha256'],'raw':EXPECTED,'metadata_code':file_hash(Path('engine/post_metadata.py')),'channels':CHANNELS,'bucket_seconds':600,'max_members':60,'index_state':file_hash(Path('data/processed/twibot22-shard7-index-v1/state.json'))}
    if ROOT.exists():
        saved_plan=json.loads((ROOT/'plan.json').read_text())
        assert saved_plan['identity']==identity
        assert saved_plan['parameters']==config['xgboost']
        cache=json.loads((ROOT/'tokens-manifest.json').read_text());assert file_hash(ROOT/'tokens.joblib')==cache['sha256']
        channels=joblib.load(ROOT/'tokens.joblib')
    else:
        ROOT.mkdir(parents=True);save_json(ROOT/'plan.json',{'identity':identity,'parameters':config['xgboost'],'design_sha256':file_hash(Path('docs/superpowers/plans/2026-11-20-graph-v6.md'))})
        mapping={};dbpath=Path('data/processed/twibot22-shard7-index-v1/tweets.sqlite3').resolve()
        with sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True) as db:
            for i,uid in enumerate(ids):
                rows=list(db.execute('SELECT id FROM tweets WHERE author=? ORDER BY sample_key,id LIMIT 100',(uid,)))
                assert len(rows)==counts[i]
                for row in rows:mapping[row[0]]=i
        print(f'Selected {len(mapping)} posts',flush=True)
        channels=[[set() for _ in ids] for _ in CHANNELS];records=0;selected=0;invalid=0;start=time.monotonic()
        with SOURCE.open('rb') as stream:
            reader=HashReader(stream)
            for raw in ijson.items(reader,'item',use_float=True):
                records+=1;pid=str(raw['id']);pid='t'+pid if pid.isdigit() else pid;i=mapping.pop(pid,None)
                if i is not None:
                    selected+=1
                    try:m=metadata(raw)
                    except (ValueError,TypeError):m={'urls':None,'targets':None};invalid+=1
                    stamp=int(datetime.fromisoformat(raw['created_at'].replace('Z','+00:00')).timestamp()//600)
                    for url in m['urls'] or []:
                        p=urlsplit(url);u=urlunsplit((p.scheme.lower(),p.netloc.lower(),p.path,p.query,''));channels[0][i].add(token(u));channels[2][i].add(token(f'{stamp}:{u}'))
                    for target in m['targets'] or []:
                        channels[1][i].add(token(target));channels[3][i].add(token(f'{stamp}:{target}'))
                if records%1000000==0:print(f'{records} records, {selected} selected, {time.monotonic()-start:.0f}s',flush=True)
            assert reader.digest.hexdigest()==EXPECTED
        assert not mapping
        joblib.dump(channels,ROOT/'tokens.joblib',compress=3)
        save_json(ROOT/'tokens-manifest.json',{'sha256':file_hash(ROOT/'tokens.joblib'),'raw_records':records,'selected':selected,'invalid':invalid,'identity':identity})
    if (ROOT/'results.json').exists():raise ValueError('Already evaluated')
    blocks=[]
    for name,values in zip(CHANNELS,channels):
        print('Graph features '+name,flush=True)
        train=reference_features(values[:cut],values[:cut],training=True);val=reference_features(values[:cut],values[cut:]);blocks.append(np.concatenate([train,val]))
    G=np.column_stack(blocks);np.save(ROOT/'graph.npy',G)
    X=np.concatenate([np.column_stack([t[n].to_numpy() for n in names]) for t in tables]);y=np.concatenate([t['label'].to_numpy() for t in tables]);vy=y[cut:]
    old=Path('media/models/experiments/feature-model-comparison-v3');report=json.loads((old/'report.json').read_text());assert file_hash(old/'profile-xgboost.joblib')==report['results']['profile-xgboost']['artifact_sha256'];b=joblib.load(old/'profile-xgboost.joblib');p=b['model'].predict_proba(X[cut:])[:,1];_,results_base=queue(vy,p,ids[cut:]);results={'active_profile':results_base}
    v5=Path('media/models/experiments/recovered-metadata-v5');ex=json.loads((v5/'extraction.json').read_text());assert ex['ids_sha256']==id_hash and file_hash(v5/'metadata.npy')==ex['metadata_sha256'];M=np.load(v5/'metadata.npy');M=np.column_stack([M,np.isnan(M).astype(float)])
    for name,A in [('profile_graph',np.column_stack([X,G])),('profile_metadata_graph',np.column_stack([X,M,G]))]:
        pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),XGBClassifier(**config['xgboost']));pipe.fit(A[:cut],y[:cut]);p=pipe.predict_proba(A[cut:])[:,1];_,r=queue(vy,p,ids[cut:]);r['ranking_metrics']=metrics(vy,p,.5);results[name]=r;joblib.dump({'model':pipe,'research_only':True,'schema':name,'reference_sha256':file_hash(ROOT/'tokens.joblib')},ROOT/f'{name}.joblib')
        np.testing.assert_allclose(joblib.load(ROOT/f'{name}.joblib')['model'].predict_proba(A[cut:cut+10])[:,1],p[:10])
    coverage={name:{split:{'any_eligible':float(np.mean(block[:,0]>0)),'repeated_peer':float(np.mean(block[:,1]>0))} for split,block in [('train',B[:cut]),('val',B[cut:])]} for name,B in zip(CHANNELS,blocks)}
    report={'results':results,'coverage':coverage,'identity':identity,'scope':'fixed train reference; exploratory validation only; no deployment','artifacts_sha256':{p.name:file_hash(p) for p in ROOT.glob('*.joblib')}}
    save_json(ROOT/'results.json',report);save_json(Path('docs/reports/2026-11-20-graph-v6.json'),report);print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
