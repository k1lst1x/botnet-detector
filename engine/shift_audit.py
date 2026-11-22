"""Aggregate train/validation distribution audit; never loads test arrays."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from .experiment import load_array_split
from .shard_index import file_hash


def distribution(values):
    finite = values[np.isfinite(values)]
    return {
        'rows': len(values),
        'observed': len(finite),
        'missing_rate': float(1 - len(finite)/len(values)) if len(values) else None,
        'quantiles': dict(zip(('min','p25','median','p75','max'), map(float,np.quantile(finite,[0,.25,.5,.75,1])))) if len(finite) else None,
    }


def summarize_shift(X, y, V, vy, names):
    if X.shape != (len(y), len(names)) or V.shape != (len(vy), len(names)):
        raise ValueError('Feature shape mismatch')
    if set(np.unique(y)) != {0,1} or set(np.unique(vy)) != {0,1}:
        raise ValueError('Both classes required in train and validation')
    report = {'scope':'train/val only; no test read', 'train':{'rows':len(y),'bots':int(y.sum()),'bot_rate':float(np.mean(y))}, 'val':{'rows':len(vy),'bots':int(vy.sum()),'bot_rate':float(np.mean(vy))}, 'train_positive_weight':float(np.sum(y==0)/np.sum(y==1)), 'features':{}}
    for i,name in enumerate(names):
        groups = {}
        for group,label in (('all',None),('human',0),('bot',1)):
            a = X[:,i] if label is None else X[y==label,i]
            b = V[:,i] if label is None else V[vy==label,i]
            af,bf = a[np.isfinite(a)],b[np.isfinite(b)]
            groups[group] = {'train':distribution(a),'val':distribution(b),'ks_statistic':float(ks_2samp(af,bf,method='exact' if min(len(af),len(bf)) < 2 else 'asymp').statistic) if len(af) and len(bf) else None}
        report['features'][name] = groups
    return report


def audit_arrays(directory, expected_manifest_hash):
    directory = Path(directory)
    if file_hash(directory/'manifest.json') != expected_manifest_hash:
        raise ValueError('Array manifest hash mismatch')
    manifest = json.loads((directory/'manifest.json').read_text())
    X,y = load_array_split(directory,manifest,'train')
    V,vy = load_array_split(directory,manifest,'val')
    result = summarize_shift(X,y,V,vy,manifest['feature_names'])
    result['arrays_manifest_sha256'] = expected_manifest_hash
    result['payload_sha256'] = {k:v for k,v in manifest.get('array_sha256',{}).items() if k.startswith(('train.','val.'))}
    return result
