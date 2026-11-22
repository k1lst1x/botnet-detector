"""Frozen, train/validation/test separated structural model experiment."""
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from .features import FEATURE_NAMES, FEATURE_VERSION
from .model import metrics
from .shard_index import file_hash


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_array_split(directory, manifest, split):
    if split not in ('train', 'val', 'test'):
        raise ValueError('Unknown split')
    names = manifest['feature_names']
    if not names or len(set(names)) != len(names) or not set(names) <= set(FEATURE_NAMES):
        raise ValueError('Invalid features')
    directory = Path(directory)
    for kind in ('X', 'y'):
        name = f'{split}.{kind}.npy'
        if file_hash(directory / name) != manifest['array_sha256'][name]:
            raise ValueError(f'Array hash mismatch: {name}')
    X = np.load(directory / f'{split}.X.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(directory / f'{split}.y.npy', mmap_mode='r', allow_pickle=False)
    expected = manifest['splits'][split]
    if X.shape != (expected['rows'], len(names)) or X.dtype != np.float32 or y.shape != (expected['rows'],) or y.dtype != np.uint8:
        raise ValueError('Array shape/dtype mismatch')
    if np.isinf(X).any() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid numeric values')
    if int(y.sum()) != expected['classes']['bot'] or len(y)-int(y.sum()) != expected['classes']['human']:
        raise ValueError('Class count mismatch')
    return X, y


def choose_thresholds(y, scores, min_precision=.9, min_predictions=100):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    support = len(scores) - np.searchsorted(np.sort(scores), thresholds, side='left')
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-30)
    def entry(i):
        return {'threshold': float(thresholds[i]), 'precision_bot': float(precision[i]), 'recall_bot': float(recall[i]), 'f1_bot': float(f1[i]), 'predicted_positive': int(support[i])}
    best = max(range(len(thresholds)), key=lambda i: (f1[i], thresholds[i]))
    eligible = np.flatnonzero((precision[:-1] >= min_precision) & (support >= min_predictions))
    target = max(eligible, key=lambda i: (recall[i], precision[i], thresholds[i])) if len(eligible) else None
    return {'best_f1': entry(best), 'precision_target': entry(target) if target is not None else None}


def train(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    arrays = Path(config['numeric_arrays_directory'])
    if file_hash(arrays/'manifest.json') != config['arrays_manifest_sha256']:
        raise ValueError('Array manifest hash mismatch')
    if file_hash(Path(config['prepared_directory'])/'manifest.json') != config['prepared_manifest_sha256']:
        raise ValueError('Prepared manifest hash mismatch')
    manifest = json.loads((arrays/'manifest.json').read_text())
    names = config['base_features'] + [f'{n}__missing' for n in config['base_features']]
    if names != manifest['feature_names'] or manifest['feature_version'] != FEATURE_VERSION:
        raise ValueError('Feature schema mismatch')
    if output.exists():
        raise ValueError('Experiment directory already exists')
    X, y = load_array_split(arrays, manifest, 'train')
    V, vy = load_array_split(arrays, manifest, 'val')
    if len(np.unique(y)) != 2:
        raise ValueError('Training requires both classes')
    output.mkdir(parents=True)
    save_json(output/'plan.json', config)
    started = time.monotonic()
    pipe = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), XGBClassifier(**config['xgboost']))
    pipe.fit(X, y)
    scores = pipe.predict_proba(V)[:, 1]
    selection = choose_thresholds(vy, scores, config['proposed_validation_min_precision'], config['validation_min_predictions'])
    threshold = (selection['precision_target'] or selection['best_f1'])['threshold']
    prior = float(np.mean(y))
    validation = {'selection': selection, 'xgboost': metrics(vy, scores, threshold), 'baseline': metrics(vy, np.full(len(vy), prior), .5), 'baseline_selection': choose_thresholds(vy, np.full(len(vy), prior), config['proposed_validation_min_precision'], config['validation_min_predictions'])}
    save_json(output/'validation.json', validation)
    bundle = {'model': pipe, 'features': names, 'feature_version': FEATURE_VERSION, 'feature_set': 'portable', 'model_version': output.name, 'threshold': threshold, 'required_observed_features': [n for i,n in enumerate(names) if not n.endswith('__missing') and np.isfinite(X[:,i]).any()], 'minimum_observed_ratio': .5, 'minimum_posts': 20, 'sampling_policy': {'max_posts':100, 'seed':42}, 'decision_policy':'review_only', 'threads_quality_validated': False, 'calibrated': False}
    joblib.dump(bundle, output/'model.joblib')
    audit = {n: {'all_missing': bool(np.isnan(X[:,i]).all()), 'constant_observed': bool(len(np.unique(X[np.isfinite(X[:,i]),i])) <= 1)} for i,n in enumerate(names)}
    frozen = {'created_at': datetime.now(timezone.utc).isoformat(), 'code_sha256': {str(p):file_hash(p) for p in (Path(__file__), Path(__file__).with_name('features.py'), Path(__file__).with_name('sampling.py'))}, 'config_sha256': file_hash(config_path), 'arrays_manifest_sha256': file_hash(arrays/'manifest.json'), 'arrays_directory': str(arrays.resolve()), 'prior':prior, 'train_rows':len(y), 'validation_rows':len(vy), 'feature_audit_train_only':audit, 'seconds':time.monotonic()-started, 'versions':{'python':platform.python_version(), 'numpy':np.__version__, 'sklearn':sklearn.__version__, 'xgboost':xgboost.__version__}, 'frozen_sha256':{name:file_hash(output/name) for name in ('plan.json','validation.json','model.joblib')}}
    save_json(output/'frozen.json', frozen)
    return validation


def evaluate(output):
    output = Path(output)
    if (output/'test-evaluation.json').exists():
        raise ValueError('Test evaluation already recorded')
    frozen = json.loads((output/'frozen.json').read_text())
    for name, digest in frozen['frozen_sha256'].items():
        if file_hash(output/name) != digest:
            raise ValueError('Frozen artifact modified')
    plan = json.loads((output/'plan.json').read_text())
    if plan.get('evaluation_policy', '').startswith('validation_only'):
        raise ValueError('This experiment is validation-only; a new independent holdout is required')
    arrays = Path(frozen['arrays_directory'])
    if file_hash(arrays/'manifest.json') != frozen['arrays_manifest_sha256']:
        raise ValueError('Array manifest modified')
    manifest = json.loads((arrays/'manifest.json').read_text())
    X,y = load_array_split(arrays, manifest, 'test')
    bundle = joblib.load(output/'model.joblib')
    selection = json.loads((output/'validation.json').read_text())['selection']
    scores = bundle['model'].predict_proba(X)[:,1]
    report = {'test_rows':len(y), 'threshold_source':'frozen validation only', 'frozen_manifest_sha256':file_hash(output/'frozen.json'), 'xgboost':{name:metrics(y,scores,entry['threshold']) if entry else None for name,entry in selection.items()}, 'xgboost_at_0_5':metrics(y,scores,.5), 'baseline':metrics(y,np.full(len(y),frozen['prior']),.5), 'baseline_best_f1':metrics(y,np.full(len(y),frozen['prior']),json.loads((output/'validation.json').read_text())['baseline_selection']['best_f1']['threshold']), 'threads_quality_validated':False}
    save_json(output/'test-evaluation.json', report)
    return report
