"""Label-free similarities to a fixed reference cohort."""
from collections import Counter
import numpy as np


def reference_features(reference, queries, training=False, max_members=60):
    if training and len(reference)!=len(queries):raise ValueError('Training reference alignment required')
    index={}
    for i,tokens in enumerate(reference):
        for token in tokens:
            members=index.setdefault(token,[])
            if members is not None:
                members.append(i)
                if len(members)>max_members:index[token]=None
    X=np.zeros((len(queries),4))
    for i,tokens in enumerate(queries):
        matches=Counter();eligible=0
        for token in tokens:
            members=index.get(token)
            if members:
                other=[j for j in members if not training or j!=i]
                if other:eligible+=1;matches.update(other)
        maximum=max(matches.values(),default=0)
        X[i]=[eligible/max(1,len(tokens)),np.log1p(sum(v>=2 for v in matches.values())),maximum,maximum/max(1,len(tokens))]
    return X
