"""Refine only degenerate coarse groups; never alter fine support or mass."""
import numpy as np
import torch

def adapt(g):
    g=dict(g);n=g['nregions'];target=min(4,n)
    if g['ncoarse']>=target:return g
    pos=g['pos'].numpy();groups=[np.flatnonzero(g['coarse_owner'].numpy()==i) for i in range(g['ncoarse'])]
    while len(groups)<target:
        i=max(range(len(groups)),key=lambda i:len(groups[i]));ids=groups.pop(i)
        assert len(ids)>1
        axis=np.ptp(pos[ids],axis=0).argmax();ids=ids[np.argsort(pos[ids,axis],kind='stable')];mid=len(ids)//2
        groups[i:i]=[ids[:mid],ids[mid:]]
    owner=torch.empty(n,dtype=torch.long)
    for i,ids in enumerate(groups):owner[ids]=i
    g['coarse_owner']=owner;g['ncoarse']=len(groups)
    return g
