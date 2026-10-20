"""Label-free deterministic sparse support for every unordered node pair."""
import argparse,time
import numpy as np
from scipy.spatial import cKDTree
from .common import *
from tools.cpsf_fixed_nredd.common import store

def support(g,key):
    q=g['query_xyz'].numpy().astype(np.float64);pos=g['pos'].numpy().astype(np.float64)
    n=len(q);ii,jj=np.triu_indices(n,1);p=len(ii);tree=cKDTree(pos);radius=2*g['radius']
    rows_out=[];cols_out=[];weights=[]
    for start in range(0,p,1024):
        aa=q[ii[start:start+1024]];bb=q[jj[start:start+1024]];v=bb-aa
        probes=aa[:,None]+np.array([.25,.5,.75])[None,:,None]*v[:,None]
        distance,ids=tree.query(probes,k=8,distance_upper_bound=radius,workers=1)
        ids=ids.reshape(len(aa),-1)
        for local,candidates in enumerate(ids):
            candidates=np.unique(candidates[candidates<len(pos)])
            if not len(candidates):continue
            delta=pos[candidates]-aa[local];den=v[local]@v[local]
            if den<=1e-12:continue
            t=(delta@v[local])/den;radial=np.linalg.norm(delta-t[:,None]*v[local],axis=1)
            keep=(t>0)&(t<1)&(radial<=radius);candidates=candidates[keep]
            rows_out.extend([start+local]*len(candidates));cols_out.extend(candidates)
            weights.extend(np.exp(-.5*(radial[keep]/g['radius'])**2))
    row=np.asarray(rows_out,np.int64);col=np.asarray(cols_out,np.int64)
    lookup=np.zeros((n,n),np.int64);lookup[ii,jj]=lookup[jj,ii]=np.arange(1,p+1)
    valid=np.unique(row);perm=np.arange(p)
    rng=np.random.RandomState(int(hashlib.sha256(('pair-region42|'+key).encode()).hexdigest()[:8],16))
    ordered=rng.permutation(valid)
    if len(ordered)>1:perm[ordered]=np.roll(ordered,1)
    return dict(bridge_row=torch.from_numpy(row),bridge_col=torch.from_numpy(col),bridge_weight=torch.tensor(np.asarray(weights),dtype=torch.float32),
        pair_count=p,pair_lookup=torch.from_numpy(lookup),bridge_permutation=torch.from_numpy(perm))

def prepare(mode):
    verify();rr,_=parent_rows(mode);folder=ROOT/'data/tlp'/mode;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'manifest.json').exists(),'Already prepared; preserve and verify existing manifest'
    result=[];start=time.time()
    for r in rr:
        assert sha(r['path'])==r['sha256'];item=load(r['path']);g=dict(item['g']);g.update(support(g,r['key']))
        assert 'target' not in g and 'observed' not in g
        path=folder/(r['key']+'.pt.gz');store(path,dict(item,g=g))
        result.append(dict(r,path=str(path),sha256=sha(path),parent_cache_sha256=r['sha256'],parent_cache_path=r['path'],
            query_nodes=len(g['query_xyz']),pair_count=g['pair_count'],incidences=len(g['bridge_row']),supported_pairs=int(g['bridge_row'].unique().numel())))
        if len(result)%20==0:print(mode,len(result),len(rr),flush=True)
    manifest=dict(complete=True,protocol_sha256=sha(ROOT/'protocol.json'),records=result,seconds=time.time()-start)
    save(folder/'manifest.json',manifest)
    train=sorted([r for r in result if r['split']=='train'],key=lambda r:(r['query_nodes'],r['key']))
    save(ROOT/'pilot/tlp'/mode/'manifest.json',dict(records=[train[k] for k in [0,len(train)//2,len(train)-1]],selection='train-only node count'))
    print(mode,'complete',manifest['seconds'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['joint','skeleton','point']);a=p.parse_args();prepare(a.mode)
