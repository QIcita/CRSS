import time
import numpy as np
from .common import *
from .descriptor import describe
from tools.cpsf_fixed_nredd.common import store

def from_point_cache(g):
    assert len(g['pos'])==g['patches'] and not g['has_g'].any()
    xyz=g['pos'].numpy().astype(np.float64)[g['point_owner'].numpy()]+.5*g['pf'][:,:3].numpy().astype(np.float64)
    feature,stats=describe(xyz,radius=g['radius'])
    stats['old_zero_fraction']=float((g['pf']==0).all(1).float().mean())
    return torch.from_numpy(feature),stats

def main():
    verify();modes=['joint','point','skeleton'];rr={m:parent_rows(m)[0] for m in modes};lookup={m:{r['key']:r for r in rr[m]} for m in modes}
    assert all([r['key'] for r in rr[m]]==[r['key'] for r in rr['joint']] for m in modes)
    assert not any((ROOT/'data/tlp'/m/'manifest.json').exists() for m in modes)
    results={m:[] for m in modes};statistics=[];start=time.time()
    for index,base in enumerate(rr['joint']):
        key=base['key'];pr=lookup['point'][key];assert sha(pr['path'])==pr['sha256'];point=load(pr['path'])
        feature,stats=from_point_cache(point['g']);statistics.append(dict(key=key,split=base['split'],**stats))
        rng=np.random.RandomState(int(hashlib.sha256(('descriptor42|'+key).encode()).hexdigest()[:8],16));permutation=torch.tensor(rng.permutation(len(feature)),dtype=torch.long)
        for mode in modes:
            row=lookup[mode][key];assert sha(row['path'])==row['sha256'];item=point if mode=='point' else load(row['path']);g=dict(item['g'])
            if mode!='skeleton':
                for field in ['pf','point_owner','point_weight','patch_mass']:assert torch.equal(g[field],point['g'][field])
                g.update(pf_neighborhood=feature,descriptor_permutation=permutation)
            else:g.update(pf_neighborhood=g['pf'].clone(),descriptor_permutation=torch.empty(0,dtype=torch.long))
            path=ROOT/'data/tlp'/mode/(key+'.pt.gz');store(path,dict(item,g=g))
            results[mode].append(dict(row,path=str(path),sha256=sha(path),parent_cache_path=row['path'],parent_cache_sha256=row['sha256']))
        if (index+1)%20==0:print(index+1,len(rr['joint']),flush=True)
    for mode,records in results.items():
        save(ROOT/'data/tlp'/mode/'manifest.json',dict(complete=True,records=records,protocol_sha256=sha(ROOT/'protocol.json')))
        train=sorted([r for r in records if r['split']=='train'],key=lambda r:(r['query_nodes'],r['key']))
        save(ROOT/'pilot/tlp'/mode/'manifest.json',dict(records=[train[k] for k in [0,len(train)//2,len(train)-1]],selection='train-only query node count'))
    save(ROOT/'descriptor_statistics.json',dict(records=statistics,seconds=time.time()-start,test_used=False))
    print('complete',len(statistics),time.time()-start,flush=True)
if __name__=='__main__':main()
