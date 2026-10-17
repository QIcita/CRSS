import json,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from tools.cpsf_utonia_context.common import ROOT,BASE,sha,save
from .data import raw

def worker(args):
    task,r,p=args;torch.set_num_threads(1)
    assert sha(r['path'])==r['sha256'];assert sha(p['path'])==p['sha256']
    a=raw(r['path']);b=raw(p['path']);g=a.get('g',a);point=b.get('g',b)
    assert point['patches']==len(point['pos']) and not point['has_g'].any()
    for k in ['pf','point_owner','point_weight','patch_mass']:
        assert torch.equal(g[k],point[k]),(task,r['key'],k)
    assert np.array_equal(g['origin'],point['origin'])
    assert torch.isfinite(g['sf']).all() and torch.isfinite(g['pf']).all()
    path=ROOT/'data'/task/(r['key']+'.features.npz')
    np.savez_compressed(path,patch_pos=point['pos'].numpy())
    result=dict(r,task=task,cell=r.get('cell',r.get('group')),feature_path=str(path),feature_sha256=sha(path),coordinate_source_path=p['path'],coordinate_source_sha256=p['sha256'],nodes=len(g['s']),regions=g['nregions'],patches=g['patches'],points=len(g['pf']),incidences=len(g['pr']),ncoarse=g['ncoarse'])
    if task=='tlp':
        y=a['target'];obs=a['observed'];n=len(g['query_xyz'])
        assert y.shape==obs.shape==(n,n) and torch.equal(obs,obs.T) and torch.equal(y,y.T)
        assert 'target' not in g and 'observed' not in g
        result.update(query_nodes=n,bridge_incidences=len(g['bridge_row']),pair_count=g['pair_count'])
    return result

def main():
    start=time.time();sources={};summary={}
    # Frozen R6 input support is also the originally planned R9 TED input.
    f=BASE/'CPSF_OFFICIAL_CORE_20261016_R6_A1/data/ted/manifest.json';sources[str(f)]=sha(f)
    rr=json.loads(f.read_text())['records'];save(ROOT/'data/ted/manifest.json',dict(records=rr,complete=True,source_manifest=str(f),source_sha256=sha(f)))
    for task in ['tlp','flywire_coarse']:
        (ROOT/'data'/task).mkdir(parents=True,exist_ok=True)
        if task=='tlp':
            f=BASE/'CPSF_TLP_DIRECTIONAL_STAGE6_20261009/data/tlp/joint/manifest.json';fp=f.parent.parent/'point/manifest.json';rr=json.loads(f.read_text())['records'];pp=json.loads(fp.read_text())['records'];sources[str(fp)]=sha(fp)
        else:
            f=BASE/'CPSF_V26_COMPLETE_20261003/data/flywire_coarse/manifest.json';allr=json.loads(f.read_text())['records'];rr=[r for r in allr if r['modal']=='joint'];pp=[r for r in allr if r['modal']=='point']
        sources[str(f)]=sha(f);lookup={r['key']:r for r in pp};done=[]
        with ProcessPoolExecutor(max_workers=8) as pool:
            for i,r in enumerate(pool.map(worker,[(task,r,lookup[r['key']]) for r in rr],chunksize=8)):
                done.append(r)
                if i%200==0:save(ROOT/'prepare_status.json',dict(task=task,done=i+1,total=len(rr),seconds=time.time()-start));print(task,i+1,len(rr),flush=True)
        assert len({r['key'] for r in done})==len(done)
        assert {r['cell'] for r in done if r['split']=='train'}.isdisjoint(r['cell'] for r in done if r['split']=='val')
        assert set(r['split'] for r in done)=={'train','val'}
        save(ROOT/'data'/task/'manifest.json',dict(records=done,complete=True,source_manifest=str(f),source_sha256=sha(f),input_labels=False))
        summary[task]=dict(samples=len(done),splits=dict(Counter(r['split'] for r in done)),maxima={k:max(r[k] for r in done) for k in ['nodes','regions','patches','points','incidences']})
    save(ROOT/'preparation.json',dict(passed=True,sources=sources,tasks=summary,seconds=time.time()-start,test_used=False))
if __name__=='__main__':main()
