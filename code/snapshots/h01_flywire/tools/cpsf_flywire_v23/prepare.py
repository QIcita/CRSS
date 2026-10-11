import argparse,json,time,shutil
import numpy as np
from .data import ROOT,DATA,read_dataset,build,store,load,record
from .model import VARIANTS
from tools.cpsf_v21_segment_units.io import save,sha

def selected(rows):
    train=[r for r in rows if r['split']=='train']
    qa={r['root_id']:json.loads((DATA/'qa'/(r['root_id']+'.json')).read_text()) for r in train}
    ordered=sorted(train,key=lambda r:qa[r['root_id']]['sample_bytes'])
    strata=[ordered[min(len(ordered)-1,int((i+.5)*len(ordered)/16))] for i in range(16)]
    extras=[max(train,key=lambda r:qa[r['root_id']]['skeleton_nodes']),max(train,key=lambda r:qa[r['root_id']]['mesh_faces'])]
    extras += [min([r for r in train if r['coarse_target']==k],key=lambda r:r['root_id']) for k in range(6)]
    union={r['root_id']:r for r in strata+extras}
    return list(union.values()),[r['root_id'] for r in strata]

def build_cell(args):
    row,profile=args;cache=ROOT/('profile_cache' if profile else 'cache');records=[]
    for mode in VARIANTS:
        assert shutil.disk_usage(ROOT).free>45*2**30,'Preserve 45GiB free disk; no deletion'
        path=cache/mode/(row['root_id']+'.pt.gz');audit=path.with_suffix('.json')
        prior=ROOT/'profile_cache'/mode/(row['root_id']+'.pt.gz')
        if not profile and prior.exists() and not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True);path.hardlink_to(prior)
            old=json.loads(prior.with_suffix('.json').read_text());save(audit,dict(old,path=str(path)))
        if path.exists():
            r=json.loads(audit.read_text());assert sha(path)==r['sha256']
        else:
            g=build(row,mode);store(path,g);r=record(row,mode,path,g);save(audit,r);del g
        records.append(r)
    return row['root_id'],records

def worker_init():
    import torch
    torch.set_num_threads(1)

def build_iter(rows,profile):
    if profile:
        yield from map(build_cell,((row,True) for row in rows));return
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from collections import deque
    # Bound in-flight work, including after a disk/geometry failure.
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=worker_init) as pool:
        source=iter(rows);queue=deque()
        for _ in range(4):
            row=next(source,None)
            if row is not None:queue.append(pool.submit(build_cell,(row,False)))
        while queue:
            result=queue.popleft().result();yield result
            row=next(source,None)
            if row is not None:queue.append(pool.submit(build_cell,(row,False)))

def main(profile=False):
    import torch
    torch.set_num_threads(2);rows,labels,ready=read_dataset();start=time.time()
    if profile:rows,strata=selected(rows)
    else:strata=[]
    cache=ROOT/('profile_cache' if profile else 'cache');records=[];bytes_used=0
    for i,(rid,cell_records) in enumerate(build_iter(rows,profile)):
        records.extend(cell_records);bytes_used+=sum(r['bytes'] for r in cell_records)
        assert bytes_used<45*2**30,'Cache budget45GiB exceeded; preserve and review'
        save(cache/'status.json',dict(state='building',cells=i+1,total=len(rows),records=len(records),
            bytes=bytes_used,seconds=time.time()-start))
        print(i+1,len(rows),rid,flush=True)
    save(cache/'manifest.json',dict(records=records,source_ready=ready,labels=labels,profile=profile))
    save(cache/'READY.json',dict(ready=True,profile=profile,manifest_sha256=sha(cache/'manifest.json')))
    save(cache/'status.json',dict(state='complete',cells=len(rows),records=len(records),bytes=bytes_used,seconds=time.time()-start))
    if profile:
        per={rid:sum(r['bytes'] for r in records if r['root_id']==rid) for rid in strata}
        estimate=float(np.mean(list(per.values()))*ready['cells'])
        save(ROOT/'cache_estimate.json',dict(estimated_bytes=estimate,stratified_cells=len(strata),
            actual_profile_bytes=bytes_used,all_profile_cells=len(rows),profile_seconds=time.time()-start,
            stratification='16 equal-frequency strata by sample file size; midpoint cell per stratum',
            source_cells=ready['cells'],hard_cache_limit_bytes=45*2**30))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',action='store_true');a=p.parse_args();main(a.profile)
