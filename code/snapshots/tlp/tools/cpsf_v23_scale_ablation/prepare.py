"""Change surface sampling only; retain full skeleton and original voxel origin."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import SOURCE,load,save,sha
from tools.cpsf_h01_topology_repair.adapter import geometry
from tools.cpsf_v22_shared_region.data import regions
from tools.cpsf_v22_shared_region.prepare import store

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/root/megrez-tmp/CPSF_V23_RESIDUAL_REGION_20261001')


def sampled_geometry(x,arm,reference):
    p=x['surface_xyz_um'];n=len(p);rng=np.random.default_rng(42)
    budget=min(n,4096) if arm=='thumbnail' else reference['surface_points']
    # Full passes only matter for tiny cells whose original draws exceed N.
    ids=[];left=budget
    while left:
        count=min(n,left);ids.append(rng.choice(n,count,replace=False));left-=count
    ids=np.sort(np.concatenate(ids));selected=p[ids];weights=np.full(budget,n/budget)
    g=geometry(x['skeleton_xyz_um'],x['edge_index'],selected,weights)
    # Keep voxel boundaries and skeleton features exactly equal to the full cache.
    # A sample-dependent bounding-box origin would otherwise be an extra change.
    origin=np.asarray(reference['origin'])
    points,inv=np.unique(np.asarray(selected,np.float64)-origin,axis=0,return_inverse=True)
    w=np.bincount(inv,weights=weights)
    _,owner=np.unique(np.floor(points/.5).astype(np.int64),axis=0,return_inverse=True)
    mass=np.bincount(owner,weights=w);centers=np.zeros((len(mass),3))
    np.add.at(centers,owner,points*w[:,None]);centers/=mass[:,None]
    rel=(points-centers[owner])/.5;pf=np.column_stack((rel,np.linalg.norm(rel,axis=1)))
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32)
    g.update(s=reference['s'].clone(),sf=reference['sf'].clone(),origin=reference['origin'],
        p=tensor(points),w=tensor(w),owner=torch.as_tensor(owner,dtype=torch.long),
        mass=tensor(mass),c=tensor(centers),pf=tensor(pf),m=len(mass),
        source_points=budget,tile_count=0)
    return g,dict(draws=budget,unique_source_indices=len(np.unique(ids)),source_points=n,
        reference_draws=reference['surface_points'],reference_tiles=reference['tile_count'],
        sampling_indices_sha256=__import__('hashlib').sha256(ids.tobytes()).hexdigest())


def main(smoke=False):
    torch.set_num_threads(2);start=time.time()
    folder='smoke_cache' if smoke else 'cache';cache=ROOT/folder
    original=json.loads((BASE/folder/'manifest.json').read_text())
    records=[dict(r) for r in original['records'] if r['modal']=='joint']
    audit=[]
    for row in [r for r in records if r['task']=='global']:
        reference=load(row['path']);path=SOURCE/'inputs'/(row['cell']+'.npz')
        assert sha(path)==row['source_sha256']
        with np.load(path) as x:
            for arm in ('thumbnail','uniform_budget'):
                dest=cache/'global'/arm/(row['cell']+'.pt')
                assert shutil.disk_usage(ROOT).free>12*2**30
                if dest.exists():
                    data=load(dest);record=json.loads(dest.with_suffix('.audit.json').read_text())
                else:
                    g,record=sampled_geometry(x,arm,reference);data=regions(g,'joint')
                    assert torch.equal(data['s'],reference['s']) and torch.equal(data['sf'],reference['sf'])
                    assert data['raw_nodes']==reference['raw_nodes'] and data['raw_edges']==reference['raw_edges']
                    torch.testing.assert_close(data['length'].sum(),reference['length'].sum(),rtol=2e-5,atol=.001)
                    assert abs(float(data['surface_mass'].sum())-len(x['surface_xyz_um']))/len(x['surface_xyz_um'])<1e-5
                    record.update(cell=row['cell'],arm=arm,skeleton_unchanged=True,origin_unchanged=True,
                        raw_nodes=data['raw_nodes'],raw_edges=data['raw_edges'],nregions=data['nregions'])
                    store(dest,data);save(dest.with_suffix('.audit.json'),record)
                records.append(dict(row,modal=arm,path=str(dest),sha256=sha(dest)))
                audit.append(record)
                save(cache/'status.json',dict(state='building',cell=row['cell'],arm=arm,
                    completed_resolution_caches=len(audit),seconds=time.time()-start))
                del data
        print(row['cell'],len(audit),flush=True)
    manifest=dict(original,records=records,inherited_manifest_sha256=sha(BASE/folder/'manifest.json'))
    save(cache/'manifest.json',manifest);save(cache/'sampling_audit.json',audit)
    save(cache/'READY.json',dict(ready=True,smoke=smoke,records=len(records),manifest_sha256=sha(cache/'manifest.json')))
    save(cache/'status.json',dict(state='complete',records=len(records),seconds=time.time()-start))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args().smoke)
