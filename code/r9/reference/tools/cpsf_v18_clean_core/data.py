"""Frozen cohorts, label-free geometry, and explicit format-only adapters."""
import argparse,json,shutil,time,sys
from pathlib import Path
import numpy as np
import torch
from .common import *
from .geometry import prepare_geometry,adapt_ted_event
from .tlp_adapter import load_tlp_v2e

def raw_local(full,sample):
    with np.load(SOURCE/'local_refs'/(sample+'.npz')) as ref:
        nodes=ref['node_ids'];points=ref['point_ids'];edges=ref['edge_ids'];query=int(ref['query_index'])
    owners=full['point_to_edge'][points];edges=np.union1d(edges,np.unique(owners))
    halo=np.setdiff1d(np.unique(full['edge_index'][edges]),nodes);nodes=np.r_[nodes,halo]
    lookup=np.full(len(full['skeleton_xyz_um']),-1,np.int64);lookup[nodes]=np.arange(len(nodes))
    center=full['skeleton_xyz_um'][nodes[query]]
    return (full['skeleton_xyz_um'][nodes]-center,lookup[full['edge_index'][edges]],full['surface_xyz_um'][points]-center),query,points

def build_h01(out,smoke=False):
    assert not out.exists();out.mkdir(parents=True);(out/'local').mkdir()
    manifest=json.loads((V9/'manifest.json').read_text());rows=manifest['records']
    if smoke:rows=[next(r for r in rows if r['split']==split and r['target']==c) for split in ('train','val') for c in range(3)]
    done=[];started=time.time()
    for cell in sorted({r['cell'] for r in rows}):
        source=SOURCE/'inputs'/(cell+'.npz')
        with np.load(source,allow_pickle=False) as a:
            full={k:a[k] for k in ('skeleton_xyz_um','edge_index','surface_xyz_um','point_to_edge')}
        source_sha=sha(source)
        for r in [r for r in rows if r['cell']==cell]:
            assert shutil.disk_usage(out).free>4*2**30,'No automatic deletion'
            raw,q,point_ids=raw_local(full,r['sample']);g=prepare_geometry(*raw)
            oldpath=V9/'local'/(r['sample']+'.pt');assert sha(oldpath)==r['sha256'];old=load(oldpath)
            assert np.array_equal(np.unique(np.sort(raw[1],axis=1),axis=0),np.unique(np.sort(old['input']['edge_index'],axis=1),axis=0))
            assert q==old['input']['query_index'] and len(raw[2])==r['points']
            assert sha_bytes(point_ids.tobytes())==r['contract']['point_ids_sha256']
            item=dict(geometry=g,query=q,target=old['target'],sample=r['sample'],cell=cell,split=r['split'])
            path=out/'local'/(r['sample']+'.pt');torch.save(item,path)
            done.append(dict(sample=r['sample'],cell=cell,split=r['split'],target=item['target'],points=len(raw[2]),
                             sha256=sha(path),source_sha256=source_sha))
        save(out/'progress.json',dict(samples=len(done),seconds=time.time()-started));print('PREPARED',cell,len(done),flush=True)
    save(out/'manifest.json',dict(records=done,source_manifest_sha256=sha(V9/'manifest.json'),smoke=smoke,
                                 raw_points=sum(r['points'] for r in done),seconds=time.time()-started))

def sha_bytes(b):return __import__('hashlib').sha256(b).hexdigest()

def local_sets(cache,smoke=False):
    m=json.loads((cache/'manifest.json').read_text());sets={k:[] for k in ('train','val')}
    for r in m['records']:
        path=cache/'local'/(r['sample']+'.pt');assert sha(path)==r['sha256']
        sets[r['split']].append(dict(**r,path=path))
    assert {r['cell'] for r in sets['train']}.isdisjoint(r['cell'] for r in sets['val'])
    if not smoke:assert [len(sets[k]) for k in ('train','val')]==[716,214]
    return sets

def ted_sets(smoke=False):
    report=json.loads((TED/'construction_report.json').read_text());sets={}
    for split,count in [('train',33519),('test',9893)]:
        records=report['splits'][split]['rows'];records=records[:1] if smoke else records
        blocks=[]
        for r in records:
            path=TED/split/(Path(r['block']).stem+'.pt');b=load(path)
            if smoke:b['events']=b['events'][:70];b['labels']=b['labels'][:70]
            events=[dict(event=e,target=b['labels'][i],name=f'{Path(b["block"]).stem}:{i}') for i,e in enumerate(b['events'])]
            blocks.append(dict(block=b['block'],events=events,source_sha256=sha(path)))
        if not smoke:assert sum(len(b['events']) for b in blocks)==count
        sets[split]=blocks
    return sets

def tlp_sets(smoke=False):
    sys.path[:0]=[str(NREDD),str(NREDD/'.deps/cpsf_geometry'),str(NREDD/'test'),
        str(NREDD/'artifacts/CPSF_GEOMETRY_V2E_TLP_20260917/data')]
    from cpsf_geometry_v2e_tlp_dataset import CPSFGeometryV2ETLPDataset
    cache=NREDD/'data/link_pred_data/official_192_81_geometry_v2e_cache';sets={}
    for split,count in [('train',192),('test',81)]:
        # The original loader can create labels/image features if absent.
        # Require its frozen prerequisites before calling it, so loading here
        # is read-only and cannot regenerate annotations from GT implicitly.
        for swc in (NREDD/'data/link_pred_data'/split/'raw_data').glob('*.swc'):
            assert (swc.parent.parent/'label'/(swc.stem+'.txt')).is_file(),'Missing frozen label: '+str(swc)
            assert (swc.parent.parent/'SP_feature'/swc.stem).is_dir(),'Missing frozen SP prerequisite: '+str(swc)
        ds=CPSFGeometryV2ETLPDataset(str(NREDD/'data/link_pred_data'/split),str(cache/'edge_all'),str(cache/'bridge_all'))
        assert len(ds)==count
        # Keep only target/observed/SWC and cache paths; do not retain all padded point arrays.
        rows=[]
        for i in range(min(2,len(ds)) if smoke else len(ds)):
            swc=Path(ds.swc_paths[i]);name=swc.stem
            rows.append(dict(name=name,swc=str(swc),observed=torch.as_tensor(ds.adjacency[i]).float(),
                target=torch.as_tensor(ds.target[i]).float(),edge=cache/'edge_all'/(name+'.npz'),bridge=cache/'bridge_all'/(name+'.npz')))
        sets[split]=rows
    return sets

def tlp_geometry(row):
    s,e,p,pairs,records=load_tlp_v2e(row['edge'],row['bridge'])
    observed=torch.from_numpy(np.array(e));expected=torch.nonzero(torch.triu(row['observed']>0,diagonal=1))
    assert torch.equal(observed,expected),'Observed edge mismatch'
    assert all(r['source_swc_sha256']==sha(row['swc']) for r in records)
    g=prepare_geometry(s,e,p);g['query_pairs']=torch.from_numpy(pairs)
    return g

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();build_h01(a.out,a.smoke)
