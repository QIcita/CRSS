import argparse,gzip,os,time,shutil,fcntl
import numpy as np
import torch
from .common import ROOT,BASE,add_backend,read,save,sha,source_rows,verify

def build(row,baseline):
    from .sources import geometry as source_geometry
    from .sampling import sample_geometry
    from tools.cpsf_h01_topology_repair.adapter import geometry
    from tools.cpsf_flywire_v23.data import fixed_origin
    from tools.cpsf_v22_shared_region.data import regions
    from tools.cpsf_v24_physical_conditioning.data import adapt
    task=row['task'];key=row['key'];folder=ROOT/'data'/task
    safe=key.replace(':','_');record=folder/(safe+'.json')
    if record.exists():
        r=read(record)
        assert r['protocol_sha256']==sha(ROOT/'protocol.json')
        for a,b in [('path','sha256'),('feature_path','feature_sha256'),('point_path','point_sha256')]:
            assert sha(r[a])==r[b]
        return r
    assert shutil.disk_usage(ROOT).free>12*2**30,'Reserve 12 GiB; do not delete other experiments'
    begin=time.time();x=source_geometry(row);p,w,ids,audit=sample_geometry(x['p'],x.get('mass'))
    origin=(x['p'].min(0)+x['p'].max(0))/2
    geo=fixed_origin(geometry(x['s'],x['e'],p,w),x['s'],x['e'],p,w,origin);geo['tile_count']=0
    g=adapt(regions(geo,'joint',x['q'] if task=='h01_local' else None))
    if task in ('ted','tlp'):g['query_xyz']=torch.tensor(x['q']-origin,dtype=torch.float32)
    if task=='tlp':
        from tools.cpsf_pair_region.data import support
        from tools.cpsf_directional.descriptor import describe
        g.update(support(g,key))
        features,valid,stats=describe(geo['p'].numpy(),radius=g['radius'])
        g['directional_valid']=torch.from_numpy(valid)
        g['pf_directional']=torch.where(g['directional_valid'][:,None],torch.from_numpy(features),g['pf'])
        audit['directional_descriptor']=stats
    assert g['raw_nodes']==len(x['s']) and g['raw_edges']==len(np.unique(np.sort(x['e'],axis=1),axis=0))
    assert 'target' not in g and 'observed' not in g and 'label' not in g
    assert np.isclose(float(g['patch_mass'].sum()),w.sum(),rtol=3e-5)
    if baseline is not None:
        assert sha(baseline['graph_path'])==baseline['graph_sha256']
        with np.load(baseline['graph_path']) as a:
            assert len(a['coord'])==len(x['s'])
            assert np.allclose(a['coord']-a['coord'][0],x['s']-x['s'][0],atol=5e-4)
            assert np.array_equal(np.unique(np.sort(a['edge_index'],axis=1),axis=0),np.unique(np.sort(x['e'],axis=1),axis=0))
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(safe+('.pt' if task.startswith('h01') else '.pt.gz'))
    item=g
    if task in ('ted','tlp'):
        assert sha(row['path'])==row['sha256']
        with gzip.open(row['path'],'rb') as f:old=torch.load(f,map_location='cpu',weights_only=False)
        item=dict(g=g,key=key,target=old['target'])
        if task=='tlp':
            item.update(observed=old['observed'],pairs=old['pairs'])
            assert len(item['target'])==len(x['s'])
            observed=np.eye(len(x['s']),dtype=bool)
            observed[x['e'][:,0],x['e'][:,1]]=True;observed|=observed.T
            assert np.array_equal(observed,item['observed'].numpy())
    temp=path.with_suffix('.tmp')
    if task.startswith('h01'):torch.save(item,temp)
    else:
        with temp.open('wb') as f:
            with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,compresslevel=1) as z:torch.save(item,z)
    temp.replace(path)
    feature=folder/(safe+'.features.npz');np.savez_compressed(feature,patch_pos=geo['c'].numpy())
    # Reconstructed network observations and exported baseline observations match.
    reconstructed=geo['c'].numpy()[g['point_owner'].numpy()]+g['pf'].numpy()[:,:3]*.5
    assert np.allclose(reconstructed,geo['p'].numpy(),atol=3e-5)
    audit.update(source_lineage=x['lineage'],coordinate_units=x['units'],source_mass_kind=x['mass_kind'],
        seconds=time.time()-begin,origin=origin.tolist(),no_legacy_sample_used=True,
        representation_memberships_recomputed=True,shared_point_export=True)
    r={k:row[k] for k in ('key','task','cell','split')}
    r.update(target=row.get('target'),path=str(path),sha256=sha(path),feature_path=str(feature),feature_sha256=sha(feature),
        point_path=str(path),point_sha256=sha(path),point_storage='canonical geometry plus patch centres',points=len(g['pf']),nodes=len(g['s']),edges=g['raw_edges'],
        patches=g['patches'],regions=g['nregions'],incidences=len(g['pr']),ncoarse=g['ncoarse'],
        source=row.get('source',{}),observation_audit=audit,protocol_sha256=sha(ROOT/'protocol.json'))
    if baseline:
        for k in ('graph_path','graph_sha256','skeleton_path','skeleton_sha256'): 
            if k in baseline:r[k]=baseline[k]
    save(record,r);return r

def main(task,limit=0):
    verify();add_backend('r28');torch.set_num_threads(1)
    (ROOT/'preparation').mkdir(exist_ok=True)
    lock=open(ROOT/'preparation'/(task+'.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=read(BASE/'CPSF_EXTERNAL_BASELINES_PHASE2_20261010/data/manifest.json')['records']
    lookup={(r['task'],r['key']):r for r in base};rr=source_rows(task)
    if limit:rr=[r for r in rr if r['split']=='train'][:limit]
    done=[];begin=time.time()
    for i,row in enumerate(rr):
        done.append(build(row,lookup.get((task,row['key']))))
        save(ROOT/'preparation'/f'{task}.json',dict(state='building',processed=i+1,total=len(rr),task=task,
            points=done[-1]['points'],last_key=row['key'],seconds=time.time()-begin,pid=os.getpid()))
        if i%10==0:print(task,i+1,len(rr),row['key'],flush=True)
    assert {r['cell'] for r in done if r['split']=='train'}.isdisjoint(r['cell'] for r in done if r['split']=='val')
    if limit:save(ROOT/'preparation'/f'{task}_pilot.json',dict(records=done,complete=True));return
    cfg=read(ROOT/'protocol.json')['tasks'][task]
    assert sum(r['split']=='train' for r in done)==cfg['train'] and sum(r['split']=='val' for r in done)==cfg['val']
    dest=ROOT/'data'/task/'manifest.json';save(dest,dict(records=done,complete=True,protocol_sha256=sha(ROOT/'protocol.json')))
    save(ROOT/'preparation'/f'{task}.json',dict(state='complete',processed=len(done),task=task,manifest_sha256=sha(dest),seconds=time.time()-begin))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--task',required=True);a.add_argument('--limit',type=int,default=0);a=a.parse_args()
    try:main(a.task,a.limit)
    except BaseException as e:
        save(ROOT/'preparation'/f'{a.task}_failure.json',dict(error=repr(e),pid=os.getpid()));raise
