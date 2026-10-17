import argparse,math,time,shutil
import numpy as np
from .common import *
from tools.cpsf_h01_topology_repair.adapter import geometry
from tools.cpsf_v22_shared_region.data import regions,whole_geometry
from tools.cpsf_v24_physical_conditioning.data import adapt
from tools.cpsf_v18_clean_core.tlp_adapter import load_tlp_v2e
from tools.cpsf_flywire_v23.data import fixed_origin

def ted_events(row):
    assert sha(row['cache'])==row['cache_sha256'] and sha(row['metadata'])==row['metadata_sha256'] and sha(row['swc'])==row['swc_sha256']
    b=torch.load(row['cache'],map_location='cpu',weights_only=False)
    with np.load(row['metadata']) as z:meta=json.loads(z['metadata_json'].item());ids=z['subgraph_node_indices'];cached=z['skeleton_xyz']
    assert meta['ground_truth_used'] is False and meta['labels_stored'] is False and meta['swc_label_column_zeroed_before_feature_extraction'] is True
    assert meta['source_swc_sha256']==row['swc_sha256']
    xyz=np.loadtxt(row['swc'],usecols=(2,3,4));assert len(b['events'])==row['events']==len(ids)
    for i,event in enumerate(b['events']):
        scale=float(meta['event_statistics'][i]['scale']);s=event['xyz'].numpy();valid=ids[i][ids[i]>=0]
        assert np.isfinite(scale) and scale>0 and len(s)==len(valid)
        np.testing.assert_allclose(s,cached[i,:len(s)],atol=0,rtol=0)
        assert np.max(abs(s*scale-(xyz[valid]-xyz[valid[0]])))<.002
        p=np.concatenate([v.numpy() for _,v in event['patches']])
        yield dict(key=row['key']+':'+str(i),s=s*scale/2,e=event['edges'].numpy(),p=p*scale/2,
            q=np.zeros((1,3)),target=b['labels'][i].float(),source_event=i)

def tlp_event(row):
    for key in ('edge','bridge','swc','label'):assert sha(row[key])==row[key+'_sha256']
    s,e,p,pairs,records=load_tlp_v2e(row['edge'],row['bridge'])
    assert all(r['source_swc_sha256']==row['swc_sha256'] for r in records)
    swc=np.loadtxt(row['swc']);n=len(swc);assert np.array_equal(swc[:,0],np.arange(1,n+1))
    np.testing.assert_allclose(s,swc[:,2:5]*[1,1,2],atol=.001)
    observed=np.eye(n,dtype=bool)
    for i,parent in enumerate(swc[:,-1].astype(int)):
        if parent!=-1:observed[i,parent-1]=observed[parent-1,i]=True
    assert np.array_equal(e,np.argwhere(np.triu(observed,1)))
    y=np.loadtxt(row['label']);assert y.shape==(n,n) and np.isin(y,[0,1]).all() and np.array_equal(y,y.T)
    # Only observed adjacency constructs support. Label matrix stays outside g.
    p=np.unique(p,axis=0);center=s.mean(0)
    return dict(key=row['key'],s=s-center,e=e,p=p-center,q=s-center,target=torch.tensor(y,dtype=torch.float32),observed=torch.tensor(observed),pairs=torch.tensor(pairs))

def build(event,task,mode):
    s,e,p=event['s'],event['e'],event['p'];empty=np.empty((0,3));edges=np.empty((0,2),int)
    if task=='ted':
        g=geometry(empty if mode=='point' else s,edges if mode=='point' else e,empty if mode=='skeleton' else p)
    elif mode=='skeleton':g=geometry(s,e,empty)
    else:
        raw=dict(skeleton_xyz_um=s,edge_index=e,surface_xyz_um=p)
        g=whole_geometry(raw,'point' if mode=='point' else 'joint')
        full_count=g['source_points'];tiles=g['tile_count']
        if mode in ('thumbnail','uniform_budget'):
            budget=min(len(p),4096) if mode=='thumbnail' else full_count
            rng=np.random.default_rng(42);left=budget;ids=[]
            while left:
                take=min(len(p),left);ids.append(np.sort(rng.choice(len(p),take,replace=False)));left-=take
            chosen=np.concatenate(ids);w=np.full(budget,len(p)/budget)
            g=geometry(s,e,p[chosen],w)
        # Fixed source origin across sampling arms; no skeleton-derived point frame.
        origin=(p.min(0)+p.max(0))/2
        sampled=g['p'].numpy()+np.asarray(g['origin']);weight=g['w'].numpy()
        g=fixed_origin(g,empty if mode=='point' else s,edges if mode=='point' else e,sampled,weight,origin)
        g['tile_count']=tiles if mode in ('joint','point') else 0
        assert abs(float(g['mass'].sum())-len(p))/len(p)<1e-5
    out=adapt(regions(g,mode if mode in ('point','skeleton') else 'joint'))
    out['query_xyz']=torch.tensor(event['q']-np.asarray(g['origin']),dtype=torch.float32)
    ids=torch.where(out['has_g']&out['has_p'])[0];rng=np.random.RandomState(int(hashlib.sha256(('within-cell-region42|'+event['key']).encode()).hexdigest()[:8],16))
    ordered=ids[torch.tensor(rng.permutation(len(ids)),dtype=torch.long)];perm=torch.arange(out['nregions'])
    if len(ids)>1:perm[ordered]=torch.roll(ordered,1)
    out['surface_content_permutation']=perm
    if mode=='point':assert out['raw_nodes']==0 and not out['has_g'].any() and len(out['adjacency'])>=0
    else:assert out['raw_nodes']==len(s) and out['raw_edges']==len(np.unique(np.sort(e,axis=1),axis=0))
    if mode=='skeleton':assert not out['has_p'].any() and len(out['pf'])==0
    assert out['has_g'].any() or out['has_p'].any()
    assert 'target' not in out and 'observed' not in out
    return out

def prepare(task,mode,pilot=False):
    verify();manifest=json.loads((ROOT/'split_manifest.json').read_text());rows=manifest['records'][task]
    selected=[r for r in rows if r['split'] in ('train','val')]
    if pilot:selected=sorted([r for r in selected if r['split']=='train'],key=lambda r:-r['nodes'])[:1]
    folder=ROOT/('pilot' if pilot else 'data')/task/mode;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():return
    done=[];start=time.time()
    for row in selected:
        events=ted_events(row) if task=='ted' else [tlp_event(row)]
        if pilot and task=='ted':
            events=list(events)
            selected_events=events[:10]
            for label in (0,1):
                matching=next((v for v in events if int(v['target'].argmax())==label),None)
                assert matching is not None,('pilot block missing class',label)
                if matching['key'] not in {v['key'] for v in selected_events}:selected_events.append(matching)
            events=selected_events
        for i,event in enumerate(events):
            file=folder/(event['key'].replace(':','_')+'.pt.gz')
            if file.exists():item=load(file);assert item['key']==event['key']
            else:
                assert shutil.disk_usage(ROOT).free>15*2**30,'disk reserve reached; preserve all files'
                g=build(event,task,mode);item=dict(key=event['key'],g=g,target=event['target'])
                if task=='tlp':item.update(observed=event['observed'],pairs=event['pairs'])
                store(file,item)
            g=item['g'];done.append(dict(key=event['key'],block=row['key'],split=row['split'],group=row['spatial_group'],path=str(file),sha256=sha(file),
                nodes=g['raw_nodes'],regions=g['nregions'],points=g['surface_points'],coarse=g['ncoarse'],target_class=int(event['target'].argmax()) if task=='ted' else None))
        save(folder/'progress.json',dict(block=row['key'],samples=len(done),blocks_expected=len(selected),seconds=time.time()-start));print(task,mode,row['key'],len(done),flush=True)
    save(folder/'manifest.json',dict(complete=True,pilot=pilot,records=done,protocol_sha256=sha(ROOT/'protocol.json'),seconds=time.time()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--mode',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();prepare(a.task,a.mode,a.pilot)
