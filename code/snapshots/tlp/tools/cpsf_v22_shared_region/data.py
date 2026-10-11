"""Label-free physical regions from full original skeleton and sampled surface."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from tools.cpsf_h01_topology_repair.adapter import geometry
from tools.cpsf_v21_segment_units.model import spatial_chunks
from .support import build_units

def whole_geometry(x,arm):
    empty=np.empty((0,3));edges=np.empty((0,2),int)
    if arm=='skeleton':return geometry(x['skeleton_xyz_um'],x['edge_index'],empty)
    p=x['surface_xyz_um'];rng=np.random.default_rng(42)
    ids=np.arange(len(p));selected=[np.sort(rng.choice(ids,min(len(ids),4096),replace=False))]
    weights=[np.full(len(selected[0]),.5*len(p)/len(selected[0]))]
    chunks=[ids];target=min(32,max(1,int(np.ceil(len(p)/16384))))
    while len(chunks)<target:
        k=max(range(len(chunks)),key=lambda i:len(chunks[i]));ids=chunks.pop(k)
        if len(ids)<2:chunks.insert(k,ids);break
        axis=np.ptp(p[ids],axis=0).argmax();ids=ids[np.argsort(p[ids,axis],kind='stable')];mid=len(ids)//2
        chunks[k:k]=[ids[:mid],ids[mid:]]
    for ids in chunks:
        take=np.sort(rng.choice(ids,min(len(ids),2048),replace=False));selected.append(take)
        weights.append(np.full(len(take),.5*len(ids)/len(take)))
    # Both thumbnail and tiles estimate the same source measure: sum weights=N.
    g=geometry(empty if arm=='point' else x['skeleton_xyz_um'],edges if arm=='point' else x['edge_index'],p[np.concatenate(selected)],np.concatenate(weights))
    g['tile_count']=len(chunks);return g

def local_geometry(g,arm):
    empty=np.empty((0,3));edges=np.empty((0,2),int)
    # Rebuild after dropping the forbidden modality; surface points are unchanged.
    return geometry(empty if arm=='point' else g['s'].numpy(),edges if arm=='point' else g['e'].numpy(),
                    empty if arm=='skeleton' else g['p'].numpy(),None if arm=='skeleton' else g['w'].numpy())

def regions(g,arm,query=None):
    tensor=lambda a,dtype=torch.float32:torch.as_tensor(np.asarray(a),dtype=dtype)
    if arm=='point':
        u=g['m'];pos=g['c'];length=torch.zeros(u);sr=sn=torch.empty(0,dtype=torch.long);sw=torch.empty(0)
        pr=pj=torch.arange(u);pw=torch.ones(u);phi=torch.zeros((u,5))
        # Pure surface neighborhood: no skeleton anchors, indices or scale.
        pairs=cKDTree(pos.numpy()).query_pairs(1.) if u else set()
        adjacency=tensor(np.asarray(sorted(pairs),int).reshape(-1,2),torch.long)
    else:
        v=build_units(g);u=v['u'];pos=v['center'];length=v['length']
        sr,sn,sw,pr,pj,pw,phi=[v[k] for k in ('sr','sn','sw','pr','pj','pw','phi')]
        adjacency=v['adjacency']
        # Surface patches without any observed-graph support remain real tokens.
        orphan=np.setdiff1d(np.arange(g['m']),np.unique(pj.numpy()))
        if len(orphan):
            pos=torch.cat((pos,g['c'][orphan]));length=torch.cat((length,torch.zeros(len(orphan))))
            pr=torch.cat((pr,torch.arange(u,u+len(orphan))));pj=torch.cat((pj,torch.tensor(orphan)))
            pw=torch.cat((pw,torch.ones(len(orphan))));phi=torch.cat((phi,torch.zeros((len(orphan),5))))
            u+=len(orphan)
    assert u>0
    gm=torch.zeros(u).index_add_(0,sr,sw)
    pm=torch.zeros(u).index_add_(0,pr,pw*g['mass'][pj])
    has_g=gm>0;has_p=pm>0
    # Coarse hierarchy budgets computation, never discards fine region support.
    groups=spatial_chunks(pos.numpy(),max(32,int(np.ceil(u/128))))
    owner=torch.empty(u,dtype=torch.long)
    for i,ids in enumerate(groups):owner[ids]=i
    result=dict(s=g['s'],sf=g['sf'],pf=g['pf'],point_owner=g['owner'],point_weight=g['w'],patch_mass=g['mass'],patches=g['m'],
                sr=sr,sn=sn,sw=sw,pr=pr,pj=pj,pw=pw,phi=phi,pos=pos,length=length,
                graph_mass=gm,surface_mass=pm,has_g=has_g,has_p=has_p,adjacency=adjacency,coarse_owner=owner,
                nregions=u,ncoarse=len(groups),radius=float(g['config']['radius']),origin=g['origin'],
                raw_nodes=g['n'],raw_edges=len(g['e']),surface_points=g['source_points'],tile_count=g.get('tile_count',0))
    if query is not None:
        result['query']=tensor(np.asarray(query)-np.asarray(g['origin']))
    return result
