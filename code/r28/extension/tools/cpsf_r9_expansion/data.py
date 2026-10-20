import gzip
import numpy as np
import torch
from tools.cpsf_utonia_context.data import load_item as original_load, FIELDS, move

def raw(path):
    with gzip.open(path,'rb') as f:return torch.load(f,map_location='cpu',weights_only=False)

def load_item(row):
    if row['task']!='tlp' and row['task']!='flywire_coarse':return dict(g=original_load(row))
    item=raw(row['path']);g=item.get('g',item)
    from tools.cpsf_v24_physical_conditioning.data import adapt
    g=adapt(g)
    extra=['bridge_row','bridge_col','bridge_weight','pair_count','pair_lookup'] if row['task']=='tlp' else []
    out={k:g[k] for k in FIELDS+extra if k in g}
    # Coordinates MUST be reconstructed from the original voxel-relative pf,
    # never from TLP's directional neighborhood descriptor.
    with np.load(row['feature_path']) as a:center=a['patch_pos'].astype(np.float64)
    pos=center[g['point_owner'].numpy()]+g['pf'].numpy()[:,:3].astype(np.float64)*.5
    if row['task']=='tlp':out['pf']=torch.cat((g['pf_directional'][:,:3],g['pf'][:,3:]),1)
    origin=pos.min(0);grid=np.floor((pos-origin)/.5+1e-5).astype(np.int64)
    unique,inverse=np.unique(grid,axis=0,return_inverse=True);w=g['point_weight'].numpy().astype(np.float64);mass=np.bincount(inverse,weights=w,minlength=len(unique))
    pf=out['pf'].numpy().astype(np.float64);mu=np.zeros((len(unique),4));second=mu.copy();coord=np.zeros((len(unique),3))
    np.add.at(mu,inverse,pf*w[:,None]);np.add.at(second,inverse,pf**2*w[:,None]);np.add.at(coord,inverse,(pos-origin)*w[:,None]);mu/=mass[:,None];second/=mass[:,None];coord/=mass[:,None]
    out.update(ucoord=torch.tensor(coord/.5,dtype=torch.float32),ugrid=torch.from_numpy(unique),uinverse=torch.from_numpy(inverse),ufeat=torch.tensor(np.column_stack((mu,(second-mu**2).clip(0),np.log1p(mass))),dtype=torch.float32))
    result=dict(g=out)
    if row['task']=='tlp':result.update(target=item['target'],observed=item['observed'],pairs=item['pairs'])
    return result
