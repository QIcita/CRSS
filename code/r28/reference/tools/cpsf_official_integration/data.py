import gzip
import numpy as np
import torch
from tools.cpsf_v24_physical_conditioning.data import adapt
FIELDS=['s','sf','pf','point_owner','point_weight','patch_mass','sr','sn','sw','pr','pj','pw','phi','pos','length','graph_mass','surface_mass','has_g','has_p','adjacency','coarse_owner','radius','ncoarse','nregions','patches','query','query_xyz']
def load_item(row):
 if row['task']=='ted':
  with gzip.open(row['path'],'rb') as f:g=torch.load(f,map_location='cpu',weights_only=False)['g']
 else:g=adapt(torch.load(row['path'],map_location='cpu',weights_only=False))
 g={k:g[k] for k in FIELDS if k in g}
 # Same old raw point features; sidecar only supplies previously audited coordinates.
 with np.load(row['feature_path']) as a:center=a['patch_pos'].astype(np.float64)
 pos=center[g['point_owner'].numpy()]+g['pf'].numpy()[:,:3].astype(np.float64)*.5
 origin=pos.min(0);grid=np.floor((pos-origin)/.5+1e-5).astype(np.int64)
 unique,inverse=np.unique(grid,axis=0,return_inverse=True);w=g['point_weight'].numpy().astype(np.float64);mass=np.bincount(inverse,weights=w,minlength=len(unique))
 pf=g['pf'].numpy().astype(np.float64);mu=np.zeros((len(unique),4));second=mu.copy();coord=np.zeros((len(unique),3))
 np.add.at(mu,inverse,pf*w[:,None]);np.add.at(second,inverse,pf**2*w[:,None]);np.add.at(coord,inverse,(pos-origin)*w[:,None]);mu/=mass[:,None];second/=mass[:,None];coord/=mass[:,None]
 g.update(ucoord=torch.tensor(coord/.5,dtype=torch.float32),ugrid=torch.from_numpy(unique),uinverse=torch.from_numpy(inverse),ufeat=torch.tensor(np.column_stack((mu,(second-mu**2).clip(0),np.log1p(mass))),dtype=torch.float32))
 # Exact matched-region partition; retain fine tokens, not coarse replacements.
 ids=torch.where(g['has_g']&g['has_p'])[0].numpy();owner=g['coarse_owner'].numpy();xyz=g['pos'].numpy();groups=[];small=[]
 for c in np.unique(owner[ids]):
  ix=ids[owner[ids]==c]
  if len(ix)<4:small.extend(ix.tolist());continue
  axis=np.ptp(xyz[ix],axis=0).argmax();ix=ix[np.argsort(xyz[ix,axis],kind='stable')]
  groups.extend(np.array_split(ix,int(np.ceil(len(ix)/128))))
 if len(small)>=4:groups.extend(np.array_split(np.array(small),int(np.ceil(len(small)/128))))
 elif small and groups:groups[-1]=np.r_[groups[-1],small]
 if len(ids)>=4:assert sorted(np.concatenate(groups).tolist())==sorted(ids.tolist())
 else:assert not groups
 g['geo_groups']=[torch.tensor(x,dtype=torch.long) for x in groups]
 return g

def move(g):return {k:v.cuda() if isinstance(v,torch.Tensor) else [z.cuda() for z in v] if k=='geo_groups' else v for k,v in g.items()}
