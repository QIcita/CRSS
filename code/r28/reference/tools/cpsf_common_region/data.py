"""Common arclength regions with observed-only geometry and disjoint batching."""
import gzip
import numpy as np
import torch
from tools.cpsf_v24_physical_conditioning.data import adapt
FIELDS=['s','sf','pf','point_owner','point_weight','patch_mass','sr','sn','sw','pr','pj','pw','phi','pos','length','graph_mass','surface_mass','has_g','has_p','adjacency','coarse_owner','patch_pos','query','radius']
def raw_graph(path):
 if str(path).endswith('.gz'):
  with gzip.open(path,'rb') as f:item=torch.load(f,map_location='cpu',weights_only=False)
  return item['g']
 return torch.load(path,map_location='cpu',weights_only=False)
def load_item(row):
 g=adapt(raw_graph(row['path']))
 with np.load(row['feature_path']) as a:
  g['pf']=torch.from_numpy(a['pf'].astype(np.float32));g['patch_pos']=torch.from_numpy(a['patch_pos'].copy())
 g['query']=g.get('query_xyz',g.get('query',torch.zeros(3))).reshape(1,3)
 g['radius']=torch.tensor([float(g['radius'])])
 # Cached labels and any diagnostic permutations never enter model inputs.
 return {k:g[k] for k in FIELDS}
def pack(items,device='cuda'):
 out={k:[] for k in FIELDS};nb=pb=rb=cb=0;sh=[];r_batch=[];radius_region=[]
 for b,g in enumerate(items):
  ns=len(g['s']);np_=len(g['patch_mass']);nr=len(g['pos']);nc=int(g['coarse_owner'].max())+1
  for k,v in g.items():
   if k in ['point_owner','pj']:v=v+pb
   elif k=='sn':v=v+nb
   elif k in ['sr','pr','adjacency']:v=v+rb
   elif k=='coarse_owner':v=v+cb
   out[k].append(v)
  sh.append((ns,np_,nr,nc));r_batch.append(torch.full((nr,),b,dtype=torch.long));radius_region.append(g['radius'].expand(nr))
  nb+=ns;pb+=np_;rb+=nr;cb+=nc
 result={k:torch.cat(v).to(device) for k,v in out.items()};result.update(shapes=sh,batch_size=len(items),nregions=rb,npatches=pb,ncoarse=cb,rbatch=torch.cat(r_batch).to(device),rradius=torch.cat(radius_region).to(device))
 return result
