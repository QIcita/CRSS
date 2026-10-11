"""Whole-cell adapter using the unchanged V19 encoder, separate coarse/fine heads."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tools.cpsf_v19_measure_core.geometry import prepare,Config,point_view,skeleton_view,relations,pack
from tools.cpsf_v19_measure_core.model import Encoder

def read_views(path,arm):
 views=[]
 with np.load(path,allow_pickle=False) as x:
  for i in range(int(x['view_count'])):
   k=f'v{i:02d}_';p=x[k+'surface_xyz_um'];weights=x[k+'surface_weight_um2']
   # Pure surface path never even reads skeleton arrays from disk.
   if arm=='point':s=np.empty((0,3));e=np.empty((0,2),np.int64)
   else:s=x[k+'skeleton_xyz_um'];e=x[k+'edge_index']
   g=prepare(s,e,p,Config(),weights=weights,wrong=False)
   if arm=='point':g['query']=relations(g['c'].numpy(),np.empty((0,2),int),g['c'].numpy(),Config());g['query_n']=g['m']
   elif arm=='skeleton':g=skeleton_view(g)
   g['view_area']=float(weights.sum());views.append(g)
 return views

class CellModel(nn.Module):
 def __init__(self,classes,arm='measure'):
  super().__init__();assert arm in ('measure','count','skeleton','point','geometry');self.arm=arm;self.encoder=Encoder();self.head=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,classes))
 def pool_view(self,g):
  packed,_=pack([g],next(self.parameters()).device);h,p,null=self.encoder(packed,self.arm)
  if self.arm=='point':hp=(h*packed['mass'][:,None]).sum(0)/packed['mass'].sum().clamp_min(1e-12)
  else:hp=h.mean(0) if len(h) else p.new_zeros(64)
  pp=(p*packed['mass'][:,None]).sum(0)/packed['mass'].sum().clamp_min(1e-12)
  return torch.cat((hp,pp))
 def embedding(self,views):
  features=[]
  for g in views:
   f=checkpoint(lambda g=g:self.pool_view(g),use_reentrant=False) if self.training and torch.is_grad_enabled() else self.pool_view(g)
   features.append(f)
  if len(features)==1:return features[0]
  # Same global/local context mixture as H01; explicit area weights for local surfaces.
  local=torch.stack(features[1:]);mass=local.new_tensor([g['view_area'] for g in views[1:]])
  h=(features[0][:64]+local[:,:64].mean(0))/2;p=(features[0][64:]+(local[:,64:]*mass[:,None]).sum(0)/mass.sum())/2
  return torch.cat((h,p))
 def forward(self,views):return self.head(self.embedding(views)[None])

def require_ready(root):
 root=Path(root);ready=json.loads((root/'dataset/READY.json').read_text());assert ready['ready'] and ready['seed']==42
 rows=[json.loads(line) for line in (root/'dataset/frozen_manifest.jsonl').read_text().splitlines()]
 mapping=json.loads((root/'dataset/labels.json').read_text());return rows,mapping
