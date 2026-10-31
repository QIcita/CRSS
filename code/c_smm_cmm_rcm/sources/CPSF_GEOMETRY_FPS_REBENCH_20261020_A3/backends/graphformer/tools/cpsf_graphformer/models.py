"""Official GraPHFormer backbones, with recorded compatibility repairs only."""
import importlib.util,sys,types,json
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import ROOT,sha

def load_tree_module():
 p=ROOT/'runtime/tree_encoder.py'
 spec=importlib.util.spec_from_file_location('graphformer_tree_compat',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

class Model(nn.Module):
 def __init__(self,task):
  super().__init__();self.task=task
  sys.path[:0]=[str(ROOT/'vendor/upstream'),str(ROOT/'vendor/dinov2')]
  from graphformer.models.image_encoder import DINOv2ImageEncoder
  from graphformer.models.finetune_model import FineTuneModel
  from dinov2.hub.backbones import dinov2_vits14
  def scratch(repo,variant,*a,**kw):
   assert repo=='facebookresearch/dinov2' and variant=='dinov2_vits14' and not a and not kw
   return dinov2_vits14(pretrained=False)
  def forbidden(*a,**kw):raise AssertionError('External weights forbidden')
  with patch('torch.hub.load',side_effect=scratch),patch('torch.hub.load_state_dict_from_url',side_effect=forbidden):
   image=DINOv2ImageEncoder(freeze_backbone=False,model_variant='dinov2_vits14')
  tree=load_tree_module().TreeLSTMDouble(x_size=5,h_size=256,num_classes=0,fc=False,bn=False,mode='sum')
  shell=nn.Module();shell.tree_encoder=tree;shell.image_encoder=image;shell.tree_encoder_type='double'
  nc=1 if task=='h01_global' else 6 if task=='flywire_coarse' else 3
  self.native=FineTuneModel(shell,nc,mode='multimodal',fusion_mode='concat',use_projection=False,freeze_encoders=False)
  dim=896 if task=='h01_local' else 640
  # Only the task readout is replaced: singleton-input inference has no cross-
  # sample BN and no label is accepted by forward. Official encoders unchanged.
  self.native.classifier=nn.Linear(dim,nc)
  self.external_pretrained=False
 def initialize_regression(self,mean):
  if self.task=='h01_global':
   nn.init.normal_(self.native.classifier.weight,std=.001)
   nn.init.constant_(self.native.classifier.bias,float(np.log(mean/(1-mean))))
 def forward(self,x):
  h=self.native.encode_tree(x['batch'])
  pooled=(h*x['mass'][:,None]).sum(0)/x['mass'].sum()
  if self.task=='h01_local':pooled=torch.cat([pooled,h[x['query_component']]])
  image=self.native.encode_image(x['image'][None])[0]
  features=torch.cat([F.normalize(pooled,dim=-1),F.normalize(image,dim=-1)])
  z=self.native.classifier(features)
  return z[0].sigmoid() if self.task=='h01_global' else z

def load_sample(row,stats,device='cuda'):
 import dgl
 with np.load(row['adapted_path']) as x:
  feat=(x['features']-np.asarray(stats['mean']))/np.asarray(stats['std'])
  e=x['edges'];roots=x['roots'].copy();mass=x['component_mass'].copy();q=int(x['query_component']);pi=x['image'].copy()
 # Explicit num_nodes includes all singleton components.
 graph=dgl.graph((e[:,0],e[:,1]),num_nodes=len(feat),device=device)
 batch=types.SimpleNamespace(graph=graph,feats=torch.as_tensor(feat,dtype=torch.float32,device=device),offset=torch.as_tensor(roots,device=device))
 return dict(batch=batch,mass=torch.as_tensor(mass,device=device),query_component=q,image=torch.as_tensor(pi,device=device))

def gradients(model):
 groups={'tree_mlp':'native.tree_encoder.mlp1','tree_W1':'native.tree_encoder.cell.W1',
  'tree_U1':'native.tree_encoder.cell.U1','tree_W2':'native.tree_encoder.cell.W2','tree_U2':'native.tree_encoder.cell.U2',
  'image_patch':'native.image_encoder.backbone.patch_embed','head':'native.classifier'}
 for i in range(12):groups['image_block_'+str(i)]='native.image_encoder.backbone.blocks.'+str(i)+'.'
 report={}
 for group,prefix in groups.items():
  gs=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
  assert gs and all(torch.isfinite(g).all() for g in gs),(group,'missing or nonfinite')
  value=sum(float(g.abs().sum()) for g in gs);assert value>0,(group,'zero gradients');report[group]=value
 return report

def manifest(model):
 return dict(tree_class=type(model.native.tree_encoder).__name__,tree_file=str(ROOT/'runtime/tree_encoder.py'),
  upstream_tree_sha256=sha(ROOT/'vendor/upstream/graphformer/models/tree_encoder.py'),
  runtime_tree_sha256=sha(ROOT/'runtime/tree_encoder.py'),image_class=type(model.native.image_encoder).__name__,
  image_backbone_class=type(model.native.image_encoder.backbone).__name__,image_blocks=len(model.native.image_encoder.backbone.blocks),
  image_width=model.native.image_encoder.backbone.embed_dim,external_pretrained=False,parameters=sum(p.numel() for p in model.parameters()),
  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
