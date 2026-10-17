"""Original fixed core plus one official module; no pretrained weights."""
import sys,types,importlib,importlib.util
from torch.nn import functional as F
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tools.cpsf_fixed_core.model import H01Model
from tools.cpsf_fixed_nredd.model import Model as TEDModel
from tools.cpsf_v21_segment_units.model import add
from .geo_import import load_official_geotransformer
ROOT=Path(__file__).resolve().parents[2]

def native_utonia():
 name='official_integrated_utonia'
 if name not in sys.modules:
  ns=types.ModuleType(name);ns.__path__=[str(ROOT/'vendor/Utonia/utonia')];sys.modules[name]=ns
 return importlib.import_module(name+'.model')

class NativePoint(nn.Module):
 def __init__(self):
  super().__init__();native=native_utonia()
  self.encoder=native.PointTransformerV3(in_channels=9,order=('z','z-trans'),stride=(2,2),enc_depths=(1,1,2),enc_channels=(48,96,192),enc_num_head=(2,4,8),enc_patch_size=(128,128,128),enc_mode=True,freeze_encoder=False,shuffle_orders=False,enable_flash=True,drop_path=0.,rope_base=10,shift_coords=None,jitter_coords=None,rescale_coords=None)
  class NativeCheckpoint(native.PointModule):
   enabled=True
   def __init__(self,module):super().__init__();self.module=module
   def forward(self,point):
    if not self.enabled or not self.training or not torch.is_grad_enabled():return self.module(point)
    template=dict(point)
    def fn(feat):
     current=native.Point(dict(template));current.feat=feat;current.sparse_conv_feat=template['sparse_conv_feat'].replace_feature(feat)
     return self.module(current).feat
    feat=checkpoint(fn,point.feat,use_reentrant=False,preserve_rng_state=True);point.feat=feat;point.sparse_conv_feat=point.sparse_conv_feat.replace_feature(feat);return point
  for stage in self.encoder.enc:
   for name,layer in list(stage.named_children()):
    if isinstance(layer,native.Block):setattr(stage,name,NativeCheckpoint(layer))
  # Official per-pooling flag is independent of the encoder-level flag.
  for layer in self.encoder.modules():
   if isinstance(layer,native.GridPooling):layer.shuffle_orders=False
  self.proj=nn.Linear(336,96)
 def forward(self,g):
  # Input-side physical voxels are unique and carry all observed mass/moments.
  point=self.encoder(dict(coord=g['ucoord'],grid_coord=g['ugrid'],feat=g['ufeat'],offset=torch.tensor([len(g['ufeat'])],device=g['pf'].device)))
  while 'pooling_parent' in point:
   parent=point.pop('pooling_parent');inverse=point.pop('pooling_inverse');parent.feat=torch.cat((parent.feat,point.feat[inverse]),-1);point=parent
  return self.proj(point.feat)[g['uinverse']]

class NativeJoint(nn.Module):
 def __init__(self):
  super().__init__();cls=load_official_geotransformer(ROOT/'vendor/GeoTransformer')
  self.transformer=cls(input_dim=96,output_dim=96,hidden_dim=96,num_heads=4,blocks=['self','cross'],sigma_d=1.,sigma_a=15.,angle_k=3,reduction_a='max')
 def forward(self,graph,point,g):
  # Every matched fine region appears once. No zero padding in geometric kNN.
  gg=graph.clone();pp=point.clone()
  groups=g['geo_groups']
  for ix in groups:
   pos=(g['pos'][ix]-g['pos'][ix].mean(0))/g['radius']
   args=(pos[None],pos[None],graph[ix][None],point[ix][None])
   a,b=checkpoint(self.transformer,*args,use_reentrant=False) if torch.is_grad_enabled() else self.transformer(*args)
   gg[ix]=a[0];pp[ix]=b[0]
  return gg,pp

class NativeGlobal(nn.Module):
 """Unmodified official ICLR2024 Polynormer GlobalAttn, one within-cell layer."""
 def __init__(self):
  super().__init__()
  name='official_r8_polynormer'
  if name not in sys.modules:
   spec=importlib.util.spec_from_file_location(name,ROOT/'vendor/Polynormer/model.py')
   module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
  self.block=sys.modules[name].GlobalAttn(hidden_channels=24,heads=4,num_layers=1,beta=.5,dropout=0.,qk_shared=True)
  self.block.reset_parameters()
 def forward(self,x):
  return self.block(F.layer_norm(x,(x.shape[-1],)))

class Integration:
 def configure(self,kind):
  self.integration_kind=kind;self.integration_enabled=True
  # Preserve every old parameter and its initialization stream exactly.
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42)
   if kind=='utonia':self.native_point=NativePoint()
   if kind=='geotransformer':self.native_joint=NativeJoint()
 @staticmethod
 def rms_delta(reference,context,alpha):
  native_rms=context.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
  reference_rms=reference.detach().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
  return alpha*reference_rms*context/native_rms
 def patch_context_delta(self,hp,pooled):
  if self.adapter_variant=='patch_raw_010':return .1*pooled
  return self.rms_delta(hp,pooled,.3 if self.adapter_variant=='patch_rms_030' else .1)
 def stems(self,g):
  u=g['nregions'];hg=self.graph(g['sf']);px=self.point(g['pf'])
  context=self.native_point(g) if self.integration_enabled and self.integration_kind=='utonia' else None
  den=g['patch_mass'].clamp_min(1e-12)[:,None]
  mu=add(px*g['point_weight'][:,None],g['point_owner'],g['patches'])/den
  second=add(px.square()*g['point_weight'][:,None],g['point_owner'],g['patches'])/den
  hp=self.patch(torch.cat((mu,(second-mu.square()).clamp_min(0)),-1))
  if context is not None:
   pooled=add(context*g['point_weight'][:,None],g['point_owner'],g['patches'])/den
   hp=hp+self.patch_context_delta(hp,pooled)
  sr,sn=g['sr'],g['sn'];pr,pj=g['pr'],g['pj']
  path=(g['s'][sn]-g['pos'][sr])/g['radius']
  graph=add((hg[sn]+self.path_position(path))*g['sw'][:,None],sr,u)/g['graph_mass'].clamp_min(1e-12)[:,None]
  wp=g['pw']*g['patch_mass'][pj]
  surface=add((hp[pj]+self.relation(g['phi']))*wp[:,None],pr,u)/g['surface_mass'].clamp_min(1e-12)[:,None]
  xyz=g['pos']-g['pos'].mean(0)
  position=self.position(torch.sign(xyz)*torch.log1p(xyz.abs()/g['radius']))
  scale=torch.stack((g['length'].log1p(),torch.full_like(g['length'],g['radius']).log1p(),torch.full_like(g['length'],.5).log1p()),1)
  code=position+self.scale(scale)
  return (graph+code)*g['has_g'][:,None],(surface+code)*g['has_p'][:,None]
 def joint_regions(self,graph,point,g):
  if self.integration_enabled and self.integration_kind=='geotransformer':graph,point=self.native_joint(graph,point,g)
  matched=g['has_g']&g['has_p'];delta=torch.zeros_like(graph)
  if matched.any():delta[matched]=self.joint(torch.cat((graph[matched],point[matched]),-1))
  return delta
 def streams(self,g):
  graph,point=self.stems(g);gm=g['graph_mass'];pm=g['surface_mass'];mass=gm/gm.sum().clamp_min(1e-12)+pm/pm.sum().clamp_min(1e-12)
  active=g['has_g']|g['has_p'];count=(g['has_g'].float()+g['has_p'].float()).clamp_min(1)
  x=(graph+point)/count[:,None]+.1*self.joint_regions(graph,point,g)
  x=self.processor(x,g,mass,active)
  if self.integration_enabled and getattr(self,'global_enabled',False):
   # One sample per forward: all valid joint-region tokens, never cross-cell mixing.
   ix=torch.where(active)[0]
   delta=self.native_global(x[ix])
   x=x.index_add(0,ix,self.rms_delta(x[ix],delta,.1))
  return [(x,mass,active)]
 def embedding(self,g):
  x,mass,active=self.streams(g)[0]
  return self.readout(self.pool(x,g,mass,active))
class H01(Integration,H01Model):
 def __init__(self,task,kind):super().__init__(task,'full');self.configure(kind)
class TED(Integration,TEDModel):
 def __init__(self,task,kind):super().__init__(task,'full');self.configure(kind)
VARIANTS=('patch_rms_010','patch_raw_010','patch_rms_030','patch_rms_010_polynormer')
def Model(task,kind):
 assert kind in ('none',)+VARIANTS
 m=TED(task,'none' if kind=='none' else 'utonia') if task=='ted' else H01(task,'none' if kind=='none' else 'utonia')
 m.adapter_variant=kind;m.global_enabled=kind=='patch_rms_010_polynormer'
 if m.global_enabled:
  # Addition cannot perturb initialization or training RNG of existing modules.
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42);m.native_global=NativeGlobal()
 return m
