"""Use original frozen networks and modality adapters, never surrogate modules."""
import importlib,inspect,math,gzip,io
import numpy as np
import torch
from .common import ROOT,BASE,FAMILIES,add_backend,read,sha

class Backend:
    def __init__(self,name,task,training):
        self.name=name;self.task=task
        family='nredd' if task in ('ted','tlp') and name!='r28' else FAMILIES[name]
        add_backend(family);self.family=family
        prefix=dict(r28='cpsf_raw_point_attention',phase2='cpsf_external_phase2',phase3='cpsf_external_phase3',
            utonia='cpsf_utonia_scratch_r2',graphformer='cpsf_graphformer',nredd='cpsf_nredd_baselines')[family]
        common=importlib.import_module('tools.'+prefix+'.common');common.setup()
        self.scale=max(float(np.std([r['target'] for r in training])),.05) if task=='h01_global' else 1.
        self.mean=float(np.mean([r['target'] for r in training])) if task=='h01_global' else 0.
        self.native_rows={}
        if family=='r28':
            from tools.cpsf_r28_expansion.model import Model
            from tools.cpsf_r28_expansion.train import Cache,eval_forward
            from tools.cpsf_r28_expansion.data import move
            self.model=Model(task).cuda();self.cache=Cache();self.move=move
            self.eval_forward=lambda g:eval_forward(self.model,g,task)
        elif family in ('phase2','phase3','utonia'):
            mod=importlib.import_module('tools.'+prefix+'.models');data=importlib.import_module('tools.'+prefix+'.data')
            self.model=mod.make_model(name,task).cuda()
            class SharedCache(data.GeometryCache):
                def build(self,row):
                    if name not in ('sgformer','polynormer'):
                        opener=gzip.open if row['path'].endswith('.gz') else open
                        with opener(row['path'],'rb') as f:item=torch.load(f,map_location='cpu',weights_only=False)
                        g=item.get('g',item)
                        with np.load(row['feature_path']) as a:centres=a['patch_pos'].astype(np.float64)
                        coord=centres[g['point_owner'].numpy()]+g['pf'][:,:3].numpy().astype(np.float64)*.5
                        query=g['query'].numpy() if task=='h01_local' else np.zeros(3,np.float32)
                        buffer=io.BytesIO();np.savez(buffer,coord=coord,mass=g['point_weight'].numpy(),query=query);buffer.seek(0)
                        return super().build(dict(row,point_path=buffer))
                    return super().build(row)
            self.cache=SharedCache(name,task)
            self.move=(lambda x:data.to_device(x,name)) if family=='phase2' else data.to_device
            self.eval_forward=lambda g:self.model(g)
            # Current-input BN, already used in the audited final comparisons.
            for m in self.model.modules():
                if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
                    m.track_running_stats=False;m.running_mean=None;m.running_var=None;m.num_batches_tracked=None
        elif family=='graphformer':
            mod=importlib.import_module('tools.cpsf_graphformer.models');self.model=mod.Model(task).cuda()
            self.native_rows={r['key']:r for r in read(BASE/'CPSF_GRAPHFORMER_ADAPTED_20261013/data/manifest.json')['records'] if r['task']==task}
            self.stats=read(BASE/'CPSF_GRAPHFORMER_ADAPTED_20261013/data/normalization.json')[task]
            self.load_graphformer=mod.load_sample;self.eval_forward=lambda g:self.model(g)
        else:
            from tools.cpsf_nredd_baselines.models import make
            from tools.cpsf_nredd_baselines.data import Cache
            self.model=make(name,task).cuda();self.cache=Cache(name,task)
            self.native_rows={r['key']:r for r in read(BASE/'CPSF_NREDD_BASELINES_20261014/data'/task/'manifest.json')['records']}
            self.eval_forward=lambda g:self.model(g)
        if task=='h01_global':
            if hasattr(self.model,'initialize_regression'):self.model.initialize_regression(self.mean)
            else:
                with torch.no_grad():self.model.head.bias.fill_(math.log(self.mean/(1-self.mean)))
        self.sources={}
        for module in self.model.modules():
            try:path=inspect.getfile(type(module))
            except (TypeError,OSError):continue
            if '/vendor/' in path or '/tools/cpsf_' in path or '/runtime/' in path:self.sources[path]=sha(path)
        self.set_epoch(1,read(ROOT/'protocol.json')['tasks'][task]['epochs'])

    def item(self,row):
        if self.family=='r28':
            item=self.cache.get(row);g=self.move(item['g'])
            y=item['target'].cuda() if self.task=='tlp' else row['target']
            observed=item['observed'].cuda() if self.task=='tlp' else None
        elif self.family=='graphformer':
            r=self.native_rows[row['key']];assert r['target']==row['target'] and r['split']==row['split']
            g=self.load_graphformer(r,self.stats);y=row['target'];observed=None
        elif self.family=='nredd':
            r=self.native_rows[row['key']];assert r['split']==row['split']
            g,y,observed=self.cache.get(r)
            if self.task=='ted':assert int(y.argmax())==row['target'];y=row['target']
        else:g=self.move(self.cache.get(row));y=row['target'];observed=None
        if isinstance(g,dict):assert not {'target','label','observed','split'}.intersection(g)
        return g,y,observed

    def set_epoch(self,epoch,epochs):
        if hasattr(self.model,'set_epoch'):self.model.set_epoch(epoch,epochs)

def digest(model):
    import hashlib
    h=hashlib.sha256()
    for n,v in model.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def loss_for(z,y,task,scale):
    from torch.nn import functional as F
    if task=='tlp':
        mask=torch.triu(torch.ones_like(y,dtype=torch.bool),1);zz=z[mask];yy=y[mask]
        positives=yy.sum();weight=torch.ones_like(yy)
        if 0<positives<len(yy):weight[yy>0]=(len(yy)-positives)/positives
        return F.binary_cross_entropy_with_logits(zz,yy,weight=weight)
    if task=='ted':return F.binary_cross_entropy_with_logits(z,F.one_hot(torch.tensor(int(y),device=z.device),2).float())*(5 if y else 1)
    if task=='h01_global':return ((z-float(y))/scale).square()
    return F.cross_entropy(z[None],torch.tensor([int(y)],device=z.device),label_smoothing=.05)
