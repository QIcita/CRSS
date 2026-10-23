import inspect,math
import numpy as np
import torch
from .common import ROOT,add_backend,read,sha
add_backend()
from tools.cpsf_geometry_rebench.backend import digest,loss_for

class Backend:
    def __init__(self,name,task,training):
        assert name in ('r28','r34')
        from tools.cpsf_raw_point_attention.common import setup
        from tools.cpsf_r28_expansion.train import Cache,eval_forward
        from tools.cpsf_r28_expansion.data import move
        from tools.cpsf_r28_expansion.model import Model as ParentModel
        from .model import Model
        setup();self.name=name;self.task=task;self.family='r28'
        native_task='flywire_coarse' if task=='flywire_fine' else task
        self.model=(Model(task) if name=='r34' else ParentModel(native_task)).cuda()
        self.cache=Cache();self.move=move
        self.scale=max(float(np.std([r['target'] for r in training])),.05) if task=='h01_global' else 1.
        self.mean=float(np.mean([r['target'] for r in training])) if task=='h01_global' else 0.
        if task=='h01_global':
            with torch.no_grad():self.model.head.bias.fill_(math.log(self.mean/(1-self.mean)))
        if task=='flywire_fine':
            old=self.model.head
            self.model.head=torch.nn.Linear(old.in_features,148,bias=old.bias is not None).cuda()
        self.eval_forward=lambda g:eval_forward(self.model,g,native_task)
        self.sources={__file__:sha(__file__)}
        for module in self.model.modules():
            try:path=inspect.getfile(type(module))
            except (TypeError,OSError):continue
            if '/vendor/' in path or '/tools/cpsf_' in path or '/runtime/' in path:self.sources[path]=sha(path)
    def item(self,row):
        r=dict(row,task='flywire_coarse',target=row['coarse_target']) if self.task=='flywire_fine' else row
        item=self.cache.get(r);g=self.move(item['g'])
        assert not {'target','label','fine_target','coarse_target','observed','split'}.intersection(g)
        y=item['target'].cuda() if self.task=='tlp' else row['target']
        observed=item['observed'].cuda() if self.task=='tlp' else None
        return g,y,observed
    def set_epoch(self,epoch,epochs):
        if hasattr(self.model,'set_epoch'):self.model.set_epoch(epoch,epochs)
