import inspect,math
import numpy as np
import torch
from .common import sha
from .model import Model
from .data import Cache,move
from tools.cpsf_raw_point_attention.common import setup
from tools.cpsf_geometry_rebench.backend import digest,loss_for
from tools.cpsf_r28_expansion.train import eval_forward

class Backend:
    def __init__(self,name,task,training,force_parent=False):
        setup();self.name=name;self.task=task;self.family='r28'
        self.model=Model(task).cuda();self.cache=Cache(force_parent);self.move=move
        self.mean=float(np.mean([r['target'] for r in training])) if task=='h01_global' else 0.
        self.scale=max(float(np.std([r['target'] for r in training])),.05) if task=='h01_global' else 1.
        if task=='h01_global':
            with torch.no_grad():self.model.head.bias.fill_(math.log(self.mean/(1-self.mean)))
        self.eval_forward=lambda g:eval_forward(self.model,g,task)
        self.sources={__file__:sha(__file__)}
        for module in self.model.modules():
            try:path=inspect.getfile(type(module))
            except (TypeError,OSError):continue
            if '/vendor/' in path or '/tools/cpsf_' in path or '/runtime/' in path:self.sources[path]=sha(path)
    def item(self,row):
        g=self.move(self.cache.get(row)['g'])
        assert not {'target','label','split','observed','fine_target','coarse_target'}.intersection(g)
        return g,row['target'],None
    def set_epoch(self,epoch,epochs):
        if hasattr(self.model,'set_epoch'):self.model.set_epoch(epoch,epochs)
