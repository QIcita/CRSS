import inspect
import torch
from .common import sha
from .model import install
from .data import Cache,move
from tools.cpsf_r34_fps_rebench.backend import Backend as ParentBackend,digest,loss_for

class Backend(ParentBackend):
    def __init__(self,name,task,training,force_parent=False):
        assert name=='r38' and task in ('h01_global','flywire_coarse')
        super().__init__('r34',task,training)
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
        install(self.model)
        assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
        self.name=name;self.cache=Cache(force_parent);self.move=move
        self.sources[__file__]=sha(__file__)
        for module in self.model.modules():
            try:path=inspect.getfile(type(module))
            except (TypeError,OSError):continue
            if '/tools/cpsf_' in path:self.sources[path]=sha(path)
    def item(self,row):return self.move(self.cache.get(row)['g']),row['target'],None
