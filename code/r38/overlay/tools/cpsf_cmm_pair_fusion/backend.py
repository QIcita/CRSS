import inspect
import torch
from .common import add_backend,sha
add_backend()
from tools.cpsf_r34_fps_rebench.backend import Backend as ParentBackend,digest,loss_for
from .model import install


class Backend(ParentBackend):
    def __init__(self,name,task,training):
        assert name in ('r34','r38') and task in ('h01_local','h01_global')
        super().__init__('r34',task,training)
        if name=='r38':
            cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
            install(self.model)
            assert torch.equal(cpu,torch.get_rng_state())
            assert all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
        self.name=name
        for module in self.model.modules():
            try:path=inspect.getfile(type(module))
            except (TypeError,OSError):continue
            if '/tools/cpsf_' in path:self.sources[path]=sha(path)
        self.sources[__file__]=sha(__file__)
