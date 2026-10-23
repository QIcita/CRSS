"""Exactly R34 stage-one messages, attached to the existing six-task R28 heads."""
import numpy as np
import torch
from .frozen_r34 import PathMessage
from tools.cpsf_v21_segment_units.model import add
from tools.cpsf_r28_expansion.model import H01Check,Fly,Ted,TLP

class MessageMixin:
    def configure_message(self):
        self.strategy='path_message';self.message_enabled=True
        state=np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42);np.random.seed(42);self.region_message=PathMessage()
        finally:np.random.set_state(state)

    def patch_context_delta(self,hp,pooled):
        original=super().patch_context_delta(hp,pooled)
        self._message_patch=hp+original
        return original

    def stems(self,g):
        def capture(module,args,value):self._message_graph=value
        handle=self.graph.register_forward_hook(capture)
        try:
            graph,point=super().stems(g)
            if not self.message_enabled:return graph,point
            sr,sn=g['sr'],g['sn'];path=(g['s'][sn]-g['pos'][sr])/g['radius']
            raw_graph=add((self._message_graph[sn]+self.path_position(path))*g['sw'][:,None],
                          sr,g['nregions'])/g['graph_mass'].clamp_min(1e-12)[:,None]
            delta=self.region_message(raw_graph,self._message_patch,g,self.strategy)
            delta=delta*(g['has_g']&g['has_p'])[:,None]
            return graph,point+.1*delta
        finally:
            handle.remove()
            for name in ('_message_graph','_message_patch'):
                if hasattr(self,name):delattr(self,name)

class H01Model(MessageMixin,H01Check):
    def __init__(self,task):super().__init__(task);self.configure_message()
class FlyModel(MessageMixin,Fly):
    def __init__(self):super().__init__();self.configure_message()
class TEDModel(MessageMixin,Ted):
    def __init__(self):super().__init__();self.configure_message()
class TLPModel(MessageMixin,TLP):
    def __init__(self):super().__init__();self.configure_message()

def Model(task):
    if task.startswith('h01_'):return H01Model(task)
    if task in ('flywire_coarse','flywire_fine'):return FlyModel()
    if task=='ted':return TEDModel()
    if task=='tlp':return TLPModel()
    raise ValueError(task)
