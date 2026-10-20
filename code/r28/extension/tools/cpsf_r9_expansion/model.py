from tools.cpsf_utonia_context.model import Integration, Model as R9Model
from tools.cpsf_fixed_core.model import FlyModel
from tools.cpsf_directional.model import Model as DirectionalModel

class Fly(Integration,FlyModel):
    def __init__(self):
        super().__init__('flywire_coarse','full');self.configure('utonia')

class TLP(Integration,DirectionalModel):
    def __init__(self):
        super().__init__('tlp','full','directional_xyz');self.configure('utonia')
    # Integration stems operates on the input descriptor already selected in data.py.
    # DirectionalModel's deterministic corridor/query/pair forward is inherited.

def Model(task):
    if task in ('h01_local','h01_global','ted'):return R9Model(task,'patch_rms_010')
    m=Fly() if task=='flywire_coarse' else TLP()
    m.adapter_variant='patch_rms_010';m.global_enabled=False
    return m
