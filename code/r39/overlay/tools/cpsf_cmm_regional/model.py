"""Explicit R39 data flow. Official Utonia/SPoTr and downstream processor retained."""
import numpy as np
import torch
from tools.cpsf_fixed_core.model import H01Model as H01Base, FlyModel as FlyBase
from tools.cpsf_utonia_context.model import NativePoint
from tools.cpsf_r38_dense.model import DenseRefinement
from .operator import RegionalIntegration, add


class RegionalMixin:
    def configure_regional(self):
        # Keep the shared stems/processor/heads and their initial state unchanged.
        # Old relation addition and joint MLP are superseded by one operator.
        del self.relation
        del self.joint
        state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                self.native_point = NativePoint()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                np.random.seed(42)
                self.raw_refine = DenseRefinement()
                # D is a separately routed descriptor, no longer a zero-start
                # outer correction. The official local block is unchanged.
                torch.nn.init.eye_(self.raw_refine.output.weight)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                self.cmm = RegionalIntegration()
        finally:
            np.random.set_state(state)

    def encode_regions(self, g):
        n = g['nregions']
        hg, px = self.graph(g['sf']), self.point(g['pf'])
        den = g['patch_mass'].clamp_min(1e-12)[:, None]
        mu = add(px*g['point_weight'][:, None], g['point_owner'], g['patches']) / den
        second = add(px.square()*g['point_weight'][:, None], g['point_owner'], g['patches']) / den
        h = self.patch(torch.cat((mu, (second-mu.square()).clamp_min(0)), -1))
        d = self.raw_refine(px, g, 'r9_lpa_attention')
        context = self.native_point(g)
        c = add(context*g['point_weight'][:, None], g['point_owner'], g['patches']) / den
        sr, sn = g['sr'], g['sn']
        path = (g['s'][sn] - g['pos'][sr]) / g['radius']
        graph = add((hg[sn] + self.path_position(path))*g['sw'][:, None], sr, n)
        graph = graph / g['graph_mass'].clamp_min(1e-12)[:, None]
        x = self.cmm(graph, h, d, c, g)
        xyz = g['pos'] - g['pos'].mean(0)
        code = self.position(torch.sign(xyz)*torch.log1p(xyz.abs()/g['radius']))
        scale = torch.stack((g['length'].log1p(), torch.full_like(g['length'], g['radius']).log1p(),
                             torch.full_like(g['length'], .5).log1p()), 1)
        return x + code + self.scale(scale)

    def streams(self, g):
        gm, pm = g['graph_mass'], g['surface_mass']
        mass = gm/gm.sum().clamp_min(1e-12) + pm/pm.sum().clamp_min(1e-12)
        active = g['has_g'] | g['has_p']
        x = self.processor(self.encode_regions(g)*active[:, None], g, mass, active)
        return [(x, mass, active)]

    def embedding(self, g):
        x, mass, active = self.streams(g)[0]
        return self.readout(self.pool(x, g, mass, active))


class H01(RegionalMixin, H01Base):
    def __init__(self, task):
        super().__init__(task, 'full')
        self.configure_regional()


class Fly(RegionalMixin, FlyBase):
    def __init__(self):
        super().__init__('flywire_coarse', 'full')
        self.configure_regional()


def Model(task):
    if task in ('h01_local', 'h01_global'):
        return H01(task)
    if task == 'flywire_coarse':
        return Fly()
    raise ValueError(task)
