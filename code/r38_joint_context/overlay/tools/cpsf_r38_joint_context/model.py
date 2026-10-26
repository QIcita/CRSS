"""Explicit CMM -> regional aggregation -> RCM, without transient forward hooks."""
import torch

from .compatibility import CompatibilityMixin, rename
from .unit import ContextualMorphology, JointContext, add
from .support import SurfaceSupport


class JointContextModel(CompatibilityMixin):
    def stems(self, g):
        if not self.integration_enabled or self.integration_kind != 'utonia':
            raise ValueError('This refactor implements the fixed R38 Utonia integration.')
        u = g['nregions']
        hg = self.graph(g['sf'])
        px = self.point(g['pf'])
        patches = self.cmm(SurfaceSupport(g, px))
        sr, sn = g['sr'], g['sn']
        pr, pj = g['pr'], g['pj']
        path = (g['s'][sn] - g['pos'][sr]) / g['radius']
        graph = add((hg[sn] + self.path_position(path)) * g['sw'][:, None], sr, u) / g['graph_mass'].clamp_min(1e-12)[:, None]
        wp = g['pw'] * g['patch_mass'][pj]
        surface = add((patches.surface[pj] + self.relation(g['phi'])) * wp[:, None], pr, u) / g['surface_mass'].clamp_min(1e-12)[:, None]
        xyz = g['pos'] - g['pos'].mean(0)
        position = self.position(torch.sign(xyz) * torch.log1p(xyz.abs() / g['radius']))
        scale = torch.stack((g['length'].log1p(), torch.full_like(g['length'], g['radius']).log1p(), torch.full_like(g['length'], .5).log1p()), 1)
        code = position + self.scale(scale)
        graph = (graph + code) * g['has_g'][:, None]
        point = (surface + code) * g['has_p'][:, None]
        if not self.message_enabled:
            return graph, point
        # Preserve the second path-position evaluation and separate graph node
        # used by the original R34/R38 RCM path (including backward arithmetic).
        path = (g['s'][sn] - g['pos'][sr]) / g['radius']
        raw_graph = add((hg[sn] + self.path_position(path)) * g['sw'][:, None], sr, g['nregions']) / g['graph_mass'].clamp_min(1e-12)[:, None]
        delta = self.region_message(raw_graph, patches.message, g, self.strategy)
        delta = delta * (g['has_g'] & g['has_p'])[:, None]
        return graph, point + .1 * delta


def install(model):
    """Transfer existing R38 dense modules in place, before creating an optimizer.

    Construct through the original backend first to preserve initialization,
    task-head setup and the exact dense-input path. Existing model/optimizer
    state dictionaries remain loadable in the preserved parameter order.
    """
    if isinstance(model, JointContextModel):
        return model
    from tools.cpsf_r38_dense.model import H01Dense, FlyDense
    if type(model) not in (H01Dense, FlyDense):
        raise TypeError('Expected the original full R38 dense H01/FlyWire model.')
    if model.adapter_variant != 'patch_rms_010' or not model.integration_enabled:
        raise ValueError('Expected the unchanged full R38 integration settings.')
    state_order = tuple(model.state_dict())
    parameter_order = tuple(rename(name) for name, _ in model.named_parameters())
    morphology = model.patch
    context = JointContext(model.native_point, model.raw_refine, model.cmm_pair_fusion)
    cmm = ContextualMorphology(morphology, context)
    # Wrapper modules have no parameters/buffers and consume no RNG.
    cmm.training = model.training
    context.training = model.training
    for name in ('patch', 'native_point', 'raw_refine', 'cmm_pair_fusion'):
        delattr(model, name)
    model.add_module('cmm', cmm)
    parent = type(model)
    cls_name = 'Joint' + parent.__name__
    cls = globals().get(cls_name)
    if cls is None:
        cls = type(cls_name, (JointContextModel, parent), {'__module__': __name__})
        globals()[cls_name] = cls
    model.__class__ = cls
    model._r38_parameter_order = parameter_order
    model._r38_state_order = state_order
    return model
