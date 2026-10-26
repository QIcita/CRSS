"""Explicit CMM -> RCM -> original regional processor and task readout."""
import torch

from .compatibility import CompatibilityMixin, rename
from .unit import ContextualMorphology, JointContext
from .support import SurfaceSupport
from .regional import RegionalSupport, RegionalGeometry, RegionalCoupling


class JointContextModel(CompatibilityMixin):
    def _region_inputs(self, g):
        if not self.integration_enabled or self.integration_kind != 'utonia':
            raise ValueError('This refactor implements the fixed R38 Utonia integration.')
        hg = self.graph(g['sf'])
        px = self.point(g['pf'])
        patches = self.cmm(SurfaceSupport(g, px))
        return hg, patches, RegionalSupport(g)

    def stems(self, g):
        return self.rcm.stems(*self._region_inputs(g), message_enabled=self.message_enabled, strategy=self.strategy)

    def joint_regions(self, graph, point, g):
        return self.rcm.joint_regions(graph, point, RegionalSupport(g))

    def streams(self, g):
        regions = self.rcm(*self._region_inputs(g), message_enabled=self.message_enabled, strategy=self.strategy)
        x, mass, active = regions
        x = self.processor(x, g, mass, active)
        if self.integration_enabled and getattr(self, 'global_enabled', False):
            ix = torch.where(active)[0]
            delta = self.native_global(x[ix])
            x = x.index_add(0, ix, self.rms_delta(x[ix], delta, .1))
        return [(x, mass, active)]


def install(model):
    """Transfer existing R38 dense modules in place, before creating an optimizer.

    Construct through the original backend first to preserve initialization,
    task-head setup and the exact dense-input path. Existing model/optimizer
    state dictionaries remain loadable in the preserved parameter order.
    """
    if isinstance(model, JointContextModel) and hasattr(model, 'rcm'):
        return model
    from tools.cpsf_r38_dense.model import H01Dense, FlyDense
    if not isinstance(model, (H01Dense, FlyDense)):
        raise TypeError('Expected the original full R38 dense H01/FlyWire model.')
    if model.adapter_variant != 'patch_rms_010' or not model.integration_enabled:
        raise ValueError('Expected the unchanged full R38 integration settings.')
    state_order = tuple(model.legacy_state_dict()) if hasattr(model, 'legacy_state_dict') else tuple(model.state_dict())
    parameter_order = tuple(rename(name) for name, _ in model.named_parameters())
    if not hasattr(model, 'cmm'):
        context = JointContext(model.native_point, model.raw_refine, model.cmm_pair_fusion)
        cmm = ContextualMorphology(model.patch, context)
        cmm.training = model.training
        context.training = model.training
        for name in ('patch', 'native_point', 'raw_refine', 'cmm_pair_fusion'):
            delattr(model, name)
        model.add_module('cmm', cmm)
    geometry = RegionalGeometry(model.path_position, model.relation, model.position, model.scale)
    rcm = RegionalCoupling(geometry, model.region_message, model.joint)
    geometry.training = model.training
    rcm.training = model.training
    for name in ('path_position', 'relation', 'position', 'scale', 'region_message', 'joint'):
        delattr(model, name)
    model.add_module('rcm', rcm)
    parent = H01Dense if isinstance(model, H01Dense) else FlyDense
    cls_name = 'Joint' + parent.__name__
    cls = globals().get(cls_name)
    if cls is None:
        cls = type(cls_name, (JointContextModel, parent), {'__module__': __name__})
        globals()[cls_name] = cls
    model.__class__ = cls
    model._r38_parameter_order = parameter_order
    model._r38_state_order = state_order
    return model
