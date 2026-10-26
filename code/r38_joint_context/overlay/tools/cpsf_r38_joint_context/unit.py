"""One morphology-anchored module owning both context paths and interactions."""
from typing import NamedTuple

import torch
from torch import nn


def add(x, index, count):
    return x.new_zeros((count, *x.shape[1:])).index_add(0, index, x)


class PatchFeatures(NamedTuple):
    surface: torch.Tensor
    message: torch.Tensor


class JointContext(nn.Module):
    """D/C are internal paths of a single H-conditioned residual operator.

    Existing module objects are transferred here, never reinitialized. Keeping
    the original two RMS evaluations and addition order is intentional.
    """

    def __init__(self, hierarchy, neighborhood, interaction):
        super().__init__()
        self.hierarchy = hierarchy
        self.neighborhood = neighborhood
        self.interaction = interaction

    @staticmethod
    def rms_delta(reference, context, alpha):
        native_rms = context.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        reference_rms = reference.detach().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return alpha * reference_rms * context / native_rms

    def encode_hierarchy(self, g):
        # Called before the anchor statistics, as in the original R38 forward.
        return self.hierarchy(g)

    def forward(self, anchor, point_features, g, pooled_context):
        context_delta = self.rms_delta(anchor, pooled_context, .1)
        geometry = dict(g, pf=g['raw_geometry_pf']) if 'raw_geometry_pf' in g else g
        neighborhood = self.neighborhood(point_features, geometry, 'r9_lpa_attention')
        residual = context_delta + .1 * neighborhood
        if self.interaction.enabled:
            calibrated = self.rms_delta(anchor, pooled_context, 1.)
            residual = residual + self.interaction(anchor, neighborhood, calibrated)
        return residual


class ContextualMorphology(nn.Module):
    """Fine-patch anchor plus joint context, with one explicit model interface."""

    def __init__(self, morphology, context):
        super().__init__()
        self.morphology = morphology
        self.context = context

    def forward(self, point_features, g):
        context = self.context.encode_hierarchy(g)
        den = g['patch_mass'].clamp_min(1e-12)[:, None]
        weights = g['point_weight'][:, None]
        owner, count = g['point_owner'], g['patches']
        mean = add(point_features * weights, owner, count) / den
        second = add(point_features.square() * weights, owner, count) / den
        anchor = self.morphology(torch.cat((mean, (second - mean.square()).clamp_min(0)), -1))
        pooled = add(context * weights, owner, count) / den
        residual = self.context(anchor, point_features, g, pooled)
        # R38 creates two boundary additions, first for RCM then for the surface
        # path. Preserve both autograd nodes, not just their equal tensor values.
        message = anchor + residual
        surface = anchor + residual
        return PatchFeatures(surface, message)
