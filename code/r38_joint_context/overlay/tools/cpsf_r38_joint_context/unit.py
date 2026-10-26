"""One morphology-anchored module owning both context paths and interactions."""
from typing import NamedTuple

import torch
from torch import nn

from .support import SurfaceSupport, HierarchyView, add


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

    def encode_hierarchy(self, support):
        # Called before the anchor statistics, as in the original R38 forward.
        view = support.hierarchy_view() if isinstance(support, SurfaceSupport) else HierarchyView(support)
        return view.encode_points(self.hierarchy)

    def forward(self, anchor, support, pooled_context, legacy_context=None):
        # Retain the former internal (H, points, geometry, C) call for overlays
        # built against the first JointContext release.
        if legacy_context is not None:
            support, pooled_context = SurfaceSupport(pooled_context, support), legacy_context
        context_delta = self.rms_delta(anchor, pooled_context, .1)
        neighborhood = support.neighborhood_view().encode_patches(self.neighborhood, support.point_features)
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

    def forward(self, support, g=None):
        if g is not None:
            support = SurfaceSupport(g, support)
        context = self.context.encode_hierarchy(support)
        reduction = support.reduction()
        mean = reduction.mean(support.point_features)
        second = reduction.mean(support.point_features.square())
        anchor = self.morphology(torch.cat((mean, (second - mean.square()).clamp_min(0)), -1))
        pooled = reduction.mean(context)
        residual = self.context(anchor, support, pooled)
        # R38 creates two boundary additions, first for RCM then for the surface
        # path. Preserve both autograd nodes, not just their equal tensor values.
        message = anchor + residual
        surface = anchor + residual
        return PatchFeatures(surface, message)
