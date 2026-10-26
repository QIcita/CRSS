"""Fine-patch carrier and parameter-free views of the same point observations."""
from dataclasses import dataclass
from typing import Mapping

import torch


def add(x, index, count):
    return x.new_zeros((count, *x.shape[1:])).index_add(0, index, x)


@dataclass(frozen=True)
class PatchReduction:
    owner: torch.Tensor
    count: int
    weights: torch.Tensor
    denominator: torch.Tensor

    def mean(self, points):
        # Retain point order and the original index_add/division arithmetic.
        return add(points * self.weights, self.owner, self.count) / self.denominator


@dataclass(frozen=True)
class HierarchyView:
    geometry: Mapping

    def encode_points(self, encoder):
        # NativePoint performs the original voxel -> point gather internally.
        return encoder(self.geometry)


@dataclass(frozen=True)
class NeighborhoodView:
    geometry: Mapping

    def encode_patches(self, encoder, point_features):
        # DenseRefinement retains its own computation-group -> point -> patch
        # return, including the original no-expansion path and float32 casts.
        return encoder(point_features, self.geometry, 'r9_lpa_attention')


@dataclass(frozen=True)
class SurfaceSupport:
    """One fine-patch carrier; points and weights are never discarded.

    Geometry contains deterministic cached views of the same observations,
    prepared by data.prepare_surface_inputs or the original R38 loader. Views
    borrow tensors without copying, detaching or changing their order.
    """
    geometry: Mapping
    point_features: torch.Tensor

    def hierarchy_view(self):
        return HierarchyView(self.geometry)

    def neighborhood_view(self):
        g = self.geometry
        geometry = dict(g, pf=g['raw_geometry_pf']) if 'raw_geometry_pf' in g else g
        return NeighborhoodView(geometry)

    def reduction(self):
        g = self.geometry
        # Construct these after the hierarchy call, as in the original CMM.
        denominator = g['patch_mass'].clamp_min(1e-12)[:, None]
        weights = g['point_weight'][:, None]
        return PatchReduction(g['point_owner'], g['patches'], weights, denominator)
