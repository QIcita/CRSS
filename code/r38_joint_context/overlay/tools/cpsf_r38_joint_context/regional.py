"""Observed-region coupling with the original R38 background and residual paths."""
from dataclasses import dataclass
from typing import Mapping, NamedTuple

import torch
from torch import nn

from .support import add


@dataclass(frozen=True)
class RegionalSupport:
    """Borrow original incidences, modality masses, geometry and validity masks."""
    geometry: Mapping

    def graph_mean(self, nodes, path_position):
        g = self.geometry
        sr, sn = g['sr'], g['sn']
        path = (g['s'][sn] - g['pos'][sr]) / g['radius']
        return add((nodes[sn] + path_position(path)) * g['sw'][:, None], sr, g['nregions']) / g['graph_mass'].clamp_min(1e-12)[:, None]

    def surface_background(self, patches, relation):
        g = self.geometry
        pr, pj = g['pr'], g['pj']
        weight = g['pw'] * g['patch_mass'][pj]
        return add((patches[pj] + relation(g['phi'])) * weight[:, None], pr, g['nregions']) / g['surface_mass'].clamp_min(1e-12)[:, None]


class RegionalFeatures(NamedTuple):
    features: torch.Tensor
    mass: torch.Tensor
    active: torch.Tensor


class RegionalGeometry(nn.Module):
    """One owner for the existing, independently parameterized geometry maps."""
    def __init__(self, path_position, relation, position, scale):
        super().__init__()
        self.path_position = path_position
        self.relation = relation
        self.position = position
        self.scale = scale

    def region_code(self, support):
        g = support.geometry
        xyz = g['pos'] - g['pos'].mean(0)
        position = self.position(torch.sign(xyz) * torch.log1p(xyz.abs() / g['radius']))
        scale = torch.stack((g['length'].log1p(), torch.full_like(g['length'], g['radius']).log1p(), torch.full_like(g['length'], .5).log1p()), 1)
        return position + self.scale(scale)


class RegionalCoupling(nn.Module):
    """Region projection, conditional patch differences and original joint fusion.

    The transferred message module retains its signed-tanh gates, pure patch
    centering, 16384-incidence chunks and checkpointing. No parameter is added.
    """
    def __init__(self, geometry, message, joint):
        super().__init__()
        self.geometry = geometry
        self.message = message
        self.joint = joint

    def stems(self, nodes, patches, support, *, message_enabled=True, strategy='path_message'):
        g = support.geometry
        graph = support.graph_mean(nodes, self.geometry.path_position)
        surface = support.surface_background(patches.surface, self.geometry.relation)
        code = self.geometry.region_code(support)
        graph = (graph + code) * g['has_g'][:, None]
        point = (surface + code) * g['has_p'][:, None]
        if not message_enabled:
            return graph, point
        # Preserve the second graph node and the original backward sum order.
        # The message query reads graph content before the shared region code.
        raw_graph = support.graph_mean(nodes, self.geometry.path_position)
        delta = self.message(raw_graph, patches.message, g, strategy)
        delta = delta * (g['has_g'] & g['has_p'])[:, None]
        return graph, point + .1 * delta

    def joint_regions(self, graph, point, support):
        g = support.geometry
        matched = g['has_g'] & g['has_p']
        delta = torch.zeros_like(graph)
        if matched.any():
            delta[matched] = self.joint(torch.cat((graph[matched], point[matched]), -1))
        return delta

    def forward(self, nodes, patches, support, *, message_enabled=True, strategy='path_message'):
        graph, point = self.stems(nodes, patches, support, message_enabled=message_enabled, strategy=strategy)
        g = support.geometry
        gm = g['graph_mass']; pm = g['surface_mass']
        mass = gm / gm.sum().clamp_min(1e-12) + pm / pm.sum().clamp_min(1e-12)
        active = g['has_g'] | g['has_p']
        count = (g['has_g'].float() + g['has_p'].float()).clamp_min(1)
        x = (graph + point) / count[:, None] + .1 * self.joint_regions(graph, point, support)
        return RegionalFeatures(x, mass, active)
