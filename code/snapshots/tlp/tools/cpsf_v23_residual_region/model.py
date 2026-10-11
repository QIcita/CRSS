"""Preserve region evidence; learn only a matched-modality residual.

Geometry, position/scale stems, Processor and task pooling are frozen in design
to V22. No new trainable parameters, auxiliary objectives or support changes.
"""
import torch
from tools.cpsf_v22_shared_region.model import Model as V22Model

ARMS = ('full', 'mean_only', 'no_fusion', 'point', 'skeleton')


class Model(V22Model):
    def __init__(self, task, arm):
        assert arm in ARMS
        super().__init__(task, 'full' if arm == 'mean_only' else arm)
        self.arm = arm

    def combine(self, graph, point, has_g, has_p):
        count = (has_g.to(graph.dtype) + has_p.to(graph.dtype)).clamp_min(1)
        base = (graph + point) / count[:, None]
        if self.arm == 'mean_only':
            return base
        matched = has_g & has_p
        # No learned cross-modal correction where only one modality is present.
        delta = torch.zeros_like(base)
        if matched.any():
            delta[matched] = self.joint(torch.cat((graph[matched], point[matched]), -1))
        return base + .1 * delta

    def forward(self, g):
        if self.arm not in ('full', 'mean_only'):
            return super().forward(g)
        graph, point = self.stems(g)
        gm, pm = g['graph_mass'], g['surface_mass']
        mass = gm / gm.sum().clamp_min(1e-12) + pm / pm.sum().clamp_min(1e-12)
        active = g['has_g'] | g['has_p']
        x = self.combine(graph, point, g['has_g'], g['has_p'])
        x = self.processor(x, g, mass, active)
        z = self.pool(x, g, mass, active)
        y = self.head(self.readout(z))
        return y if self.task == 'local' else y.sigmoid().squeeze(-1)
