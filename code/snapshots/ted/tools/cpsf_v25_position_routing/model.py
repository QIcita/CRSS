"""Factor shared position content and scalar spatial incidence routing."""
import torch
from torch import nn
from tools.cpsf_v21_segment_units.model import add
from tools.cpsf_v23_scale_ablation.model import Model as Original

ARMS = ('baseline', 'no_shared_position', 'routing_additive', 'routing_only')
JOBS = [(task, arm) for arm in ('routing_only', 'baseline', 'no_shared_position', 'routing_additive')
        for task in ('local', 'global')]


class Model(Original):
    def __init__(self, task, arm):
        assert arm in ARMS
        super().__init__(task, 'full')
        self.variant = arm
        self.shared_position = arm in ('baseline', 'routing_additive')
        self.use_routing = arm in ('routing_additive', 'routing_only')
        # No RNG consumption, and no effect on the initialization of old weights.
        self.routing_weight = nn.Parameter(torch.zeros(4))

    def incidence_weights(self, g):
        """Feature attention only: leave empirical geometry masses untouched.

        Prior alpha_uj = pw_uj * patch_mass_j / surface_mass_u.
        New alpha is proportional to prior alpha * exp(2*tanh(w.phi/2)).
        Algebra below exactly recovers the old unnormalized weights at w=0.
        """
        wp = g['pw'] * g['patch_mass'][g['pj']]
        if not self.use_routing:
            return wp
        spatial = g['phi'][:, :4]  # d/r, (d/r)^2, axial, axial^2
        score = 2. * torch.tanh((spatial @ self.routing_weight) / 2.)
        factor = score.exp()
        relative_den = 1. + add(wp * (factor - 1.), g['pr'], g['nregions']) / g['surface_mass'].clamp_min(1e-12)
        return wp * factor / relative_den[g['pr']].clamp_min(1e-12)

    def routing_stats(self, g):
        wp = g['pw'] * g['patch_mass'][g['pj']]
        prior = wp / g['surface_mass'][g['pr']].clamp_min(1e-12)
        posterior = self.incidence_weights(g) / g['surface_mass'][g['pr']].clamp_min(1e-12)
        active = g['has_p']
        if not bool(active.any()):
            return dict(region_mean_tv=0., region_mean_kl=0.)
        tv = .5 * add((posterior - prior).abs(), g['pr'], g['nregions'])
        kl = add(posterior * (posterior.clamp_min(1e-12).log() - prior.clamp_min(1e-12).log()), g['pr'], g['nregions'])
        return dict(region_mean_tv=float(tv[active].mean()), region_mean_kl=float(kl[active].mean()))

    def stems(self, g):
        u = g['nregions']; hg = self.graph(g['sf']); px = self.point(g['pf'])
        den = g['patch_mass'].clamp_min(1e-12)[:, None]
        mu = add(px * g['point_weight'][:, None], g['point_owner'], g['patches']) / den
        second = add(px.square() * g['point_weight'][:, None], g['point_owner'], g['patches']) / den
        hp = self.patch(torch.cat((mu, (second - mu.square()).clamp_min(0)), -1))
        sr, sn = g['sr'], g['sn']; pr, pj = g['pr'], g['pj']
        path = (g['s'][sn] - g['pos'][sr]) / g['radius']
        graph = add((hg[sn] + self.path_position(path)) * g['sw'][:, None], sr, u) / g['graph_mass'].clamp_min(1e-12)[:, None]
        wp = self.incidence_weights(g)
        surface = add((hp[pj] + self.relation(g['phi'])) * wp[:, None], pr, u) / g['surface_mass'].clamp_min(1e-12)[:, None]
        xyz = g['pos']; xyz = xyz - xyz.mean(0)
        position = self.position(torch.sign(xyz) * torch.log1p(xyz.abs() / g['radius'])) if self.shared_position else torch.zeros_like(graph)
        # Preserve the V23 scale encoding exactly for this controlled experiment.
        scale = torch.stack((g['length'].log1p(), torch.full_like(g['length'], g['radius']).log1p(), torch.full_like(g['length'], .5).log1p()), 1)
        code = position + self.scale(scale)
        return (graph + code) * g['has_g'][:, None], (surface + code) * g['has_p'][:, None]
