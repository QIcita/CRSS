"""Explicit SMM / CMM / RCM with an isolated 256-parameter SMM interaction."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .common import ARMS, setup
from tools.cpsf_topology_surface.models import Model as Parent, add


PREFIXES = (
    ('native.tree_encoder', 'smm.tree'),
    ('native.image_encoder', 'smm.persistence'),
    ('native.classifier', 'head'),
    ('point', 'cmm.point'),
    ('cmm', 'cmm.encoder'),
    ('coupling.position', 'smm.position'),
    ('coupling.scale', 'smm.scale'),
    ('coupling.query', 'smm.query'),
    ('coupling.relation', 'rcm.relation'),
    ('coupling.key', 'rcm.key'),
    ('coupling.value', 'rcm.value'),
    ('coupling.score_geometry', 'rcm.score_geometry'),
    ('coupling.output', 'rcm.output'),
)


def rename(name, reverse=False):
    for old, new in PREFIXES:
        source, target = (new, old) if reverse else (old, new)
        if name == source or name.startswith(source + '.'):
            return target + name[len(source):]
    return name


@dataclass(frozen=True)
class Structure:
    global_feature: torch.Tensor
    branches: torch.Tensor
    topology: torch.Tensor


class SkeletonMorphology(nn.Module):
    def __init__(self, native, coupling, joint):
        super().__init__()
        self.tree = native.tree_encoder
        self.persistence = native.image_encoder
        self.position = coupling.position
        self.scale = coupling.scale
        self.query = coupling.query
        # No random draws: shared initial tensors and training RNG remain exact.
        self.register_parameter('interaction', nn.Parameter(self.query.weight.new_zeros(16, 16)) if joint else None)

    def tree_states(self, batch):
        import dgl
        tree = self.tree
        tree.cell.init_state = True
        original = batch.graph
        g = dgl.graph(original.edges(), num_nodes=original.num_nodes(), device=original.device)
        feats = tree.mlp1(batch.feats)
        n = g.num_nodes()
        g.ndata['iouf'] = tree.cell.W1_iouf(feats)
        for key in ('h1', 'c1', 'h2', 'c2'):
            g.ndata[key] = feats.new_zeros((n, tree.h_size))
        dgl.prop_nodes_topo(g, message_func=tree.cell.message_func,
                           reduce_func=tree.cell.reduce_func, apply_node_func=tree.cell.apply_node_func)
        branch = g.ndata.pop('c2')
        return branch[batch.offset.long()], branch

    def forward(self, x):
        roots, branch = self.tree_states(x['batch'])
        root = (roots * x['mass'][:, None]).sum(0) / x['mass'].sum()
        topology = self.persistence(x['image'][None])[0]
        direct = torch.cat((F.normalize(root, dim=-1), F.normalize(topology, dim=-1)))
        return Structure(direct, branch, topology)

    def regional_query(self, structure, x):
        branch, topology = structure.branches, structure.topology
        g = x['g']
        xyz = g['pos'] - g['pos'].mean(0)
        scale = torch.stack((g['length'].log1p(), torch.full_like(g['length'], g['radius']).log1p(),
                             torch.full_like(g['length'], .5).log1p()), 1)
        code = self.position(torch.sign(xyz) * torch.log1p(xyz.abs() / g['radius'])) + self.scale(scale)
        # Keep the original order: LN(JB), not J LN(B).
        region_branch = add(branch[x['jb']] * x['jw'][:, None], x['jr'], g['nregions'])
        branch_view = F.layer_norm(region_branch, (256,))
        branch_query = F.linear(branch_view, self.query.weight[:, :256], self.query.bias)
        topology_query = F.linear(F.layer_norm(topology, (384,)), self.query.weight[:, 256:640])
        query = branch_query + topology_query
        query = query + F.linear(F.layer_norm(code, (96,)), self.query.weight[:, 640:])
        if self.interaction is not None:
            # Bias-free factors, sharing the original query projections.
            local_query = F.linear(branch_view, self.query.weight[:, :256])
            query = query + F.linear(local_query * topology_query, self.interaction)
        return query


class SurfaceMorphology(nn.Module):
    def __init__(self, point, encoder):
        super().__init__()
        self.point = point
        self.encoder = encoder

    def forward(self, g):
        from tools.cpsf_r38_joint_context.support import SurfaceSupport
        return self.encoder(SurfaceSupport(g, self.point(g['pf']))).surface


class RegionalCoupling(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.relation = original.relation
        self.key = original.key
        self.value = original.value
        self.score_geometry = original.score_geometry
        self.output = original.output

    def forward(self, query, patches, g, conditioned):
        n = g['nregions']
        pr, pj = g['pr'], g['pj']
        key = self.key(patches)
        value = self.value(patches)
        prior = g['pw'] * g['patch_mass'][pj] / g['surface_mass'][pr]
        logits = []
        for lo in range(0, len(pr), 16384):
            rr, pp, phi = pr[lo:lo+16384], pj[lo:lo+16384], g['phi'][lo:lo+16384]
            def score(q, k, geo, rr=rr, pp=pp):
                return (q[rr] * k[pp]).sum(-1) / 4 + self.score_geometry(geo).squeeze(-1)
            s = checkpoint(score, query, key, phi, use_reentrant=False) if torch.is_grad_enabled() else score(query, key, phi)
            logits.append(s)
        logits = torch.cat(logits)
        if conditioned:
            log_weight = prior.log() + logits
            maximum = log_weight.new_full((n,), -torch.inf)
            maximum.scatter_reduce_(0, pr, log_weight.detach(), reduce='amax', include_self=True)
            w = (log_weight - maximum[pr]).exp()
            probability = w / add(w, pr, n)[pr]
        else:
            probability = prior + logits * 0.
        weight = g['surface_mass'] / g['surface_mass'].sum()
        pooled = patches.new_zeros(96)
        for lo in range(0, len(pr), 16384):
            rr, pp = pr[lo:lo+16384], pj[lo:lo+16384]
            phi = g['phi'][lo:lo+16384]
            pi, a = probability[lo:lo+16384], prior[lo:lo+16384]
            def gather(v, prob, base, geo, rr=rr, pp=pp):
                message = prob[:, None] * v[pp] + base[:, None] * self.relation(geo)
                return (message * weight[rr, None]).sum(0)
            pooled = pooled + (checkpoint(gather, value, pi, a, phi, use_reentrant=False)
                               if torch.is_grad_enabled() else gather(value, pi, a, phi))
        return self.output(pooled)


class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        original = Parent('surface_mean' if arm == 'surface_mean' else 'surface_conditioned')
        self._parent_parameter_order = tuple(rename(n) for n, _ in original.named_parameters())
        self._parent_state_order = tuple(original.state_dict())
        self.smm = SkeletonMorphology(original.native, original.coupling, arm == 'smm_joint')
        self.cmm = SurfaceMorphology(original.point, original.cmm)
        self.rcm = RegionalCoupling(original.coupling)
        self.head = original.native.classifier
        self.conditioned = arm != 'surface_mean'
        self.surface_enabled = True
        # Keep native CMM block checkpoints; an outer sample checkpoint retains
        # Utonia pooling tensors across optimizer steps in this runtime.
        self.checkpoint_samples = False

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        pairs = list(super().named_parameters(prefix, recurse, remove_duplicate))
        if recurse:
            stem = prefix + '.' if prefix else ''
            order = {stem + n: i for i, n in enumerate(self._parent_parameter_order)}
            pairs.sort(key=lambda item: order.get(item[0], len(order)))
        yield from pairs

    def parent_state_dict(self):
        state = {rename(n, reverse=True): v for n, v in self.state_dict().items()}
        return OrderedDict((n, state[n]) for n in self._parent_state_order)

    def load_parent_state_dict(self, state):
        mapped = OrderedDict((rename(n), v) for n, v in state.items())
        if self.smm.interaction is not None:
            mapped['smm.interaction'] = self.smm.interaction.detach().clone()
        return self.load_state_dict(mapped, strict=True)

    def encode(self, x):
        structure = self.smm(x)
        if not self.surface_enabled:
            return structure.global_feature
        patches = self.cmm(x['g'])
        query = self.smm.regional_query(structure, x)
        return structure.global_feature + self.rcm(query, patches, x['g'], self.conditioned)

    def forward_group(self, items):
        features = [checkpoint(self.encode, x, use_reentrant=False) if self.training and self.checkpoint_samples
                    else self.encode(x) for x in items]
        return self.head(torch.stack(features))

    def forward(self, x):
        return self.forward_group([x])[0]


def parent_digest(model, prefixes=None):
    h = hashlib.sha256()
    for name, value in model.parent_state_dict().items():
        if prefixes is None or name.startswith(prefixes):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    setup()
    return Model(arm).cuda()
