"""V18 clean-core skeleton-surface encoder.

One geometry-conditioned bidirectional correspondence operator replaces the
separate ABF-LC and GCHT branches. The implementation preserves the V17
geometry adapters and task-independent input contract, but exposes one clear
message-passing path.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .geometry import GeometryConfig


def add(x, ids, n):
    return x.new_zeros((n, *x.shape[1:])).index_add(0, ids, x)


def mean(x, ids, n):
    mass = add(x.new_ones(len(x)), ids, n).clamp_min(1)
    return add(x, ids, n) / mass.reshape(n, *([1] * (x.ndim - 1)))


def mlp(inp, dim):
    return nn.Sequential(nn.Linear(inp, dim), nn.GELU(), nn.Linear(dim, dim))


def segment_softmax(logits, ids, n):
    if not len(logits):
        return logits
    maximum = logits.new_full((n,), -torch.inf).scatter_reduce(
        0, ids, logits.detach(), 'amax')
    finite = torch.isfinite(maximum)
    safe = torch.where(finite, maximum, torch.zeros_like(maximum))
    exp = torch.exp(logits - safe[ids])
    return exp / add(exp, ids, n)[ids].clamp_min(torch.finfo(exp.dtype).tiny)


@dataclass(frozen=True)
class EncoderConfig:
    dim: int = 64
    rounds: int = 2
    point_chunk: int = 4096
    association: str = 'dynamic'
    surface_context: bool = True
    stage: str = 'clean_core'
    residual_scale: float = 0.1
    geometry: GeometryConfig = GeometryConfig()
    clean_core: bool = True
    temperature: float = 0.15

    def __post_init__(self):
        if min(self.dim, self.rounds, self.point_chunk) < 1:
            raise ValueError('Invalid V18 encoder dimensions')
        if self.association not in ('dynamic', 'fixed', 'once'):
            raise ValueError('Unknown V18 association control')
        if self.temperature <= 0:
            raise ValueError('Temperature must be positive')


@dataclass
class JointEmbedding:
    skeleton: torch.Tensor
    surface: torch.Tensor
    skeleton_xyz: torch.Tensor
    surface_xyz: torch.Tensor
    surface_mass: torch.Tensor


class SharedPointTokenizer(nn.Module):
    """Point-to-patch tokenizer; execution chunks do not change geometry."""
    def __init__(self, dim, chunk):
        super().__init__()
        self.chunk = chunk
        self.stem = mlp(7, dim)
        self.relation = mlp(2 * dim + 4, dim)
        self.norm = nn.LayerNorm(dim)
        self.output = mlp(2 * dim + 1, dim)

    def forward(self, g):
        f, p = g['point_features'], g['point_xyz']
        m = g['n_patches']
        dim = self.norm.normalized_shape[0]
        if not len(f):
            return f.new_zeros((0, dim))
        chunks = []
        for start in range(0, len(f), self.chunk):
            stop = min(start + self.chunk, len(f))

            def encode(start=start, stop=stop):
                ids = g['point_neighbors'][start:stop]
                center = self.stem(f[start:stop])
                other = self.stem(f[ids])
                delta = p[ids] - p[start:stop, None]
                geo = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), -1)
                expanded = center[:, None].expand_as(other)
                rel = self.relation(torch.cat((expanded, other - expanded, geo), -1))
                rel = rel.masked_fill(
                    ~g['neighbor_valid'][start:stop, :, None], -torch.inf
                ).amax(1)
                return self.norm(center + rel)

            token = checkpoint(encode, use_reentrant=False) if torch.is_grad_enabled() else encode()
            chunks.append(token)
        token = torch.cat(chunks)
        owner = g['point_patch']
        total = add(token, owner, m)
        maximum = f.new_full((m, dim), -torch.inf).scatter_reduce(
            0, owner[:, None].expand_as(token), token, 'amax')
        mass = g['patch_point_count'].to(total.dtype)[:, None]
        return self.output(torch.cat((total / mass, maximum, torch.log1p(mass)), -1))


class GeometryCorrespondenceBlock(nn.Module):
    """Single correspondence and bidirectional fusion operator."""
    def __init__(self, dim, residual_scale=0.1, temperature=0.15):
        super().__init__()
        self.dim = dim
        self.residual_scale = residual_scale
        self.temperature = temperature
        self.local_value = mlp(2 * dim + 6, dim)
        self.local_update = mlp(2 * dim, dim)
        self.local_norm = nn.LayerNorm(dim)
        self.query = nn.Sequential(nn.Linear(2 * dim + 4, dim), nn.GELU(), nn.Linear(dim, dim, bias=False))
        self.key = nn.Linear(dim, dim, bias=False)
        self.geometry_bias = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, 1))
        self.null_score = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.value = mlp(dim + 4, dim)
        self.edge_update = mlp(3 * dim + 1, dim)
        self.edge_norm = nn.LayerNorm(dim)
        self.vertex_update = mlp(2 * dim + 1, dim)
        self.vertex_norm = nn.LayerNorm(dim)
        self.gather = nn.Sequential(nn.Linear(3 * dim + 4, dim), nn.GELU(), nn.Linear(dim, 1))
        self.scatter = nn.Sequential(nn.Linear(3 * dim + 4, dim), nn.GELU(), nn.Linear(dim, 1))

    def local_messages(self, target, source, delta, typ):
        geo = torch.cat((delta, delta.norm(dim=-1, keepdim=True),
                         typ[:, None].to(target.dtype), (1 - typ)[:, None].to(target.dtype)), -1)
        return self.local_value(torch.cat((target, source - target, geo), -1))

    def context(self, state, g, use_surface=True):
        r, s = g['local_receiver'], g['local_sender']
        if not use_surface:
            keep = r < g['n_nodes']
            r, s = r[keep], s[keep]
        delta = g['positions'][s] - g['positions'][r]
        value = self.local_messages(state[r], state[s], delta, (r >= g['n_nodes']).long())
        total = add(value, r, len(state))
        count = add(state.new_ones(len(r)), r, len(state))
        message = total / count[:, None].clamp_min(1)
        updated = self.local_norm(state + self.local_update(torch.cat((state, message), -1)))
        return torch.where((count > 0)[:, None], updated, state)

    def correspondence(self, anchor_state, surface_state, edge_state, anchor, surface,
                       geo, n_anchor, n_surface):
        if not len(anchor):
            empty = anchor_state.new_empty(0)
            return empty, anchor_state.new_zeros(n_anchor), empty, empty
        q = self.query(torch.cat((anchor_state[anchor], edge_state[anchor], geo), -1))
        k = self.key(surface_state[surface])
        logits = (nn.functional.normalize(q, dim=-1) * nn.functional.normalize(k, dim=-1)).sum(-1)
        logits = logits / self.temperature + self.geometry_bias(geo).squeeze(-1)
        null_logits = self.null_score(torch.cat((anchor_state, edge_state), -1)).squeeze(-1)
        row_max = logits.new_full((n_anchor,), -torch.inf).scatter_reduce(0, anchor, logits.detach(), 'amax')
        safe = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        row_sum = add(torch.exp(logits - safe[anchor]), anchor, n_anchor).clamp_min(torch.finfo(logits.dtype).tiny)
        row_lse = safe + torch.log(row_sum)
        denom = torch.logaddexp(row_lse, null_logits)
        mass = torch.exp(logits - denom[anchor]).clamp_max(1.)
        null = torch.exp(null_logits - denom).clamp_max(1.)
        # The same logits are normalized in the reverse direction for scatter.
        beta = segment_softmax(logits, surface, n_surface)
        return mass, null, beta, logits

    def fuse(self, anchor_state, surface_state, edge_state, anchor, surface,
             geo, n_anchor, n_surface):
        mass, null, beta, logits = self.correspondence(
            anchor_state, surface_state, edge_state, anchor, surface,
            geo, n_anchor, n_surface)
        if not len(anchor):
            return anchor_state, surface_state, edge_state, null, logits
        denom = add(mass, anchor, n_anchor).clamp_min(1e-8)
        alpha = mass / denom[anchor]
        features = torch.cat((anchor_state[anchor], surface_state[surface], edge_state[anchor], geo), -1)
        values = self.value(torch.cat((surface_state[surface], geo), -1))
        pooled = add(values * alpha[:, None], anchor, n_anchor)
        support = mean(mass, anchor, n_anchor)
        updated_edge = self.edge_norm(edge_state + self.edge_update(
            torch.cat((edge_state, anchor_state, pooled, support[:, None]), -1)))
        message = add(updated_edge[anchor] * beta[:, None], surface, n_surface)
        surface_support = add(mass, surface, n_surface)
        surface_delta = self.vertex_norm(self.vertex_update(
            torch.cat((surface_state, message, surface_support[:, None]), -1)))
        updated_surface = surface_state + self.residual_scale * surface_support[:, None].clamp(0, 1) * surface_delta
        anchor_delta = self.vertex_norm(self.vertex_update(
            torch.cat((anchor_state, updated_edge - anchor_state, support[:, None]), -1)))
        updated_anchor = anchor_state + self.residual_scale * support[:, None].clamp(0, 1) * anchor_delta
        return updated_anchor, updated_surface, updated_edge, null, logits


class UnifiedDynamicHypergraphEncoder(nn.Module):
    """V18 clean core: one geometry-conditioned correspondence path."""
    def __init__(self, config=EncoderConfig()):
        super().__init__()
        self.config = config
        self.skeleton_stem = mlp(7, config.dim)
        self.surface_stem = SharedPointTokenizer(config.dim, config.point_chunk)
        self.segment_stem = mlp(8, config.dim)
        self.block = GeometryCorrespondenceBlock(config.dim, config.residual_scale, config.temperature)
        self.output = nn.LayerNorm(config.dim)
        self.alignment_loss = torch.zeros(())
        self.diagnostics = {}

    def forward(self, g, return_trace=False):
        if g['geometry_config'] != self.config.geometry:
            raise ValueError('Geometry cache and V18 encoder config disagree')
        n, m, q = g['n_nodes'], g['n_patches'], g.get('n_segments', 0)
        skeleton = self.skeleton_stem(g['skeleton_features'])
        surface = self.surface_stem(g)
        edge = skeleton
        segment_enabled = q > 0
        segment = self.segment_stem(g['segment_features']) if segment_enabled else skeleton.new_empty((0, skeleton.shape[-1]))
        if segment_enabled:
            segment = segment + .5 * (skeleton[g['observed_edges'][:, 0]] + skeleton[g['observed_edges'][:, 1]])
        anchor = g['cross_anchor']
        cross_surface = g['cross_surface']
        delta = g['patch_xyz'][cross_surface] - g['skeleton_xyz'][anchor]
        geo = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), -1)
        seg_anchor = g.get('cross_segment', skeleton.new_empty((0,), dtype=torch.long))
        seg_surface = g.get('cross_segment_surface', skeleton.new_empty((0,), dtype=torch.long))
        seg_delta = (g['patch_xyz'][seg_surface] - g['segment_xyz'][seg_anchor]
                     if len(seg_anchor) else skeleton.new_empty((0, 3)))
        seg_geo = torch.cat((seg_delta, seg_delta.norm(dim=-1, keepdim=True)), -1)
        trace = []
        null_values = []
        for _ in range(self.config.rounds):
            state = torch.cat((skeleton, surface))
            state = self.block.context(state, g, self.config.surface_context)
            skeleton, surface = state[:n], state[n:]
            if segment_enabled and len(seg_anchor):
                segment, surface, _, seg_null, _ = self.block.fuse(
                    segment, surface, segment, seg_anchor, seg_surface, seg_geo, q, m)
                null_values.append(seg_null)
                endpoint_values = torch.cat((segment, segment), dim=0)
                endpoint_ids = torch.cat((g['observed_edges'][:, 0], g['observed_edges'][:, 1]))
                endpoint_total = add(endpoint_values, endpoint_ids, n)
                endpoint_count = add(skeleton.new_ones(len(endpoint_ids)), endpoint_ids, n)
                skeleton = torch.where(endpoint_count[:, None] > 0,
                                       skeleton + endpoint_total / endpoint_count[:, None].clamp_min(1), skeleton)
            if len(anchor):
                skeleton, surface, edge, null, logits = self.block.fuse(
                    skeleton, surface, edge, anchor, cross_surface, geo, n, m)
                null_values.append(null)
                if return_trace:
                    trace.append(dict(null=null, logits=logits, skeleton=skeleton,
                                      surface=surface, edge=edge))
        self.diagnostics = {
            'null_mass_mean': float(torch.cat(null_values).detach().mean()) if null_values else 0.,
            'candidate_count': int(len(anchor) + len(seg_anchor)),
            'rounds': self.config.rounds,
            'clean_core': True,
        }
        state = self.output(torch.cat((skeleton, surface)))
        result = JointEmbedding(state[:n], state[n:], g['skeleton_xyz'], g['patch_xyz'], g['patch_point_count'])
        return (result, trace) if return_trace else result

    def forward_batch(self, geometries):
        return [self(g) for g in geometries]
