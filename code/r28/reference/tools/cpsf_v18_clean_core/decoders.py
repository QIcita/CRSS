"""Task-only readouts. Receive embeddings and query indices, never raw points."""
import torch
from torch import nn
from .encoder import mlp
import numpy as np
from scipy.spatial import cKDTree


def surface_pool(z):
    if not len(z.surface):
        return z.skeleton.new_zeros(z.skeleton.shape[-1])
    weight = z.surface_mass.to(z.surface.dtype)
    return (z.surface*weight[:, None]).sum(0)/weight.sum()


def _topological_branches(edge_index, n_nodes):
    """Return edge-disjoint paths between endpoints and junctions.

    The operation is label-free and depends only on the observed skeleton
    graph.  A branch starts at a node whose degree is not two and continues
    through degree-two nodes until the next such node.  Isolated nodes and
    residual cycle edges are retained as singleton branches.
    """
    if isinstance(edge_index, torch.Tensor): edge_index=edge_index.detach().cpu().numpy()
    edges=np.asarray(edge_index, dtype=np.int64).reshape(-1,2)
    adjacency=[[] for _ in range(int(n_nodes))]
    for ei,(u,v) in enumerate(edges):
        adjacency[int(u)].append((int(v),ei)); adjacency[int(v)].append((int(u),ei))
    degree=np.asarray([len(x) for x in adjacency], dtype=np.int64)
    terminals=np.flatnonzero(degree!=2)
    used=np.zeros(len(edges), dtype=bool); branches=[]
    for start in terminals.tolist():
        for nxt,ei in adjacency[start]:
            if used[ei]: continue
            nodes=[start]; edge_ids=[]; prev=start; cur=nxt; edge=ei
            while True:
                used[edge]=True; edge_ids.append(edge); nodes.append(cur)
                if degree[cur]!=2: break
                options=[x for x in adjacency[cur] if x[0]!=prev]
                if not options: break
                prev,cur=cur,options[0][0]
                edge=options[0][1]
            branches.append((np.asarray(nodes,dtype=np.int64),np.asarray(edge_ids,dtype=np.int64)))
    # A pure cycle has no terminal.  Keep each remaining connected edge as a
    # small branch so pooling is still defined rather than silently dropping
    # a component.
    for ei,(u,v) in enumerate(edges):
        if not used[ei]: branches.append((np.asarray([u,v],dtype=np.int64),np.asarray([ei],dtype=np.int64)));used[ei]=True
    for i in np.flatnonzero(degree==0): branches.append((np.asarray([i],dtype=np.int64),np.empty(0,dtype=np.int64)))
    return branches


@torch.no_grad()
def branch_aware_pool(z, edge_index):
    """Hierarchical branch-conditioned pooling for frozen cell readout.

    Surface patches are assigned to the nearest observed segment midpoint in
    bounded chunks.  Each topological branch first forms its own skeleton and
    surface token; the final cell tokens are a length/support-weighted
    aggregation over branches.  The return shape is identical to
    ``(skeleton_pool, surface_pool)`` and no encoder state is changed.
    """
    d=z.surface.shape[-1] if len(z.surface) else z.skeleton.shape[-1]
    if not len(z.skeleton): return z.surface.new_zeros(d),surface_pool(z)
    branches=_topological_branches(edge_index,len(z.skeleton))
    if not branches: return z.skeleton.mean(0),surface_pool(z)
    xyz=z.skeleton_xyz.detach().cpu().numpy()
    if isinstance(edge_index, torch.Tensor): edge_index=edge_index.detach().cpu().numpy()
    edges=np.asarray(edge_index,dtype=np.int64).reshape(-1,2)
    mids=(xyz[edges[:,0]]+xyz[edges[:,1]])/2 if len(edges) else np.empty((0,3))
    branch_of_edge=np.full(len(edges),-1,np.int64)
    lengths=np.zeros(len(branches),np.float64)
    for bi,(nodes,eids) in enumerate(branches):
        branch_of_edge[eids]=bi
        if len(eids): lengths[bi]=np.linalg.norm(xyz[edges[eids,0]]-xyz[edges[eids,1]],axis=1).sum()
        else: lengths[bi]=1.
    mass=z.surface_mass.detach().cpu().numpy().astype(np.float64) if len(z.surface) else np.empty(0)
    assignment=np.full(len(z.surface),-1,np.int64)
    if len(z.surface) and len(mids):
        tree=cKDTree(mids);points=z.surface_xyz.detach().cpu().numpy()
        for start in range(0,len(points),65536):
            _,idx=tree.query(points[start:start+65536],k=1,workers=1)
            assignment[start:start+len(idx)]=branch_of_edge[np.asarray(idx,dtype=np.int64)]
    sk_tokens=[]; surf_tokens=[]; branch_mass=[]
    for bi,(nodes,_) in enumerate(branches):
        sk_tokens.append(z.skeleton[torch.as_tensor(nodes,device=z.skeleton.device)].mean(0))
        mask=torch.as_tensor(assignment==bi,device=z.surface.device)
        if mask.any():
            w=z.surface_mass[mask].to(z.surface.dtype);surf_tokens.append((z.surface[mask]*w[:,None]).sum(0)/w.sum());branch_mass.append(float(w.sum()))
        else:
            surf_tokens.append(z.surface.new_zeros(d));branch_mass.append(0.)
    sk=torch.stack(sk_tokens);sf=torch.stack(surf_tokens)
    branch_mass=np.asarray(branch_mass);mean_mass=max(float(branch_mass[branch_mass>0].mean()) if (branch_mass>0).any() else 0.,1e-12)
    weights=torch.as_tensor(np.sqrt(np.maximum(lengths,1e-12))*np.sqrt(1+branch_mass/mean_mass),device=sk.device,dtype=sk.dtype)
    sk_pool=(sk*weights[:,None]).sum(0)/weights.sum().clamp_min(1e-12)
    active=branch_mass>0
    if active.any():
        sf_pool=(sf*weights[:,None]).sum(0)/weights.sum().clamp_min(1e-12)
    else: sf_pool=surface_pool(z)
    return sk_pool,sf_pool


class LocalCompartmentDecoder(nn.Module):
    def __init__(self, dim=64, classes=3):
        super().__init__(); self.head = nn.Sequential(mlp(dim, dim), nn.Linear(dim, classes))

    def forward(self, z, node_indices):
        return self.head(z.skeleton[node_indices])


class PointLocalCompartmentDecoder(LocalCompartmentDecoder):
    """Query XYZ locates the readout only; no skeleton features or graph.

    Radius 2 is micrometers under the fixed physical point-only adapter.
    Nearest-patch fallback is explicit when the sphere contains no patch.
    """
    def forward(self,z,query_xyz):
        if not len(z.surface):raise ValueError('Point-only local readout needs real surface points')
        distance=(query_xyz[:,None]-z.surface_xyz[None]).square().sum(-1)
        weight=torch.exp(-distance)*(distance<=4)*z.surface_mass[None]
        missing=weight.sum(-1)==0
        fallback=torch.zeros_like(weight).scatter(1,distance.argmin(-1,keepdim=True),1.)
        weight=torch.where(missing[:,None],fallback,weight)
        feature=(weight/weight.sum(-1,keepdim=True))@z.surface
        return self.head(feature)


class TEDDecoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__(); self.head = nn.Sequential(mlp(3*dim, dim), nn.Linear(dim, 2))

    def forward(self, z, root_indices):
        root = z.skeleton[root_indices].reshape(-1, z.skeleton.shape[-1])
        return self.head(torch.cat((root, z.skeleton.mean(0).expand_as(root), surface_pool(z).expand_as(root)), -1))


class CellCompositionDecoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__(); self.head = nn.Sequential(mlp(2*dim, dim), nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, z):
        return self.head(torch.cat((z.skeleton.mean(0), surface_pool(z)), -1))


class TLPDecoder(nn.Module):
    def __init__(self, dim=64, support_radius=2., pair_chunk=256):
        super().__init__(); self.radius = support_radius; self.chunk = pair_chunk
        self.head = nn.Sequential(mlp(3*dim, dim), nn.Linear(dim, 1))

    def forward(self, z, pairs):
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
            raise ValueError('pairs must be int64 [Q,2]')
        if pairs.numel() and ((pairs < 0).any() or (pairs >= len(z.skeleton)).any()):
            raise ValueError('Pair query outside sample')
        scores = []
        for pair in pairs.split(self.chunk):
            u, v = pair.unbind(1); a, b = z.skeleton_xyz[u], z.skeleton_xyz[v]
            direction = b-a
            rel = z.surface_xyz[None]-a[:, None]
            t = (rel*direction[:, None]).sum(-1)/direction.square().sum(-1).clamp_min(1e-12)[:, None]
            closest = a[:, None]+t.clamp(0, 1)[..., None]*direction[:, None]
            distance = (z.surface_xyz[None]-closest).norm(dim=-1)
            weight = torch.exp(-distance.square())*(distance <= self.radius)*z.surface_mass[None]
            surface = weight.to(z.surface.dtype)@z.surface/weight.sum(-1, keepdim=True).clamp_min(1e-12)
            # Symmetric in endpoint order; coordinates only select embeddings.
            pair_feature = torch.cat((z.skeleton[u]+z.skeleton[v],
                                      (z.skeleton[u]-z.skeleton[v]).abs(), surface), -1)
            scores.append(self.head(pair_feature).squeeze(-1))
        return torch.cat(scores) if scores else z.skeleton.new_empty(0)
