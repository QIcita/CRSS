"""Batch disjoint trees only: no new parameters or cross-cell messages."""
import types
import torch
from torch.nn import functional as F
from .common import EXPERIMENT
from tools.cpsf_smm_joint.models import Model, Structure, make as original_make


class RuntimeModel(Model):
    def structures(self, items):
        import dgl
        counts = [x['batch'].graph.num_nodes() for x in items]
        root_counts = [len(x['mass']) for x in items]
        offsets = [0]
        for n in counts[:-1]:
            offsets.append(offsets[-1]+n)
        batch = types.SimpleNamespace(
            graph=dgl.batch([x['batch'].graph for x in items]),
            feats=torch.cat([x['batch'].feats for x in items]),
            offset=torch.cat([x['batch'].offset+offset for x,offset in zip(items, offsets)]))
        roots, branches = self.smm.tree_states(batch)
        roots = roots.split(root_counts)
        branches = branches.split(counts)
        for x, component_roots, branch in zip(items, roots, branches):
            root = (component_roots*x['mass'][:,None]).sum(0)/x['mass'].sum()
            # Keep the same per-cell PI execution and RNG consumption.
            topology = self.smm.persistence(x['image'][None])[0]
            direct = torch.cat((F.normalize(root,dim=-1), F.normalize(topology,dim=-1)))
            yield Structure(direct, branch, topology)

    def forward_group(self, items):
        structures = self.structures(items)
        features = []
        for x, structure in zip(items, structures):
            patches = self.cmm(x['g'])
            query = self.smm.regional_query(structure, x)
            features.append(structure.global_feature+self.rcm(query, patches, x['g'], self.conditioned))
        return self.head(torch.stack(features))


def make(arm):
    model = original_make(arm)
    model.__class__ = RuntimeModel
    return model
