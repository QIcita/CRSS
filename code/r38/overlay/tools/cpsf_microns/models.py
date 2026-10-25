"""Frozen real R38/SGFormer/Utonia encoders with a shared count-rate decoder."""
from types import SimpleNamespace
import torch
from torch import nn
from .common import backends


class CountModel(nn.Module):
    def __init__(self, name, initial_rate):
        super().__init__()
        backends()
        self.name = name
        if name == 'r38_dense':
            from tools.cpsf_r34_fps_rebench.model import H01Model
            from tools.cpsf_r38_dense.model import install
            self.encoder = install(H01Model('h01_global'))
            # The global task head/readout are replaced by a regional decoder.
            del self.encoder.head
            del self.encoder.readout
            dim = 96
        elif name == 'sgformer':
            from tools.cpsf_external_phase3.models import SGFormerTask
            self.encoder = SGFormerTask('h01_global')
            del self.encoder.head
            dim = 64
        elif name == 'utonia_scratch_r2':
            from tools.cpsf_utonia_scratch_r2.models import UtoniaTask
            self.encoder = UtoniaTask('h01_global')
            del self.encoder.head
            dim = sum(self.encoder.architecture_config['enc_channels'])
        else:
            raise ValueError(name)
        self.head = nn.Sequential(nn.Linear(dim, 96), nn.LayerNorm(96), nn.GELU(), nn.Linear(96, 1))
        with torch.no_grad():
            self.head[-1].weight.mul_(.01)
            self.head[-1].bias.fill_(float(__import__('math').log(initial_rate)))

    def forward(self, item):
        if self.name == 'r38_dense':
            feature = self.encoder.streams(item['r38'])[0][0][item['query_index']]
        elif self.name == 'sgformer':
            from tools.cpsf_v21_segment_units.model import add
            g = item['graph']
            node = self.encoder.backbone(SimpleNamespace(graph={'node_feat': g['feat'], 'edge_index': g['edge_index']}))
            # Task-defined regional query over graph-node features.
            q = item['graph_query']
            feature = add(node[q['node']] * q['weight'][:, None], q['region'], len(item['length']))
            feature = feature / q['mass'].clamp_min(1e-12)[:, None]
        else:
            g = item['point']
            point = self.encoder.backbone(dict(coord=g['coord'], feat=g['feat'], grid_coord=g['grid_coord'],
                      offset=torch.tensor([len(g['coord'])], device=g['coord'].device, dtype=torch.long)))
            while 'pooling_parent' in point.keys():
                parent = point.pop('pooling_parent')
                inverse = point.pop('pooling_inverse')
                parent.feat = torch.cat((parent.feat, point.feat[inverse]), dim=-1)
                point = parent
            feature = (point.feat[g['query_neighbors']] * g['query_weights'][:, :, None]).sum(1)
        log_rate = self.head(feature).squeeze(-1).clamp(-12, 8)
        return log_rate.exp() * item['length']


def read_item(row, device, name):
    from .common import sha
    import numpy as np
    assert sha(row['input_path']) == row['input_sha256']
    value = torch.load(row['input_path'], map_location='cpu', weights_only=False)
    query = value['query_index']
    if name == 'sgformer':
        # Only graph plus the shared target query; no surface-derived arrays.
        g = value['r38']
        remap = torch.full((g['nregions'],), -1, dtype=torch.long)
        remap[query] = torch.arange(len(query))
        keep = remap[g['sr']] >= 0
        item = dict(graph=value['graph'], length=value['length'],
                    graph_query=dict(node=g['sn'][keep], region=remap[g['sr'][keep]],
                                     weight=g['sw'][keep], mass=g['graph_mass'][query]))
    elif name == 'utonia_scratch_r2':
        # The encoder sees no skeleton nodes/edges. Spatial queries are shared
        # task information, and carry neither connectivity nor path features.
        item = dict(point=value['point'], length=value['length'])
    else:
        item = dict(r38=value['r38'], query_index=query, length=value['length'])
    def move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        return x
    assert sha(row['label_path']) == row['label_sha256']
    with np.load(row['label_path']) as a:
        assert np.array_equal(a['query_index'], query.numpy())
        target = torch.tensor(a['target'], dtype=torch.float32, device=device)
    del value
    return move(item), target
