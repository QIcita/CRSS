"""Reuse C verbatim; change only task output and regional/global readout scope."""
import math
import torch
from torch import nn
from .common import TASKS
from tools.cpsf_h01_progressive_repair.model import Model as H01C


class Model(H01C):
    def __init__(self, task, initial):
        # No trained H01 weights: this recreates its frozen scratch core.
        super().__init__('regional_scaled_pi', 'h01_global', .5)
        self.task = task
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(42)
            self.head = nn.Linear(128, 6 if task == 'flywire_coarse' else 1).cuda()
            if task == 'microns':
                with torch.no_grad():
                    self.head.weight.mul_(.01)
                    self.head.bias.fill_(math.log(initial))

    def regional_features(self, structure, patches, x):
        # C exposes the actual post-processor region states. Apply the same
        # 96->128 readout to each requested region, before any cell pooling.
        _, trace = self.core.rcm(structure, patches, x, trace=True)
        return self.core.rcm.readout(trace['processed'][x['query_index']])

    def forward_group(self, items):
        features = []
        for x, raw in zip(items, self.core.structures(items)):
            structure = self.core.smm.refine(raw, x)
            patches = self.core.cmm(x['g'])
            features.append(self.core.rcm(structure, patches, x))
        return self.head(torch.stack(features))

    def forward(self, x):
        if self.task == 'flywire_coarse':
            return self.forward_group([x])[0]
        raw = next(self.core.structures([x]))
        structure = self.core.smm.refine(raw, x)
        features = self.regional_features(structure, self.core.cmm(x['g']), x)
        return self.head(features).squeeze(-1).clamp(-12, 8).exp()*x['length']


def make(task, initial=0.):
    if task not in TASKS:
        raise ValueError(task)
    return Model(task, initial).cuda()
