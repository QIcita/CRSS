"""R39 MICrONS regional count decoder; model definition only.

Uses the same prepared region input as the published R38 model dependencies.
The unchanged ``r38`` item key denotes the shared geometry input format.
"""
import torch
from torch import nn
from .model import Model


class CountModel(nn.Module):
    def __init__(self, name, initial_rate):
        super().__init__()
        assert name == 'r39'
        self.name = name
        self.encoder = Model('h01_global')
        del self.encoder.head
        del self.encoder.readout
        self.head = nn.Sequential(nn.Linear(96, 96), nn.LayerNorm(96), nn.GELU(), nn.Linear(96, 1))
        with torch.no_grad():
            self.head[-1].weight.mul_(.01)
            self.head[-1].bias.fill_(float(__import__('math').log(initial_rate)))

    def forward(self, item):
        feature = self.encoder.streams(item['r38'])[0][0][item['query_index']]
        log_rate = self.head(feature).squeeze(-1).clamp(-12, 8)
        return log_rate.exp() * item['length']
