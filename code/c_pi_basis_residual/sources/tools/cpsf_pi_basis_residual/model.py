"""Separate the shared PI density without dropping weighted-channel information."""
import torch
from torch import nn
from torch.nn import functional as F
from tools.cpsf_c_transfer import common
from tools.cpsf_c_transfer.model import make as make_transfer
from tools.cpsf_h01_progressive_repair.model import make as make_h01


def factor_channels(images):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError('PI input must have shape [batch, 3, height, width]')
    base = images[:, :1]
    weighted = images[:, 1:]
    energy = base.square().sum((-2, -1), keepdim=True)
    # Original FlyWire has six empty PIs. Their coefficients remain zero.
    denominator = torch.where(energy > 0, energy, torch.ones_like(energy))
    coefficients = (weighted*base).sum((-2, -1), keepdim=True)/denominator
    residual = weighted - coefficients*base
    return torch.cat((base, residual), dim=1), coefficients.flatten(1)


class BasisResidualPersistence(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.coefficient_projection = nn.Linear(2, 384, bias=False)
        # Let the retained coefficients enter through learning, without an
        # arbitrary common offset dominating the initial image statistics.
        nn.init.zeros_(self.coefficient_projection.weight)

    def forward(self, images):
        factors, coefficients = factor_channels(images)
        statistics = F.adaptive_avg_pool2d(factors, (8, 16)).flatten(1)
        base = F.layer_norm(statistics+self.coefficient_projection(coefficients), (384,))
        update = self.encoder(factors)
        return base+.1*update/torch.sqrt(1+update.square().mean(-1, keepdim=True))


def make(task, initial):
    if task not in ('h01_global', 'flywire_coarse', 'microns'):
        raise ValueError(task)
    model = make_h01('regional_scaled_pi', task, initial) if task == 'h01_global' else make_transfer(task, initial)
    original = model.core.smm.persistence
    device = next(original.parameters()).device
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(42)
        replacement = BasisResidualPersistence(original.encoder).to(device)
    model.core.smm.persistence = replacement
    return model
