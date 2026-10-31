"""Regional task readout, dimension-correct residuals and compact PI encoding."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import ARMS,TASKS
from tools.cpsf_smm_transfer.model import Model as Parent
from tools.cpsf_progressive_smm.model import PairResidual,CrossResidual,normalize
from tools.cpsf_r34_fps_rebench.model import H01Model
from .regional import RegionalCoupling


class RelativePair(PairResidual):
    @staticmethod
    def project(product,weight,reference):
        bounded=weight/torch.sqrt(1+weight.square().sum())
        scale=reference.detach().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
        # Compensate RMS versus vector norm: relative L2 change is bounded
        # by 0.1, rather than by 0.1 / sqrt(feature width).
        return .1*math.sqrt(reference.shape[-1])*scale*F.linear(product,bounded)


class RelativeCross(CrossResidual):
    def forward(self,local,global_context):
        product=torch.tanh(self.local(normalize(local)))*torch.tanh(self.global_context(normalize(global_context)))
        return RelativePair.project(product/math.sqrt(self.rank),self.output,local)


class CompactPersistence(nn.Module):
    """384D coarse PI statistics with a bounded learned convolutional residual."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(3,24,5,stride=2,padding=2),nn.GroupNorm(6,24),nn.GELU(),
            nn.Conv2d(24,48,3,stride=2,padding=1),nn.GroupNorm(8,48),nn.GELU(),
            nn.Conv2d(48,96,3,stride=2,padding=1),nn.GroupNorm(8,96),nn.GELU(),
            nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(1))

    def forward(self,images):
        base=F.layer_norm(F.adaptive_avg_pool2d(images,(8,16)).flatten(1),(384,))
        update=self.encoder(images)
        return base+.1*update/torch.sqrt(1+update.square().mean(-1,keepdim=True))


class Model(Parent):
    def __init__(self,arm,task,initial):
        super().__init__('progressive_gated',task,initial)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(42)
            original=H01Model(task).cuda()
        self.core.rcm=RegionalCoupling(self.core.rcm,original.processor,original.readout,task)
        self.head=original.head
        if task=='h01_global':
            nn.init.constant_(self.head.bias,math.log(initial/(1-initial)))
        if arm!='regional':
            self.core.smm.interaction.__class__=RelativePair
            self.core.rcm.local_pair.__class__=RelativePair
            self.core.rcm.global_pair.__class__=RelativePair
            self.core.rcm.cross_skeleton.__class__=RelativeCross
            self.core.rcm.cross_surface.__class__=RelativeCross
        if arm=='regional_scaled_pi':
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(42)
                self.core.smm.persistence=CompactPersistence().cuda()
        self.arm=arm

    def features(self,x):
        raw=next(self.core.structures([x]))
        structure=self.core.smm.refine(raw,x)
        return self.core.rcm(structure,self.core.cmm(x['g']),x)


def make(arm,task,initial):
    if arm not in ARMS or task not in TASKS:raise ValueError((arm,task))
    return Model(arm,task,initial).cuda()
