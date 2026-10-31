import hashlib
import sys
import types
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import FROZEN, backends, setup


def digest(model, prefixes=None):
    h = hashlib.sha256()
    for n, v in model.state_dict().items():
        if prefixes is None or n.startswith(prefixes):
            h.update(n.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class OfficialHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        from tools.cpsf_graphformer.models import load_tree_module
        sys.path[:0] = [str(FROZEN/'vendor/upstream'), str(FROZEN/'vendor/dinov2')]
        from graphformer.models.image_encoder import DINOv2ImageEncoder
        from graphformer.models.finetune_model import FineTuneModel
        from dinov2.hub.backbones import dinov2_vits14
        def scratch(repo, variant, *a, **kw):
            assert repo=='facebookresearch/dinov2' and variant=='dinov2_vits14' and not a and not kw
            return dinov2_vits14(pretrained=False)
        def forbidden(*a, **kw):
            raise AssertionError('External weights forbidden')
        with patch('torch.hub.load',side_effect=scratch), patch('torch.hub.load_state_dict_from_url',side_effect=forbidden):
            image = DINOv2ImageEncoder(freeze_backbone=False,model_variant='dinov2_vits14')
        tree = load_tree_module().TreeLSTMDouble(x_size=5,h_size=256,num_classes=0,fc=False,bn=False,mode='sum')
        shell=nn.Module(); shell.tree_encoder=tree; shell.image_encoder=image; shell.tree_encoder_type='double'
        self.native=FineTuneModel(shell,6,mode='multimodal',fusion_mode='concat',use_projection=False,freeze_encoders=False)
        self.external_pretrained=False
        head=self.native.classifier
        assert isinstance(head,nn.Sequential) and isinstance(head[2],nn.BatchNorm1d)
        assert head[0].p==.5 and head[4].p==.35

    def encode(self,x):
        h=self.native.encode_tree(x['batch'])
        assert h.shape[0]==1
        # The original one-component mass readout is preserved exactly.
        h=(h*x['mass'][:,None]).sum(0)/x['mass'].sum()
        image=self.native.encode_image(x['image'][None])[0]
        return torch.cat([F.normalize(h,dim=-1),F.normalize(image,dim=-1)])

    def forward_group(self,items):
        assert not self.training or len(items)>=2, 'No singleton BatchNorm training'
        features=torch.stack([self.encode(x) for x in items])
        return self.native.classifier(features)

    def forward(self,x):
        assert not self.training, 'Training uses effective-batch head'
        return self.forward_group([x])[0]


def make(arm):
    setup()
    if arm=='official_pi_only':
        from tools.cpsf_graphformer.models import Model
        return Model('flywire_coarse').cuda()
    assert arm=='official_features_head'
    return OfficialHeadModel().cuda()


def load_sample(row,stats,arm):
    import dgl
    with np.load(row['cache_path']) as x:
        f=x['features' if arm=='official_pi_only' else 'official_features']
        f=(f-np.asarray(stats['mean']))/np.asarray(stats['std'])
        e=x['edges']; roots=x['roots'].copy(); mass=x['component_mass'].copy(); pi=x['image'].copy()
    graph=dgl.graph((e[:,0],e[:,1]),num_nodes=len(f),device='cuda')
    batch=types.SimpleNamespace(graph=graph,feats=torch.as_tensor(f,dtype=torch.float32,device='cuda'),offset=torch.as_tensor(roots,device='cuda'))
    return dict(batch=batch,mass=torch.as_tensor(mass,device='cuda'),query_component=-1,image=torch.as_tensor(pi,device='cuda'))


def backward_group(model,group,stats,arm):
    """Same sample-mean CE objective, with native BN at effective batch size."""
    if arm=='official_pi_only':
        value=0.
        for row in group:
            x=load_sample(row,stats,arm); z=model(x)
            loss=F.cross_entropy(z[None],torch.tensor([int(row['target'])],device='cuda'),label_smoothing=.05)
            assert torch.isfinite(loss)
            (loss/len(group)).backward(); value+=float(loss.detach())
            del x,z,loss
        return value
    items=[load_sample(row,stats,arm) for row in group]
    z=model.forward_group(items)
    target=torch.tensor([int(r['target']) for r in group],device='cuda')
    loss=F.cross_entropy(z,target,label_smoothing=.05)
    assert torch.isfinite(loss)
    loss.backward()
    return float(loss.detach())*len(group)
