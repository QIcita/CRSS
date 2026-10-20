"""Pinned SPoTr LPAMLP forward, convolution factories and attention, unmodified.

Only the source-defined symbols are executed, avoiding unrelated CUDA sampling
imports. The explicit adapter supplies the existing ragged patch memberships and
adds original point mass / padding to attention logits; no official code edit.
"""
import ast
import importlib
import sys
import types
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'vendor/SPoTr'
_SYMBOLS = None


def symbols():
    global _SYMBOLS
    if _SYMBOLS is not None:
        return _SYMBOLS
    package = 'official_r28_spotr_layers'
    if package not in sys.modules:
        mod = types.ModuleType(package)
        mod.__path__ = [str(VENDOR / 'openpoints/models/layers')]
        sys.modules[package] = mod
    conv = importlib.import_module(package + '.conv')
    act = importlib.import_module(package + '.activation')
    file = VENDOR / 'openpoints/models/backbone/spotr.py'
    tree = ast.parse(file.read_text())
    kept = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name in ('new_attention', 'get_aggregation_features', 'LPAMLP')]
    namespace = dict(torch=torch, nn=nn,
                     create_convblock2d=conv.create_convblock2d,
                     create_convblock1d=conv.create_convblock1d,
                     create_act=act.create_act)
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(file), 'exec'), namespace)
    _SYMBOLS = namespace
    return namespace


class OriginalPatchGrouper(nn.Module):
    """Bipartite grouping adapter; all original points, no FPS/KNN/sampling."""
    def forward(self, query, support, features):
        return self.relative_xyz, self.raw_features


def build(context):
    original = symbols()
    namespace = dict(original)
    # LPAMLP only needs this map entry for our declared feature_type.
    namespace['CHANNEL_MAP'] = {'dp_fj': lambda channels: channels + 3}
    namespace['create_grouper'] = lambda config: OriginalPatchGrouper()
    original_attention = original['new_attention']

    def mass_attention(values, y_j=None, attn=None, aux_attn=None, tau=1):
        assert values.ndim == 4 and y_j is None and aux_attn is None and tau == 1
        score = attn if context['kind'] == 'r9_lpa_attention' else torch.zeros_like(attn)
        # Original softmax per channel, over true members of this patch only.
        score = score + context['log_mass']
        return original_attention(values, attn=score, tau=tau)

    namespace['new_attention'] = mass_attention
    file = VENDOR / 'openpoints/models/backbone/spotr.py'
    tree = ast.parse(file.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LPAMLP')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(file), 'exec'), namespace)
    module = namespace['LPAMLP'](
        in_channels=96, norm_args=None, act_args={'act': 'relu'},
        aggr_args={'feature_type': 'dp_fj', 'reduction': 'sum'},
        group_args={}, conv_args={}, expansion=1, use_res=True,
        num_posconvs=1, less_act=False)
    # Official alpha is declared but never read by LPAMLP.forward.
    module.alpha.requires_grad_(False)
    return module
