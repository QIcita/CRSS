"""GeoTransformer MultiHeadAttention with two explicit score-prior adaptations.

Original class/forward retained upstream. QKV projections, scale, masking,
softmax, dropout and value matmul are the official implementation.
attention_factors now means additive geometry logits; key_weights is positive
mass used as log prior, not multiplicative score attenuation. No pretrained.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from einops import rearrange
import importlib
import sys
import types
ROOT=Path(__file__).resolve().parents[2]
# Import only the official vanilla attention and its real layer dependencies.
# The upstream transformer __init__ also imports unused RPE/debug IPython code.
_upstream=(ROOT/'vendor/GeoTransformer/geotransformer').resolve()
for _name,_path in [('geotransformer',_upstream),('geotransformer.modules',_upstream/'modules'),
                    ('geotransformer.modules.transformer',_upstream/'modules/transformer')]:
    if _name not in sys.modules:
        _mod=types.ModuleType(_name);_mod.__path__=[str(_path)];_mod.__package__=_name;sys.modules[_name]=_mod
    else:
        assert str(_path) in list(getattr(sys.modules[_name],'__path__',[])), 'different official source loaded'
from geotransformer.modules.transformer.vanilla_transformer import MultiHeadAttention as OfficialAttention
class MassAttention(OfficialAttention):
    def forward(
        self, input_q, input_k, input_v, key_weights=None, key_masks=None, attention_factors=None, attention_masks=None
    ):
        """Vanilla Self-attention forward propagation.

        Args:
            input_q (Tensor): input tensor for query (B, N, C)
            input_k (Tensor): input tensor for key (B, M, C)
            input_v (Tensor): input tensor for value (B, M, C)
            key_weights (Tensor): soft masks for the keys (B, M)
            key_masks (BoolTensor): True if ignored, False if preserved (B, M)
            attention_factors (Tensor): factors for attention matrix (B, N, M)
            attention_masks (BoolTensor): True if ignored, False if preserved (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: intermediate values
                'attention_scores': torch.Tensor (B, H, N, M), attention scores before dropout
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_scores + attention_factors.unsqueeze(1)
        if key_weights is not None:
            attention_scores = attention_scores + key_weights.clamp_min(1e-12).log().unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        if attention_masks is not None:
            attention_scores = attention_scores.masked_fill(attention_masks, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores
