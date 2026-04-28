import torch
import diffusers.models.attention

from dwm.models.rayrope.rayrope_mha import MultiheadAttention
from dwm.models.rayrope.rayrope_cross_attention_rowwise import (
    RayRoPE_DotProductAttention_Cross_Rowwise,
)


class VTRayRoPERowwiseAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        time_mix_inner_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm=None,
        rayrope_config=None,
    ):
        super().__init__()

        self.norm_in = torch.nn.LayerNorm(dim)
        self.ff_in = diffusers.models.attention.FeedForward(
            dim,
            dim_out=time_mix_inner_dim,
            activation_fn="geglu",
        )

        self.norm1 = torch.nn.LayerNorm(time_mix_inner_dim)
        self.norm3 = torch.nn.LayerNorm(time_mix_inner_dim)

        self.ff = diffusers.models.attention.FeedForward(
            time_mix_inner_dim,
            activation_fn="geglu",
        )

        self.attention_head_dim = attention_head_dim
        self.rayrope_config = rayrope_config or {}

        self.attn1 = MultiheadAttention(
            embed_dim=time_mix_inner_dim,
            num_heads=num_attention_heads,
            qk_norm=qk_norm is not None,
            predict_d="predict_dsig",
            cross_attn=True,
        )

        self.rayrope_attn = None
        self.cached_hw = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        w2cs: torch.Tensor,
        Ks: torch.Tensor,
        row_indices: torch.Tensor,
        full_height: int,
        width: int,
        self_attention_mask: torch.Tensor = None,
    ):
        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.ff_in(hidden_states)
        hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)

        if self.rayrope_attn is None or self.cached_hw != (full_height, width):
            self.rayrope_attn = RayRoPE_DotProductAttention_Cross_Rowwise(
                head_dim=self.attention_head_dim,
                patches_x=width,
                patches_y_total=full_height,
                image_width=width,
                image_height=full_height,
                **self.rayrope_config,
            ).to(device=hidden_states.device, dtype=hidden_states.dtype)

            self.attn1.sdpa_fn = self.rayrope_attn.forward
            self.cached_hw = (full_height, width)

        self.rayrope_attn._precompute_and_cache_apply_fns(
            w2cs=w2cs,
            Ks=Ks,
            w2cs_kv=w2cs,
            Ks_kv=Ks,
            row_indices=row_indices,
        )

        attn_output = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            attn_mask=self_attention_mask,
        )

        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states