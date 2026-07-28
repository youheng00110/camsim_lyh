from typing import Optional

import diffusers
import einops
import torch

from dwm.models.crossview_temporal_dit_rayrope import (
    DiTCrossviewTemporalConditionModel as RayRoPEModelBase,
)
from dwm.models.urope.block import VTURoPEAttentionBlock


class DiTCrossviewTemporalConditionModel(RayRoPEModelBase):
    """OpenDWM DiT with full cross-view URoPE self-attention."""

    @diffusers.configuration_utils.register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        projection_class_embeddings_input_dim: int = None,
        condition_image_adapter_config: Optional[dict] = None,
        enable_crossview: bool = False,
        enable_temporal: bool = False,
        urope_config: Optional[dict] = None,
        crossview_attention_type: str = "full",
        temporal_attention_type: str = None,
        merge_factor: float = 2,
        merge_strategy: str = "learned_with_images",
        crossview_block_layers: Optional[dict] = None,
        temporal_block_layers: Optional[dict] = None,
        crossview_gradient_checkpointing: bool = False,
        temporal_gradient_checkpointing: bool = False,
        mixer_type: str = "AlphaBlender",
        perspective_modeling_type: str = "urope",
        disable_view_emb_on_temporal_module: bool = False,
        qk_norm_on_additional_modules=None,
        mask_module=None,
        **kwargs,
    ):
        if perspective_modeling_type != "urope":
            raise ValueError(
                "This model requires perspective_modeling_type='urope'."
            )
        if crossview_attention_type != "full":
            raise ValueError(
                "The first URoPE reproduction supports full attention only."
            )

        self.urope_config = {
            "min_depth": 2.0,
            "max_depth": 20.0,
            "freq_base": 100.0,
            "freq_scale": 1.0,
            "group_size": 4,
            "leaveout_head": 0,
            "camera_convention": "opencv",
            **(urope_config or {}),
        }

        super().__init__(
            patch_size=patch_size,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            projection_class_embeddings_input_dim=(
                projection_class_embeddings_input_dim
            ),
            condition_image_adapter_config=condition_image_adapter_config,
            enable_crossview=enable_crossview,
            enable_temporal=enable_temporal,
            rayrope_config=None,
            crossview_attention_type=crossview_attention_type,
            temporal_attention_type=temporal_attention_type,
            merge_factor=merge_factor,
            merge_strategy=merge_strategy,
            crossview_block_layers=crossview_block_layers,
            temporal_block_layers=temporal_block_layers,
            crossview_gradient_checkpointing=(
                crossview_gradient_checkpointing
            ),
            temporal_gradient_checkpointing=temporal_gradient_checkpointing,
            mixer_type=mixer_type,
            perspective_modeling_type=perspective_modeling_type,
            disable_view_emb_on_temporal_module=(
                disable_view_emb_on_temporal_module
            ),
            qk_norm_on_additional_modules=qk_norm_on_additional_modules,
            mask_module=mask_module,
            **kwargs,
        )

        if enable_crossview:
            inner_dim = attention_head_dim * num_attention_heads
            self.crossview_transformer_blocks = torch.nn.ModuleList([
                VTURoPEAttentionBlock(
                    dim=inner_dim,
                    time_mix_inner_dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    qk_norm=qk_norm_on_additional_modules,
                    urope_config=self.urope_config,
                )
                for _ in range(len(crossview_block_layers))
            ])

    def forward_crossview_block_and_mix_result(
        self,
        crossview_block,
        mixer,
        hidden_states,
        view_emb,
        batch_size,
        sequence_length,
        view_count,
        width,
        height,
        disable_crossview,
        crossview_attention_mask,
        crossview_attention_index,
        camera_intrinsics_norm=None,
        camera2referego=None,
    ):
        del crossview_attention_index

        if camera_intrinsics_norm is None or camera2referego is None:
            raise ValueError(
                "URoPE requires camera_intrinsics_norm and camera2referego."
            )

        crossview_hidden_states = hidden_states + view_emb
        crossview_hidden_states = einops.rearrange(
            crossview_hidden_states,
            "(bt v) (h w) c -> bt (v h w) c",
            bt=batch_size * sequence_length,
            v=view_count,
            h=height,
            w=width,
        )

        if crossview_attention_mask is not None:
            if crossview_attention_mask.ndim == 2:
                crossview_attention_mask = (
                    crossview_attention_mask.unsqueeze(0)
                )
            if crossview_attention_mask.shape[-2:] == (
                view_count,
                view_count,
            ):
                mask_batch_size = crossview_attention_mask.shape[0]
                expected_mask_batch_size = batch_size * sequence_length
                if mask_batch_size == batch_size:
                    crossview_attention_mask = (
                        crossview_attention_mask.repeat_interleave(
                            sequence_length,
                            dim=0,
                        )
                    )
                elif mask_batch_size != expected_mask_batch_size:
                    raise ValueError(
                        "Camera mask batch must be B or B*T, got {} "
                        "for B={} and T={}.".format(
                            mask_batch_size,
                            batch_size,
                            sequence_length,
                        )
                    )

        intrinsics = camera_intrinsics_norm.clone().float()
        intrinsics[..., 0, 0] = intrinsics[..., 0, 0] * width
        intrinsics[..., 1, 1] = intrinsics[..., 1, 1] * height
        intrinsics[..., 0, 2] = intrinsics[..., 0, 2] * width
        intrinsics[..., 1, 2] = intrinsics[..., 1, 2] * height
        intrinsics = intrinsics.reshape(
            batch_size * sequence_length,
            view_count,
            3,
            3,
        )

        viewmats = torch.linalg.inv(camera2referego.float())
        viewmats = viewmats.reshape(
            batch_size * sequence_length,
            view_count,
            4,
            4,
        )
        intrinsics = intrinsics.to(device=hidden_states.device)
        viewmats = viewmats.to(device=hidden_states.device)

        crossview_hidden_states = crossview_block(
            crossview_hidden_states,
            viewmats=viewmats,
            intrinsics=intrinsics,
            patch_height=height,
            patch_width=width,
            self_attention_mask=crossview_attention_mask,
        )
        crossview_hidden_states = einops.rearrange(
            crossview_hidden_states,
            "bt (v h w) c -> (bt v) (h w) c",
            bt=batch_size * sequence_length,
            v=view_count,
            h=height,
            w=width,
        )

        if mixer is None:
            return crossview_hidden_states

        original_states = hidden_states.view(
            batch_size,
            sequence_length * view_count,
            *hidden_states.shape[1:],
        )
        crossview_states = crossview_hidden_states.view(
            batch_size,
            sequence_length * view_count,
            *crossview_hidden_states.shape[1:],
        )
        return mixer(
            original_states,
            crossview_states,
            image_only_indicator=disable_crossview,
        ).flatten(0, 1)
