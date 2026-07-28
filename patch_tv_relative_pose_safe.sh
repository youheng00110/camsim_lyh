set -euo pipefail

FILE="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src/dwm/models/crossview_temporal_dit_mdtoken_tv.py"
BACKUP="${FILE}.bak.$(date +%Y%m%d_%H%M%S)"

cp -a "$FILE" "$BACKUP"
echo "backup: $BACKUP"

python3 - "$FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

if "tv_use_relative_ego_pose: bool" in text:
    raise SystemExit(
        "The file already contains tv_use_relative_ego_pose; "
        "abort to avoid applying the patch twice."
    )

def replace_once(old: str, new: str, name: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{name}: expected exactly one match, found {count}. "
            "Restore the backup and inspect the current source."
        )
    text = text.replace(old, new, 1)

replace_once(
'''    def forward(
        self,
        query_hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = query_hidden_states

        query_hidden_states = self.norm_q(query_hidden_states)
        context_hidden_states = self.norm_context(context_hidden_states)

        q = self.q_proj(query_hidden_states)
        k = self.k_proj(context_hidden_states)
        v = self.v_proj(context_hidden_states)
''',
'''    def forward(
        self,
        query_hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
        context_pose_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        residual = query_hidden_states

        query_hidden_states = self.norm_q(query_hidden_states)
        context_hidden_states = self.norm_context(context_hidden_states)

        if context_pose_embedding is not None:
            if context_pose_embedding.shape != context_hidden_states.shape:
                raise ValueError(
                    "context_pose_embedding must match context_hidden_states, "
                    f"but got pose={tuple(context_pose_embedding.shape)} and "
                    f"context={tuple(context_hidden_states.shape)}."
                )
            context_hidden_states = (
                context_hidden_states
                + context_pose_embedding.to(
                    device=context_hidden_states.device,
                    dtype=context_hidden_states.dtype,
                )
            )

        q = self.q_proj(query_hidden_states)
        k = self.k_proj(context_hidden_states)
        v = self.v_proj(context_hidden_states)
''',
"patch VTLocalCrossAttentionBlock.forward",
)

replace_once(
'''        tv_height_chunk_size: int = 0,
        mdtoken_bbox_config: Optional[dict] = None,
''',
'''        tv_height_chunk_size: int = 0,
        tv_use_relative_ego_pose: bool = False,
        tv_pose_translation_scale: float = 10.0,
        mdtoken_bbox_config: Optional[dict] = None,
''',
"add model config arguments",
)

replace_once(
'''        self.tv_gradient_checkpointing = tv_gradient_checkpointing
        self.disable_view_emb_on_temporal_module = disable_view_emb_on_temporal_module
''',
'''        self.tv_gradient_checkpointing = tv_gradient_checkpointing
        self.tv_use_relative_ego_pose = bool(tv_use_relative_ego_pose)
        self.tv_pose_translation_scale = float(tv_pose_translation_scale)
        if self.tv_pose_translation_scale <= 0:
            raise ValueError("tv_pose_translation_scale must be positive.")
        self.disable_view_emb_on_temporal_module = disable_view_emb_on_temporal_module
''',
"store model config arguments",
)

replace_once(
'''            self.tv_mixers = torch.nn.ModuleList([
                AlphaBlender(merge_factor, merge_strategy=merge_strategy)
                if mixer_type == "AlphaBlender" else Mixer(channel=inner_dim)
                for _ in range(len(self.tv_block_layers))
            ])
''',
'''            self.tv_mixers = torch.nn.ModuleList([
                AlphaBlender(merge_factor, merge_strategy=merge_strategy)
                if mixer_type == "AlphaBlender" else Mixer(channel=inner_dim)
                for _ in range(len(self.tv_block_layers))
            ])

            if self.tv_use_relative_ego_pose:
                self.tv_relative_pose_embeds = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        torch.nn.Linear(12, inner_dim, bias=False),
                        torch.nn.SiLU(),
                        torch.nn.Linear(inner_dim, inner_dim, bias=False),
                    )
                    for _ in range(len(self.tv_block_layers))
                ])
                for pose_encoder in self.tv_relative_pose_embeds:
                    torch.nn.init.zeros_(pose_encoder[-1].weight)
''',
"add per-layer relative-pose encoders",
)

replace_once(
'''        height: int,
        disable_tv: torch.BoolTensor,
        crossview_attention_mask: torch.Tensor = None,
        crossview_attention_index: torch.Tensor = None,
    ):
''',
'''        height: int,
        disable_tv: torch.BoolTensor,
        crossview_attention_mask: torch.Tensor = None,
        crossview_attention_index: torch.Tensor = None,
        tv_relative_pose_emb: torch.Tensor = None,
    ):
''',
"extend TV helper signature",
)

replace_once(
'''            query_hidden_states = einops.rearrange(
                tv_hidden_states_h,
                "b t v h w c -> (b t v h) w c",
            )

            tv_chunk = tv_block(query_hidden_states, context_hidden_states)
''',
'''            query_hidden_states = einops.rearrange(
                tv_hidden_states_h,
                "b t v h w c -> (b t v h) w c",
            )

            context_pose_embedding = None
            if tv_relative_pose_emb is not None:
                if tv_relative_pose_emb.shape != (
                    batch_size,
                    sequence_length,
                    3,
                    channel,
                ):
                    raise ValueError(
                        "tv_relative_pose_emb should be [B,T,3,C], "
                        f"but got {tuple(tv_relative_pose_emb.shape)}."
                    )

                local_pose_embedding = tv_relative_pose_emb[
                    :, :, None, None, :, None, None, :
                ].expand(
                    batch_size,
                    sequence_length,
                    view_count,
                    chunk_height,
                    3,
                    3,
                    width,
                    channel,
                )
                context_pose_embedding = einops.rearrange(
                    local_pose_embedding,
                    "b t v h lt lv w c -> (b t v h) (lt lv w) c",
                ).contiguous()

            tv_chunk = tv_block(
                query_hidden_states,
                context_hidden_states,
                context_pose_embedding,
            )
''',
"inject pose after context LayerNorm and before K/V projection",
)

replace_once(
'''        last_tv_emb = None
        for i, block in enumerate(self.transformer_blocks):
''',
'''        tv_relative_pose_features = None
        if self.enable_tv and self.tv_use_relative_ego_pose:
            if camera_transforms is None:
                raise ValueError(
                    "camera_transforms is required when "
                    "tv_use_relative_ego_pose=True."
                )
            if camera2referego is None:
                raise ValueError(
                    "camera2referego is required for the model-only relative "
                    "ego pose implementation."
                )
            if camera_transforms.ndim != 5 or camera2referego.ndim != 5:
                raise ValueError(
                    "camera_transforms and camera2referego should both be "
                    "[B,T,V,4,4], but got "
                    f"{tuple(camera_transforms.shape)} and "
                    f"{tuple(camera2referego.shape)}."
                )

            camera_to_current_ego = camera_transforms.to(
                device=hidden_states.device,
                dtype=torch.float32,
            )
            camera_to_reference_ego = camera2referego.to(
                device=hidden_states.device,
                dtype=torch.float32,
            )

            # camera->reference @ inverse(camera->current ego)
            # = current ego->reference ego
            ego_to_reference_all_views = (
                camera_to_reference_ego
                @ torch.linalg.inv(camera_to_current_ego)
            )
            ego_to_reference = ego_to_reference_all_views[:, :, 0]

            time_base = torch.arange(
                sequence_length,
                device=hidden_states.device,
            )
            time_offsets = torch.tensor(
                [-1, 0, 1],
                device=hidden_states.device,
            )
            tv_context_time_index = (
                time_base[:, None] + time_offsets[None, :]
            ).clamp(
                0,
                sequence_length - 1,
            ).long()

            context_ego_to_reference = ego_to_reference[
                :,
                tv_context_time_index.reshape(-1),
            ].reshape(
                batch_size,
                sequence_length,
                3,
                4,
                4,
            )
            query_ego_to_reference = ego_to_reference[:, :, None]

            # context ego -> query ego
            context_ego_to_query_ego = (
                torch.linalg.inv(query_ego_to_reference)
                @ context_ego_to_reference
            )

            relative_translation = (
                context_ego_to_query_ego[..., :3, 3]
                / self.tv_pose_translation_scale
            )
            relative_rotation = context_ego_to_query_ego[..., :3, :3]
            rotation_identity = torch.eye(
                3,
                device=hidden_states.device,
                dtype=torch.float32,
            ).reshape(1, 1, 1, 3, 3)
            relative_rotation_delta = (
                relative_rotation - rotation_identity
            ).flatten(-2)

            # [B,T,3,12]: xyz translation + flattened (R-I)
            tv_relative_pose_features = torch.cat(
                [
                    relative_translation,
                    relative_rotation_delta,
                ],
                dim=-1,
            )

            if not hasattr(self, "_tv_relative_pose_debug_printed"):
                self._tv_relative_pose_debug_printed = False
            if not self._tv_relative_pose_debug_printed:
                cross_view_error = (
                    ego_to_reference_all_views
                    - ego_to_reference_all_views[:, :, :1]
                ).abs().amax()
                center_identity_error = (
                    context_ego_to_query_ego[:, :, 1]
                    - torch.eye(
                        4,
                        device=hidden_states.device,
                        dtype=torch.float32,
                    )
                ).abs().amax()
                translation_norm = (
                    context_ego_to_query_ego[..., :3, 3].norm(dim=-1)
                )
                print(
                    "[TV relative ego pose] "
                    f"cross_view_error={cross_view_error.item():.6f}, "
                    f"center_identity_error={center_identity_error.item():.6f}, "
                    f"translation_mean={translation_norm.mean().item():.4f}, "
                    f"translation_max={translation_norm.max().item():.4f}",
                    flush=True,
                )
                self._tv_relative_pose_debug_printed = True

        last_tv_emb = None
        for i, block in enumerate(self.transformer_blocks):
''',
"compute query-context relative ego pose",
)

replace_once(
'''                tv_emb = tv_time_emb + view_cam_emb.to(dtype=hidden_states.dtype)
                last_tv_emb = tv_emb

                tv_disable = disable_tv
''',
'''                tv_emb = tv_time_emb + view_cam_emb.to(dtype=hidden_states.dtype)
                last_tv_emb = tv_emb

                tv_relative_pose_emb = None
                if tv_relative_pose_features is not None:
                    tv_relative_pose_emb = self.tv_relative_pose_embeds[
                        tv_layer_index
                    ](
                        tv_relative_pose_features.to(
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )

                tv_disable = disable_tv
''',
"project relative pose in each TV layer",
)

replace_once(
'''                        tv_disable,
                        crossview_attention_mask,
                        crossview_attention_index,
                        use_reentrant=False,
''',
'''                        tv_disable,
                        crossview_attention_mask,
                        crossview_attention_index,
                        tv_relative_pose_emb,
                        use_reentrant=False,
''',
"pass pose through checkpointed TV path",
)

replace_once(
'''                        tv_disable,
                        crossview_attention_mask,
                        crossview_attention_index,
                    )
''',
'''                        tv_disable,
                        crossview_attention_mask,
                        crossview_attention_index,
                        tv_relative_pose_emb,
                    )
''',
"pass pose through normal TV path",
)

path.write_text(text, encoding="utf-8")
print(f"patched: {path}")
PY

python3 -m py_compile "$FILE"

echo
echo "Patch applied and py_compile passed."
echo "Add these fields to pipeline.model config:"
echo '  "tv_use_relative_ego_pose": true,'
echo '  "tv_pose_translation_scale": 10.0'
echo
grep -nE \
  'tv_use_relative_ego_pose|tv_relative_pose_embeds|TV relative ego pose|context_pose_embedding' \
  "$FILE" | head -n 30
