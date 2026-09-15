# Apache-2.0. Copyright 2025 The MiniMax authors and The HuggingFace Team.
# Extracted unchanged from Diffusers abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc.
from typing import Any
import torch
MINIMAX_H3_TRANSFORMER_DROPPED_KEYS = ("rope.inv_freq",)


def split_fused_qkv(
    weight: torch.Tensor, num_attention_heads: int, attention_head_dim: int
) -> tuple[torch.Tensor, ...]:
    """Split a fused `[q_all; k_all; v_all]` QKV weight into separate `to_q` / `to_k` / `to_v` weights.

    The input is the *reference model* layout — what `MiniMaxH3DiTModel.state_dict()` holds after the reference's
    load-time reorder — i.e. the three logical projection matrices stacked contiguously, NOT the raw checkpoint's
    per-head interleave (see `reorder_interleaved_qkv`, which the shard streamer applies first).
    """
    inner_dim = num_attention_heads * attention_head_dim
    if weight.shape[0] != 3 * inner_dim:
        raise ValueError(
            f"fused qkv weight has {weight.shape[0]} rows, expected "
            f"{3 * inner_dim} = 3 * {num_attention_heads} heads * {attention_head_dim}."
        )
    query, key, value = weight.split(inner_dim, dim=0)
    return tuple(tensor.contiguous() for tensor in (query, key, value))

def convert_transformer_key(
    source_key: str, tensor: torch.Tensor, config: dict[str, Any]
) -> list[tuple[str, torch.Tensor]]:
    """Convert one original key/tensor pair into the diffusers key/tensor pair(s) it maps to."""
    if source_key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        return []

    target_key = source_key
    if target_key.startswith("token_refiner.blocks."):
        target_key = target_key.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif target_key.startswith("blocks."):
        target_key = target_key.replace("blocks.", "transformer_blocks.", 1)
    target_key = target_key.replace("time_embedder.proj_in.", "time_embedder.linear_1.")
    target_key = target_key.replace("time_embedder.proj_out.", "time_embedder.linear_2.")
    target_key = target_key.replace("video_patch_proj.", "proj_in.")
    target_key = target_key.replace("audio_patch_proj.", "audio_proj_in.")
    target_key = target_key.replace("condition_proj.", "context_embedder.")
    target_key = target_key.replace("final_layer.norm.", "norm_out.norm.")
    target_key = target_key.replace("final_layer.adaln_proj.linear.", "norm_out.linear.")
    target_key = target_key.replace("final_layer.video_out.", "proj_out.")
    target_key = target_key.replace("final_layer.audio_out.", "audio_proj_out.")
    target_key = target_key.replace(".attn.q_norm.", ".attn.norm_q.")
    target_key = target_key.replace(".attn.k_norm.", ".attn.norm_k.")
    target_key = target_key.replace(".attn.out_proj.", ".attn.to_out.0.")

    if target_key.endswith(".attn.qkv_proj.weight"):
        # `convert_transformer_key` consumes tensors in the reference model's state-dict layout, where the fused QKV
        # rows are already `[q_all; k_all; v_all]`. Raw checkpoint shards are per-head interleaved instead; the shard
        # streamer (`convert_transformer`) normalizes them with `reorder_interleaved_qkv` before calling this.
        query, key, value = split_fused_qkv(tensor, config["num_attention_heads"], config["attention_head_dim"])
        prefix = target_key.removesuffix("qkv_proj.weight")
        return [(f"{prefix}to_q.weight", query), (f"{prefix}to_k.weight", key), (f"{prefix}to_v.weight", value)]

    if target_key.endswith(".mlp.fc1.weight"):
        # The reference computes `fc2(silu(gate) * value)` from a fused `[gate; value]`; diffusers' `SwiGLU` computes
        # `value * silu(gate)` from a fused `[value; gate]`, so the two halves swap places. Identical transform to the
        # video VAE's `ff.w1` (see `convert_video_vae_key`).
        gate, value = tensor.chunk(2, dim=0)
        target_key = target_key.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight")
        return [(target_key, torch.cat([value, gate], dim=0).contiguous())]

    target_key = target_key.replace(".mlp.fc2.", ".ff.net.2.")
    return [(target_key, tensor)]
