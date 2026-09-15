"""Flash W8A8 arithmetic inside ComfyUI's managed mixed-precision operations.

Activation scales use FP32, as in the released LynnReal reference path. Native
cast contexts still own transfers, LoRA application, and weight offloading.
"""
from functools import partial

import torch
import comfy.ops
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout


def flash_operations():
    base = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)

    class FlashOps(base):
        class Linear(base.Linear):
            def _forward(self, value, weight, bias):
                if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
                    return super()._forward(value, weight, bias)
                packed, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
                flat = value.reshape(-1, value.shape[-1]).float()
                scale = flat.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
                quantized = (flat / scale).round().clamp(-127, 127).to(torch.int8)
                rows = len(flat)
                if value.device.type == "cuda":
                    padding = max(32, (rows + 7) // 8 * 8) - rows
                    quantized = torch.nn.functional.pad(quantized, (0, 0, 0, padding))
                    accumulator = torch._int_mm(quantized, packed.contiguous().t())[:rows]
                else:
                    accumulator = quantized.int() @ packed.int().t()
                output = accumulator.float() * scale * weight_scale.float().reshape(1, -1)
                if bias is not None:
                    output += bias.float()
                return output.to(value.dtype).reshape(*value.shape[:-1], packed.shape[0])

    return FlashOps


def _mlp_forward(mlp, value):
    gate, up = mlp.fc1(value).chunk(2, dim=-1)
    return mlp.fc2(torch.nn.functional.silu(gate) * up)


def patch_flash_mlps(model):
    # Avoid ComfyUI's generic fused INT8 activation kernel: its BF16 scaling
    # differs from the released FP32 activation quantizer. Object patches are
    # installed/restored by ModelPatcher and affect only this model instance.
    for index, block in enumerate(model.model.diffusion_model.blocks):
        model.add_object_patch(f"diffusion_model.blocks.{index}.mlp.forward", partial(_mlp_forward, block.mlp))
