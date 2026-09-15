"""Depth-distilled decoder on the native H3 VAE and ComfyUI memory manager."""
import torch
import comfy.model_management as mm
import comfy.model_patcher
import comfy.sd
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE


class LightVideoModel(MiniMaxH3VideoVAE):
    def __init__(self, layers, settings):
        super().__init__()
        self.decoder.transformer_blocks = self.decoder.transformer_blocks[:layers]
        self.image_context_tokens = settings["image_context_tokens"]
        self.image_output_phase = settings["image_context_output_phase"]

    def decode(self, z, output_buffer=None):
        if z.shape[2] != 1:
            return super().decode(z, output_buffer)
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(z)
        std = self.latents_std.view(1, -1, 1, 1, 1).to(z)
        expanded = (z * std + mean).expand(-1, -1, self.image_context_tokens, -1, -1).contiguous()
        result = self._adaptive_decode(expanded)[:, :, self.image_output_phase:self.image_output_phase + 1]
        result = self._finalize_pixels(result)
        if output_buffer is not None:
            output_buffer.copy_(result)
            return output_buffer
        return result


class LightVAE(comfy.sd.VAE):
    """Initialize the native VAE interface with the released 26-layer model."""
    def __init__(self, state, config, settings):
        if config.get("decoder_num_layers") != 26 or settings.get("tile_layout") != "native":
            raise ValueError("This loader supports the released 26-layer lightweight VAE with native tiles.")
        if settings.get("temporal_decode_protocol") != "released-segment-prepadding-v3":
            raise ValueError("Unsupported lightweight VAE temporal protocol.")
        self.first_stage_model = LightVideoModel(26, settings).eval()
        self.latent_channels, self.latent_dim, self.output_channels = 24, 3, 3
        self.upscale_ratio = (lambda n: max(1, (n - 2) // 5 * 17 + 5), 16, 16)
        self.downscale_ratio = (lambda n: max(1, (n - 5) // 17 * 5 + 2) if n > 1 else 1, 16, 16)
        self.upscale_index_formula = self.downscale_index_formula = (4, 16, 16)
        self.process_input = lambda images: images * 2 - 1
        self.process_output = lambda images: images
        self.crop_input, self.handles_tiling = True, True
        self.not_video, self.disable_offload = False, False
        self.pad_channel_value = self.extra_1d_channel = self.format_encoded = self.size = None
        self.audio_sample_rate = 44100
        self.working_dtypes = [torch.float16, torch.float32]
        self.device, self.output_device = mm.vae_device(), mm.intermediate_device()
        self.vae_dtype = mm.vae_dtype(self.device, self.working_dtypes)
        self.first_stage_model.to(self.vae_dtype)
        mm.archive_model_dtypes(self.first_stage_model)
        self.patcher = comfy.model_patcher.CoreModelPatcher(self.first_stage_model, load_device=self.device,
                                                           offload_device=mm.vae_offload_device())
        missing, unexpected = self.first_stage_model.load_state_dict(state, strict=False, assign=self.patcher.is_dynamic())
        missing = [key for key in missing if key != "decoder.mask_token"]
        if missing or unexpected:
            raise ValueError(f"Incomplete lightweight VAE: missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.memory_used_encode = lambda shape, dtype: int((9.5 * min(shape[2], 17) * shape[3] * shape[4]
                                                           + 1_300_000_000) * mm.dtype_size(dtype) * 1.03)
        self.memory_used_decode = lambda shape, dtype: int((9.5 * 40 * shape[3] * shape[4] * 256
                                                           + 270_000_000) * mm.dtype_size(dtype) * 1.03)
