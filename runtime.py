"""Native ComfyUI conditioning, sampling and audio/video tensor operations."""
from dataclasses import dataclass
from fractions import Fraction
import math

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sample
import comfy.samplers
import comfy.utils
from comfy.ldm.minimax.model import PackedLayout, _frame_grid, _video_grid
from comfy_api.latest import InputImpl, Types
from comfy_extras.nodes_minimax_h3 import _empty_av_latent, _resize, temporal_shape
import node_helpers


@dataclass(frozen=True)
class Reference:
    kind: str
    images: torch.Tensor
    role: str
    fps: float = 24.0
    aligned: bool = False


def prompt_sections(scene, sound="N/A", music="N/A"):
    if "integrated_multimodal_description:" in scene or "subject_definitions:" in scene:
        return scene
    return f"integrated_multimodal_description: {scene}\n\noverall_soundscape: {sound}\n\nnon_diegetic_music: {music}"


def reference_prompt(prompt, references, task="reference generation"):
    if "subject_definitions:" in prompt:
        return prompt
    counts = {"image": 0, "video": 0}
    subjects, retention = [], []
    for ref in references:
        counts[ref.kind] += 1
        label = f"<{'Picture' if ref.kind == 'image' else 'Video'} {counts[ref.kind]}>"
        subjects.append(f"{label}: {ref.role}")
        retention.append(f"{label}: attribute_transfer - {ref.role}")
    return ("subject_definitions:\n" + "\n".join(subjects) + f"\n\nsummary:\n[{task}] {prompt}\n\n"
            + "retention_analysis:\n" + "\n".join(retention)
            + f"\n\ndetailed_description:\n{prompt}\n\noverall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A")


def image_batch(images):
    if images.ndim != 4 or images.shape[-1] not in (3, 4) or len(images) == 0:
        raise ValueError("Expected an IMAGE tensor [frames, height, width, 3 or 4].")
    return images[..., :3]


def resample_frames(images, fps, limit=None):
    images = image_batch(images)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Input fps must be positive.")
    count = max(1, math.ceil(len(images) * 24 / fps))
    if limit is not None:
        count = min(count, limit)
    indices = (torch.arange(count, device=images.device) * (fps / 24)).floor().long().clamp(max=len(images) - 1)
    return images.index_select(0, indices)


def pad_frames(images, count):
    images = images[:count]
    if len(images) < count:
        images = torch.cat((images, images[-1:].expand(count - len(images), -1, -1, -1)))
    return images


def empty_latent(width, height, frames):
    if width < 32 or height < 32 or width % 32 or height % 32 or frames < 1:
        raise ValueError("Use positive frame counts and dimensions divisible by 32.")
    latent, native_frames = _empty_av_latent(width, height, max(22, frames))
    latent.update(lynnreal_frames=frames, lynnreal_native_frames=native_frames)
    return latent


def keyframe_conditioning(clip, vae, prompt, width, height, frames, first=None, last=None):
    latent = empty_latent(width, height, frames)
    images, keyframes = [], []
    for image, index in ((first, 0), (last, latent["lynnreal_native_frames"] - 1)):
        if image is not None:
            if vae is None:
                raise ValueError("Image conditioning requires a video VAE.")
            image = image_batch(image)
            if len(image) != 1:
                raise ValueError("Each keyframe must contain one image. Use Reference Images for a batch.")
            resized = _resize(image, width, height, "center")
            images.append(resized)
            keyframes.append({"resolved_frame_index": index, "latent": vae.encode(resized)})
    text = prompt_sections(prompt)
    if first is not None:
        text = "Picture 1 is fully referenced at 0.00 seconds.\n" + text
    if last is not None:
        label = 2 if first is not None else 1
        text = f"Picture {label} defines the ending keyframe at {(latent['lynnreal_native_frames'] - 1) / 24:.6f} seconds.\n" + text
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text, images=images))
    cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes,
                                                       "lynnreal_task": "keyframes" if keyframes else "t2v"})
    return cond, latent


def reference_conditioning(clip, vae, prompt, references, width, height, frames, image_short_edge=768,
                           task="reference generation"):
    if not references:
        raise ValueError("Connect reference images or a reference/control video.")
    latent = empty_latent(width, height, frames)
    count = latent["lynnreal_native_frames"]
    items, blocks = [], []
    for ref in references:
        mm.throw_exception_if_processing_interrupted()
        images = image_batch(ref.images)
        if ref.kind == "image":
            if len(images) != 1:
                raise ValueError("Each image reference must be a single image.")
            h, w = images.shape[1:3]
            scale = image_short_edge / min(h, w)
            rw, rh = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
            images = _resize(images, rw, rh, "disabled")
            items.append({"type": "image", "data": images})
            blocks.append({"kind": "image", "latent_h": rh // 16, "latent_w": rw // 16,
                           "latent": vae.encode(images)})
        elif ref.kind == "video":
            images = resample_frames(images, ref.fps, count)
            if ref.aligned:
                rw, rh = width, height
                # Match LynnReal's aligned-control resize and terminal-frame padding.
                images = F.interpolate(images.movedim(-1, 1), size=(rh, rw), mode="bilinear",
                                       align_corners=False, antialias=True).movedim(1, -1)
                images = pad_frames(images, count)
            else:
                h, w = images.shape[1:3]
                scale = min(1.0, 768 / min(h, w), math.sqrt(768 * 1344 / (h * w)))
                rw, rh = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
                images = _resize(images, rw, rh, "disabled")
                n = max(5, ((len(images) - 5) // 17) * 17 + 5)
                images = pad_frames(images, n)
            sample_indices = list(range(0, len(images), 12))
            items.append({"type": "video", "data": images[sample_indices],
                          "timestamps": [i / 24 for i in sample_indices]})
            encoded = vae.encode(images)
            blocks.append({"kind": "video", "latent": encoded, "latent_t": encoded.shape[2],
                           "latent_h": rh // 16, "latent_w": rw // 16, "ref_audio_t": 0})
        else:
            raise ValueError(f"Unsupported reference kind: {ref.kind}")
    text = reference_prompt(prompt, references, task)
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text, minimax_ref_items=items))
    cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": blocks, "lynnreal_task": task})
    return cond, latent


def sigma_schedule(steps):
    base = torch.linspace(1, 0, steps + 1, dtype=torch.float32)
    return 12 * base / (1 + 11 * base)


def continuation_layout(text_len, future_t, height, width, audio_t, prefix):
    prefix_frames, prefix_latent = prefix
    prefix_t = prefix_latent.shape[2]
    layout = PackedLayout(text_len, future_t, height, width, audio_t,
                          keyframes=[{"resolved_frame_index": 0, "latent": prefix_latent}])
    grid, _ = _frame_grid(height, width)
    full_grid = _video_grid(prefix_t + future_t, grid, text_len)
    for start, end, kind in layout.segments:
        if kind == "video":
            layout.position_ids[start:end] = full_grid[prefix_t * len(grid):]
        elif kind == "audio":
            layout.position_ids[start:end, 0] += round(prefix_frames * 40 / 24)
    return layout


class ContinuationPatch:
    def __init__(self, prefix_frames):
        self.prefix_frames = prefix_frames

    def __call__(self, executor, x, timestep, context, transformer_options, **kwargs):
        payload = kwargs["minimax_payload"].copy()
        prefix = payload["cond_video_latents"][0]
        video, audio = x
        payload["layout"] = continuation_layout(context.shape[1], video.shape[2], video.shape[3],
                                                 video.shape[4], audio.shape[-1], (self.prefix_frames, prefix))
        return executor(x, timestep, context, transformer_options, **dict(kwargs, minimax_payload=payload))


def continuation_conditioning(clip, prompt, prefix_latent, prefix_frames, future_frames):
    total_frames, total_t, _ = temporal_shape(prefix_frames + future_frames)
    t = total_t - prefix_latent.shape[2]
    native_future = total_frames - prefix_frames
    video = prefix_latent.new_zeros((1, 24, t, *prefix_latent.shape[-2:]))
    audio = prefix_latent.new_zeros((1, 32, 2, round(native_future * 40 / 24)))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio)),
              "lynnreal_frames": future_frames, "lynnreal_native_frames": native_future,
              "lynnreal_prefix_frames": prefix_frames, "lynnreal_prefix": prefix_latent}
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt_sections(prompt)))
    cond = node_helpers.conditioning_set_values(cond, {
        "minimax_keyframes": [{"resolved_frame_index": 0, "latent": prefix_latent}],
        "lynnreal_task": "continuation"})
    return cond, latent


def sample(model, positive, latent, seed, sigmas=None, add_noise=True):
    config = model.model_options.get("lynnreal")
    if config is None:
        raise ValueError("Connect the MODEL from LynnReal Model Loader.")
    if config["variant"] == "flash" and positive[0][1].get("lynnreal_task") not in ("t2v", "keyframes"):
        raise ValueError("Flash supports text-to-video and native keyframes. Use Standard for reference/edit/continuation tasks.")
    active = model
    if "lynnreal_prefix" in latent:
        active = model.clone()
        active.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "lynnreal_continuation",
                                    ContinuationPatch(latent["lynnreal_prefix_frames"]))
    if sigmas is None:
        sigmas = sigma_schedule(config["steps"])
    noise = comfy.sample.prepare_noise(latent["samples"], seed) if add_noise else comfy.sample.prepare_empty_noise(latent["samples"])
    pbar = comfy.utils.ProgressBar(len(sigmas) - 1)

    def callback(step, denoised, current, total):
        mm.throw_exception_if_processing_interrupted()
        pbar.update_absolute(step + 1, total)

    result = comfy.sample.sample_custom(active, noise, 1.0, comfy.samplers.sampler_object("euler"),
                                        sigmas, positive, [], latent["samples"], noise_mask=latent.get("noise_mask"),
                                        callback=callback, seed=seed)
    return dict(latent, samples=result)


def decode(vae, latent, audio_vae=None):
    video, audio = latent["samples"].unbind()
    count = latent["lynnreal_frames"]
    if "lynnreal_prefix" in latent:
        video = torch.cat((latent["lynnreal_prefix"].to(video), video), dim=2)
    images = vae.decode(video)
    if images.ndim == 5:
        images = images.flatten(0, 1)
    start = latent.get("lynnreal_prefix_frames", 0)
    images = images[start:start + count]
    if len(images) != count:
        raise RuntimeError(f"VAE decoded {len(images)} future frames; expected {count}.")
    soundtrack = None
    if audio_vae is not None:
        waveform = audio_vae.decode(audio).movedim(-1, 1)
        rate = audio_vae.audio_sample_rate
        soundtrack = {"waveform": waveform[..., :round(count * rate / 24)], "sample_rate": rate}
    return images, soundtrack


def as_video(images, audio=None):
    return InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(24)))
