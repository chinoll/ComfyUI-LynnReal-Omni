"""LynnReal nodes using only the running ComfyUI service and native model types."""
from pathlib import Path
import json

import torch
import folder_paths
import comfy.model_management as mm
import comfy.sd
import comfy.utils
from comfy_extras.nodes_minimax_h3 import _resize

from .weights import converted_state
from .flash import apply_flash
from .light_vae import LightVAE
from .quantization import flash_operations, patch_flash_mlps
from . import runtime as rt


folder_paths.add_model_folder_path("lynnreal", str(Path(folder_paths.models_dir) / "lynnreal"))


def bundles():
    names = set()
    for root in folder_paths.get_folder_paths("lynnreal"):
        root = Path(root)
        if root.is_dir():
            names.update(p.name for p in root.iterdir() if p.is_dir() and (p / "inference_config.json").is_file())
    return sorted(names) or ["standard", "flash"]


def bundle_path(name):
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Invalid LynnReal bundle name.")
    for root in folder_paths.get_folder_paths("lynnreal"):
        candidate = Path(root) / name
        if (candidate / "inference_config.json").is_file():
            return candidate
    raise FileNotFoundError(f"Put the complete {name} bundle in ComfyUI/models/lynnreal/{name}/.")


def geometry(frames=120):
    return {"width": ("INT", {"default": 1344, "min": 32, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32}),
            "frames": ("INT", {"default": frames, "min": 1, "max": 360})}


PROMPT = ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})
SEED = ("INT", {"default": 7, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})


class LynnRealModelLoader:
    CATEGORY = "LynnReal/loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL",)
    DESCRIPTION = "Load a local Standard or Flash bundle into ComfyUI's native H3 ModelPatcher. First load converts and caches checkpoint blocks."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"bundle": (bundles(),)}}

    def load(self, bundle):
        root = bundle_path(bundle)
        config = json.loads((root / "inference_config.json").read_text())
        if config.get("variant") not in ("standard", "flash") or config.get("steps") != (3 if config["variant"] == "flash" else 4):
            raise ValueError("Expected a four-step Standard or three-step Flash bundle.")
        state, dit_config = converted_state(root / "transformer", "transformer", mm.throw_exception_if_processing_interrupted)
        metadata = {"config": json.dumps({"transformer": {
            "norm_eps": dit_config.get("norm_eps", 1e-5), "qk_norm_eps": dit_config.get("qk_norm_eps", 1e-5),
            "final_norm_eps": dit_config.get("final_norm_eps", 1e-5)}})}
        options = {"dtype": torch.bfloat16}
        if config["variant"] == "flash":
            options["custom_operations"] = flash_operations()
        model = comfy.sd.load_diffusion_model_state_dict(state, model_options=options, metadata=metadata)
        if model is None:
            raise RuntimeError("This ComfyUI version cannot load native MiniMax H3. Update ComfyUI.")
        if config["variant"] == "flash":
            model = apply_flash(model, config)
            patch_flash_mlps(model)
        model.model_options["lynnreal"] = config
        return (model,)


class LynnRealTextEncoderLoader:
    CATEGORY = "LynnReal/loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("CLIP",)
    DESCRIPTION = "Load Qwen3-VL's first 50 layers and vision encoder as native ComfyUI CLIP."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"bundle": (bundles(),)}}

    def load(self, bundle):
        state, _ = converted_state(bundle_path(bundle) / "text_encoder", "text_encoder",
                                   mm.throw_exception_if_processing_interrupted)
        clip = comfy.sd.load_text_encoder_state_dicts([state], clip_type=comfy.sd.CLIPType.MINIMAX,
                                                      embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (clip,)


class LynnRealVAELoader:
    CATEGORY = "LynnReal/loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("VAE", "VAE")
    RETURN_NAMES = ("video_vae", "audio_vae")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"bundle": (bundles(),)}}

    def load(self, bundle):
        root = bundle_path(bundle)
        video_state, _ = converted_state(root / "vae", "vae", mm.throw_exception_if_processing_interrupted)
        video = comfy.sd.VAE(sd=video_state)
        audio_state, _ = converted_state(root / "audio_vae", "audio_vae", mm.throw_exception_if_processing_interrupted)
        audio = comfy.sd.VAE(sd=audio_state)
        return video, audio


class LynnRealPrompt:
    CATEGORY = "LynnReal/conditioning"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scene_and_dialogue": PROMPT, "soundscape": ("STRING", {"default": "N/A", "multiline": True}),
                              "music": ("STRING", {"default": "N/A", "multiline": True})}}

    def build(self, scene_and_dialogue, soundscape, music):
        return (rt.prompt_sections(scene_and_dialogue, soundscape, music),)


class LynnRealLightVAELoader:
    CATEGORY = "LynnReal/loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("VAE",)
    DESCRIPTION = "Load models/lynnreal/light-vae with the native 26-layer decoder. Optional replacement for the video VAE."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"folder": ("STRING", {"default": "light-vae"})}}

    def load(self, folder):
        if Path(folder).name != folder or folder in (".", ".."):
            raise ValueError("Use a folder name inside models/lynnreal.")
        for root in folder_paths.get_folder_paths("lynnreal"):
            path = Path(root) / folder
            if (path / "decode_config.json").is_file():
                state, config = converted_state(path, "vae", mm.throw_exception_if_processing_interrupted)
                return (LightVAE(state, config, json.loads((path / "decode_config.json").read_text())),)
        raise FileNotFoundError(f"Missing models/lynnreal/{folder}/decode_config.json")


class LynnRealReferenceImages:
    CATEGORY = "LynnReal/conditioning"
    FUNCTION = "append"
    RETURN_TYPES = ("LYNNREAL_REFERENCES",)
    DESCRIPTION = "Append every image in a batch as a separate reference. Chain to add more subjects."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "role": ("STRING", {"default": "Subject identity, clothing and appearance"})},
                "optional": {"references": ("LYNNREAL_REFERENCES",)}}

    def append(self, images, role, references=()):
        images = rt.image_batch(images)
        return (tuple(references) + tuple(rt.Reference("image", image.unsqueeze(0), role) for image in images),)


class LynnRealReferenceVideo:
    CATEGORY = "LynnReal/conditioning"
    FUNCTION = "append"
    RETURN_TYPES = ("LYNNREAL_REFERENCES",)
    DESCRIPTION = "Append video frames; aligned controls cover body/hand poses, game renders, mesh renders and editing."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "fps": ("FLOAT", {"default": 24, "min": 0.01, "max": 240}),
                              "aligned": ("BOOLEAN", {"default": True}),
                              "role": ("STRING", {"default": "Frame-aligned motion, geometry, camera and contact timing"})},
                "optional": {"references": ("LYNNREAL_REFERENCES",)}}

    def append(self, images, fps, aligned, role, references=()):
        return (tuple(references) + (rt.Reference("video", rt.image_batch(images), role, fps, aligned),),)


class LynnRealTextToVideo:
    CATEGORY = "LynnReal/conditioning"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "LATENT")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "prompt": PROMPT, **geometry()}}

    def encode(self, clip, prompt, width, height, frames):
        return rt.keyframe_conditioning(clip, None, prompt, width, height, frames)


class LynnRealImageToVideo:
    CATEGORY = "LynnReal/conditioning"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "LATENT")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "vae": ("VAE",), "first_image": ("IMAGE",),
                              "prompt": PROMPT, **geometry()}, "optional": {"last_image": ("IMAGE",)}}

    def encode(self, clip, vae, first_image, prompt, width, height, frames, last_image=None):
        return rt.keyframe_conditioning(clip, vae, prompt, width, height, frames, first_image, last_image)


class LynnRealReferenceToVideo:
    CATEGORY = "LynnReal/conditioning"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "LATENT")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "vae": ("VAE",), "references": ("LYNNREAL_REFERENCES",),
                              "prompt": PROMPT, **geometry(),
                              "image_short_edge": ("INT", {"default": 768, "min": 32, "max": 2048, "step": 32})}}

    def encode(self, clip, vae, references, prompt, width, height, frames, image_short_edge):
        return rt.reference_conditioning(clip, vae, prompt, references, width, height, frames, image_short_edge)


class LynnRealVideoControl:
    CATEGORY = "LynnReal/conditioning"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    DESCRIPTION = "Aligned pose, hand, game, mesh or video-edit conditioning. Controls are supplied as IMAGE frame batches."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "vae": ("VAE",), "control_frames": ("IMAGE",),
                              "fps": ("FLOAT", {"default": 24, "min": 0.01, "max": 240}),
                              "task": (["body pose", "hand pose", "game render", "mesh render", "video editing"],),
                              "prompt": PROMPT, **geometry()}, "optional": {"appearance": ("IMAGE",)}}

    def encode(self, clip, vae, control_frames, fps, task, prompt, width, height, frames, appearance=None):
        refs = []
        if appearance is not None:
            refs.extend(rt.Reference("image", image.unsqueeze(0), "Target subject appearance, style and materials")
                        for image in rt.image_batch(appearance))
        refs.append(rt.Reference("video", control_frames, f"{task}: preserve motion, camera, geometry and timing", fps, True))
        return rt.reference_conditioning(clip, vae, prompt, refs, width, height, frames,
                                         task="video editing" if task == "video editing" else "reference generation")


class LynnRealImageEdit:
    CATEGORY = "LynnReal/conditioning"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    DESCRIPTION = "Encode an image edit as a 22-frame camera-locked clip. Select an edited still after decoding."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "vae": ("VAE",), "image": ("IMAGE",), "prompt": PROMPT,
                              "width": geometry()["width"], "height": geometry()["height"]}}

    def encode(self, clip, vae, image, prompt, width, height):
        if len(image) != 1:
            raise ValueError("Image Edit accepts one image. Use Frame Repair to edit each frame of a batch.")
        refs = [rt.Reference("image", image, "Source image; preserve all features except the instructed edit")]
        return rt.reference_conditioning(clip, vae, prompt, refs, width, height, 22, task="image editing")


class LynnRealSampler:
    CATEGORY = "LynnReal/sampling"
    FUNCTION = "generate"
    RETURN_TYPES = ("LATENT",)
    DESCRIPTION = "Native ComfyUI Euler sampling: four Standard or three Flash evaluations, CFG 1, joint video/audio schedules."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "positive": ("CONDITIONING",), "latent": ("LATENT",), "seed": SEED}}

    def generate(self, model, positive, latent, seed):
        return (rt.sample(model, positive, latent, seed),)


class LynnRealDecode:
    CATEGORY = "LynnReal/decoding"
    FUNCTION = "decode"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "VIDEO")
    RETURN_NAMES = ("images", "audio", "fps", "video")
    DESCRIPTION = "Decode and trim native AV latents. Audio is absent when audio_vae is disconnected. VIDEO connects to Save Video."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "latent": ("LATENT",)}, "optional": {"audio_vae": ("VAE",)}}

    def decode(self, vae, latent, audio_vae=None):
        images, audio = rt.decode(vae, latent, audio_vae)
        return images, audio, 24.0, rt.as_video(images, audio)


class LynnRealFlashRefine:
    CATEGORY = "LynnReal/sampling"
    FUNCTION = "generate"
    RETURN_TYPES = ("LATENT",)
    DESCRIPTION = "Upscale Flash T2V video latents and run the trained two-step refinement tail. Preserve first-pass audio."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "positive": ("CONDITIONING",), "latent": ("LATENT",),
                              "width": geometry()["width"], "height": geometry()["height"], "seed": SEED}}

    def generate(self, model, positive, latent, width, height, seed):
        if model.model_options.get("lynnreal", {}).get("variant") != "flash" or positive[0][1].get("lynnreal_task") != "t2v":
            raise ValueError("Flash refinement requires text-to-video conditioning and a Flash MODEL.")
        if width % 32 or height % 32 or min(width, height) < 32:
            raise ValueError("Refinement dimensions must be positive multiples of 32.")
        video, audio = latent["samples"].unbind()
        batch, channels, time, h, w = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, h, w)
        flat = torch.nn.functional.interpolate(flat, size=(height // 16, width // 16), mode="bilinear", align_corners=False)
        video = flat.reshape(batch, time, channels, height // 16, width // 16).permute(0, 2, 1, 3, 4).contiguous()
        source = dict(latent, samples=comfy.nested_tensor.NestedTensor((video, audio)),
                      noise_mask=comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(audio))))
        result = rt.sample(model, positive, source, seed, sigmas=torch.tensor([0.94, 12 / 14, 0], dtype=torch.float32))
        result["samples"] = comfy.nested_tensor.NestedTensor((result["samples"].unbind()[0], audio.clone()))
        result.pop("noise_mask", None)
        return (result,)


class LynnRealSelectEditedImage:
    CATEGORY = "LynnReal/decoding"
    FUNCTION = "select"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "frame_index": ("INT", {"default": 11, "min": 0, "max": 359})}}

    def select(self, images, frame_index):
        if not 0 <= frame_index < len(images):
            raise ValueError(f"Frame index must be below {len(images)}.")
        return (images[frame_index:frame_index + 1].clone(),)


class LynnRealContinueVideo:
    CATEGORY = "LynnReal/conditioning"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    DESCRIPTION = "Encode the source's last 22 frames as a temporal prefix. Sampling/decoding outputs only future frames."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "vae": ("VAE",), "source_frames": ("IMAGE",),
                              "fps": ("FLOAT", {"default": 24, "min": 0.01, "max": 240}),
                              "prompt": PROMPT, **geometry()}}

    def encode(self, clip, vae, source_frames, fps, prompt, width, height, frames):
        source = rt.resample_frames(source_frames, fps)
        if len(source) < 22:
            raise ValueError("Continuation needs at least 22 source frames after resampling to 24 fps.")
        prefix = vae.encode(_resize(source[-22:], width, height, "center"))
        return rt.continuation_conditioning(clip, prompt, prefix, 22, frames)


class LynnRealStream:
    CATEGORY = "LynnReal/sampling"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "FLOAT", "VIDEO")
    RETURN_NAMES = ("images", "fps", "video")
    DESCRIPTION = "Standard native latent continuation: 22 initial frames, then 17 frames per caption; four forwards per chunk. Silent output."

    @classmethod
    def INPUT_TYPES(cls):
        g = geometry()
        g["frames"] = ("INT", {"default": 120, "min": 22, "max": 7200})
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",), "first_image": ("IMAGE",),
                              "initial_prompt": PROMPT, "captions_json": ("STRING", {"default": "[]", "multiline": True}),
                              **g, "seed": SEED}}

    def generate(self, model, clip, vae, first_image, initial_prompt, captions_json, width, height, frames, seed):
        if model.model_options.get("lynnreal", {}).get("variant") != "standard":
            raise ValueError("Streaming requires LynnReal Standard.")
        captions = json.loads(captions_json)
        needed = max(0, (frames - 22 + 16) // 17)
        if not isinstance(captions, list) or len(captions) < needed or not all(isinstance(c, str) and c.strip() for c in captions):
            raise ValueError(f"Supply at least {needed} nonempty captions as a JSON array, one per 17 new frames.")
        positive, latent = rt.keyframe_conditioning(clip, vae, initial_prompt, width, height, 22, first_image)
        generated = rt.sample(model, positive, latent, seed)
        images, _ = rt.decode(vae, generated)
        outputs = [images]
        history = generated["samples"].unbind()[0][:, :, -7:].clone()
        for index in range(needed):
            mm.throw_exception_if_processing_interrupted()
            positive, latent = rt.continuation_conditioning(clip, captions[index], history, 22, 17)
            generated = rt.sample(model, positive, latent, (seed + index + 1) % (1 << 64))
            images, _ = rt.decode(vae, generated)
            outputs.append(images)
            future = generated["samples"].unbind()[0]
            history = torch.cat((history.to(future), future), dim=2)[:, :, -7:].clone()
        images = torch.cat(outputs)[:frames]
        return images, 24.0, rt.as_video(images)


class LynnRealFrameRepair:
    CATEGORY = "LynnReal/sampling"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "FLOAT", "VIDEO")
    RETURN_NAMES = ("images", "fps", "video")
    DESCRIPTION = "Edit each source frame independently using four Standard forwards and a selected image from its 22-frame clip. May introduce flicker."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",), "source_frames": ("IMAGE",),
                              "fps": ("FLOAT", {"default": 24, "min": 0.01, "max": 240}),
                              "prompt": PROMPT, "frame_index": ("INT", {"default": 11, "min": 0, "max": 21}), "seed": SEED},
                "optional": {"appearance": ("IMAGE",)}}

    def generate(self, model, clip, vae, source_frames, fps, prompt, frame_index, seed, appearance=None):
        if model.model_options.get("lynnreal", {}).get("variant") != "standard":
            raise ValueError("Frame Repair requires LynnReal Standard.")
        source = rt.resample_frames(source_frames, fps)
        height, width = source.shape[1:3]
        result = []
        for image in source:
            mm.throw_exception_if_processing_interrupted()
            refs = [rt.Reference("image", image.unsqueeze(0), "Source frame; correct the instructed defects while preserving the scene")]
            if appearance is not None:
                refs.extend(rt.Reference("image", frame.unsqueeze(0), "Desired corrected appearance") for frame in appearance)
            positive, latent = rt.reference_conditioning(clip, vae, prompt, refs, width, height, 22, task="image editing")
            images, _ = rt.decode(vae, rt.sample(model, positive, latent, seed))
            result.append(images[frame_index:frame_index + 1].clone())
        images = torch.cat(result)
        return images, 24.0, rt.as_video(images)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    LynnRealModelLoader, LynnRealTextEncoderLoader, LynnRealVAELoader, LynnRealLightVAELoader, LynnRealPrompt,
    LynnRealReferenceImages, LynnRealReferenceVideo, LynnRealTextToVideo, LynnRealImageToVideo,
    LynnRealReferenceToVideo, LynnRealVideoControl, LynnRealImageEdit, LynnRealSampler,
    LynnRealDecode, LynnRealFlashRefine, LynnRealSelectEditedImage, LynnRealContinueVideo, LynnRealStream, LynnRealFrameRepair)}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LynnRealModelLoader": "LynnReal Model Loader · 模型加载",
    "LynnRealTextEncoderLoader": "LynnReal Text Encoder · 文本视觉编码器",
    "LynnRealVAELoader": "LynnReal Video & Audio VAE",
    "LynnRealLightVAELoader": "LynnReal Lightweight VAE · 轻量解码器",
    "LynnRealPrompt": "LynnReal Prompt · 场景、对白与声音",
    "LynnRealReferenceImages": "LynnReal Reference Images · 主体参考图",
    "LynnRealReferenceVideo": "LynnReal Reference Video · 参考视频",
    "LynnRealTextToVideo": "LynnReal Text to Video · 文生视频",
    "LynnRealImageToVideo": "LynnReal Image to Video · 首尾帧",
    "LynnRealReferenceToVideo": "LynnReal Reference Conditioning · 多参考生成",
    "LynnRealVideoControl": "LynnReal Pose / Game / Mesh / Video Edit",
    "LynnRealImageEdit": "LynnReal Image Edit · 图片编辑",
    "LynnRealSampler": "LynnReal Sampler · 原生音视频采样",
    "LynnRealDecode": "LynnReal Decode · 音视频解码",
    "LynnRealFlashRefine": "LynnReal Flash Refine · 两步细化",
    "LynnRealSelectEditedImage": "LynnReal Select Image · 编辑结果选帧",
    "LynnRealContinueVideo": "LynnReal Continue Video · 视频续写",
    "LynnRealStream": "LynnReal Stream · 分段长视频",
    "LynnRealFrameRepair": "LynnReal Frame Repair · 逐帧修复",
}
