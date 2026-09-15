"""Local Hugging Face bundles -> native ComfyUI H3 checkpoint tensors.

The inverse tensor transforms follow Hugging Face's Apache-2.0
convert_minimax_h3_to_diffusers.py at abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc.
"""
from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import os
import re
import uuid

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


CONVERSION_VERSION = 1


class Shards(Mapping):
    """Read only the tensors needed by the current conversion group."""

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        indices = sorted(self.folder.glob("*.safetensors.index.json"))
        self.files = {}
        if indices:
            if len(indices) != 1:
                raise ValueError(f"Expected one Safetensors index in {folder}")
            index = json.loads(indices[0].read_text())
            self.files = index["weight_map"]
            for file in set(self.files.values()):
                path = self.folder / file
                if Path(file).name != file or not path.is_file() or not path.resolve().is_relative_to(self.folder):
                    raise ValueError(f"Missing or invalid Safetensors shard: {file}")
        else:
            for file in sorted(self.folder.glob("*.safetensors")):
                with safe_open(file, framework="pt", device="cpu") as reader:
                    for key in reader.keys():
                        if key in self.files:
                            raise ValueError(f"Duplicate checkpoint tensor: {key}")
                        self.files[key] = file.name
        if not self.files:
            raise FileNotFoundError(f"No Safetensors weights in {folder}")

    def __getitem__(self, key):
        with safe_open(self.folder / self.files[key], framework="pt", device="cpu") as reader:
            return reader.get_tensor(key)

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def signature(self):
        files = sorted(set(self.files.values()) | {p.name for p in self.folder.glob("*.json")})
        return [(name, (self.folder / name).stat().st_size, (self.folder / name).stat().st_mtime_ns)
                for name in files]


def swap_halves(value):
    first, second = value.chunk(2, dim=0)
    return torch.cat((second, first), dim=0)


def dit_name(key):
    prefixes = {
        "token_refiner.refiner_blocks.": "token_refiner.blocks.",
        "transformer_blocks.": "blocks.",
        "time_embedder.linear_1.": "time_embedder.proj_in.",
        "time_embedder.linear_2.": "time_embedder.proj_out.",
        "proj_in.": "video_patch_proj.", "audio_proj_in.": "audio_patch_proj.",
        "context_embedder.": "condition_proj.", "norm_out.norm.": "final_layer.norm.",
        "norm_out.linear.": "final_layer.adaln_proj.linear.",
        "proj_out.": "final_layer.video_out.", "audio_proj_out.": "final_layer.audio_out.",
    }
    for old, new in prefixes.items():
        if key.startswith(old):
            key = new + key[len(old):]
            break
    for old, new in ((".attn.norm_q.", ".attn.q_norm."), (".attn.norm_k.", ".attn.k_norm."),
                     (".attn.to_out.0.", ".attn.out_proj."), (".ff.net.0.proj.", ".mlp.fc1."),
                     (".ff.net.2.", ".mlp.fc2.")):
        key = key.replace(old, new)
    return key


def convert_dit_group(source, keys):
    out = {}
    consumed = set()
    for key in keys:
        if key in consumed:
            continue
        if ".attn.to_" in key and any(f".attn.to_{p}." in key for p in ("q", "k", "v")):
            prefix, suffix = re.split(r"to_[qkv]\.", key, maxsplit=1)
            siblings = [prefix + f"to_{p}." + suffix for p in ("q", "k", "v")]
            value = torch.cat([source[name] for name in siblings], dim=0)
            consumed.update(siblings)
            target = dit_name(prefix + "qkv_proj." + suffix)
        else:
            value = source[key]
            target = dit_name(key)
            if ".ff.net.0.proj." in key:
                # Swap both the quantized rows and their per-output-channel scales.
                value = swap_halves(value)
        if target.endswith(".weight_int8"):
            target = target.removesuffix("_int8")
            out[target.removesuffix("weight") + "comfy_quant"] = torch.tensor(
                list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8)
        out[target] = value.contiguous()
    return out


def vae_name(key):
    key = key.replace("encoder.down_blocks.", "encoder.down.")
    key = key.replace(".resnets.", ".block.").replace(".conv_shortcut.", ".nin_shortcut.")
    key = key.replace(".downsamplers.0.", ".downsample.")
    key = key.replace("decoder.proj_in.", "decoder.x_embedder.")
    key = key.replace(".attn.to_out.0.", ".attn.to_out.")
    return key.replace(".ff.net.0.proj.", ".ff.w1.").replace(".ff.net.2.", ".ff.w2.")


def convert_vae_group(source, keys, heads):
    out, consumed = {}, set()
    for key in keys:
        if key in consumed:
            continue
        if any(f".attn.to_{p}." in key for p in ("q", "k", "v")):
            prefix, suffix = re.split(r"to_[qkv]\.", key, maxsplit=1)
            siblings = [prefix + f"to_{p}." + suffix for p in ("q", "k", "v")]
            values = [source[name] for name in siblings]
            # The native video VAE expects [head0:q,k,v; head1:q,k,v; ...].
            shape = values[0].shape
            value = torch.cat([v.reshape(heads, shape[0] // heads, *shape[1:]) for v in values], dim=1)
            value = value.reshape(3 * shape[0], *shape[1:])
            consumed.update(siblings)
            target = vae_name(prefix + "to_qkv." + suffix)
        else:
            value, target = source[key], vae_name(key)
            if ".ff.net.0.proj." in key:
                value = swap_halves(value)
        out[target] = value.contiguous()
    return out


def text_name(key):
    if key.startswith("model.language_model.layers."):
        if int(key.split(".")[3]) >= 50:
            return None
    if key == "lm_head.weight":
        return None
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    if key.startswith("model.visual."):
        return "visual." + key[len("model.visual."):]
    return key


def group_name(key, component):
    patterns = {
        "transformer": r"((?:transformer_blocks|token_refiner.refiner_blocks)\.\d+)\.",
        "vae": r"((?:decoder.transformer_blocks|encoder.down_blocks)\.\d+)\.",
        "text_encoder": r"((?:model.language_model.layers|model.visual.blocks)\.\d+)\.",
    }
    match = re.match(patterns.get(component, r"(?!)"), key)
    return match.group(1) if match else key.split(".")[0]


def converted_state(folder, component, check_interrupt=lambda: None):
    """Cache one block at a time, so conversion does not duplicate the full DiT in RAM."""
    source = Shards(folder)
    config_path = source.folder / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    identity = {"version": CONVERSION_VERSION, "component": component, "source": source.signature()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache = source.folder / ".comfyui-cache" / digest
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "index.json"
    if not manifest.is_file():
        groups = {}
        for key in source:
            if component == "text_encoder" and text_name(key) is None:
                continue
            groups.setdefault(group_name(key, component), []).append(key)
        files = []
        for i, keys in enumerate(groups.values()):
            check_interrupt()
            if component == "transformer":
                tensors = convert_dit_group(source, keys)
            elif component == "vae":
                tensors = convert_vae_group(source, keys, config.get("decoder_num_attention_heads", 32))
            elif component == "text_encoder":
                tensors = {text_name(k): source[k] for k in keys}
            else:
                tensors = {k: source[k] for k in keys}
            filename = f"{i:04d}.safetensors"
            destination = cache / filename
            temporary = cache / f"{uuid.uuid4().hex}.tmp"
            try:
                save_file(tensors, str(temporary))
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            files.append(filename)
            del tensors
        if component == "transformer":
            dim, theta = config.get("rope_freq_dim", 16), config.get("rope_theta", 10000.0)
            save_file({"rope.inv_freq": 1.0 / theta ** (torch.arange(dim, dtype=torch.float32) / dim)},
                      str(cache / "rope.safetensors"))
            files.append("rope.safetensors")
        elif component in ("vae", "audio_vae"):
            save_file({"latents_mean": torch.tensor(config["latents_mean"]),
                       "latents_std": torch.tensor(config["latents_std"])}, str(cache / "statistics.safetensors"))
            files.append("statistics.safetensors")
        temporary = cache / f"{uuid.uuid4().hex}.json.tmp"
        temporary.write_text(json.dumps({"identity": identity, "files": files}))
        os.replace(temporary, manifest)
    info = json.loads(manifest.read_text())
    state = {}
    for file in info["files"]:
        check_interrupt()
        if Path(file).name != file:
            raise ValueError("Invalid converted checkpoint filename.")
        state.update(load_file(str(cache / file), device="cpu"))
    return state, config
