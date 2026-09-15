import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("COMFYUI_PATH", ROOT.parent / "ComfyUI-ssr-reference"))
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(ROOT / "tests"))

# A custom-node directory may contain hyphens; load it with a package spec just
# as ComfyUI does. Pytest otherwise tries to import the root as bare __init__.
spec = importlib.util.spec_from_file_location("lynnreal_test", ROOT / "__init__.py",
                                            submodule_search_locations=[str(ROOT)])
MODULE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)
sys.modules["__init__"] = MODULE


@pytest.fixture(scope="session")
def plugin():
    return MODULE


@pytest.fixture
def tiny_native_state(plugin):
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model
    model = MiniMaxH3Model(hidden_size=32, num_layers=4, token_refiner_num_layers=1,
                           num_attention_heads=2, attention_head_dim=16, ffn_hidden_size=32,
                           text_dim=16, timestep_input_dim=8, time_embed_hidden_size=32, time_embed_dim=16,
                           rope_inv_freq_len=2, operations=comfy.ops.disable_weight_init, dtype=torch.float32)
    generator = torch.Generator().manual_seed(123)
    state = {}
    for name, value in model.state_dict().items():
        if name == "rope.inv_freq":
            state[name] = torch.tensor([1.0, 0.01])
        elif "norm" in name and name.endswith("weight"):
            state[name] = torch.ones_like(value)
        else:
            state[name] = torch.randn(value.shape, generator=generator) * 0.03
    return state


@pytest.fixture
def tiny_model(plugin, tiny_native_state):
    import comfy.sd
    model = comfy.sd.load_diffusion_model_state_dict(tiny_native_state, model_options={"dtype": torch.float32})
    model.model_options["lynnreal"] = {"variant": "standard", "steps": 4}
    return model


@pytest.fixture
def tiny_bundle(tmp_path, plugin, tiny_native_state):
    from safetensors.torch import save_file
    from upstream_conversion import convert_transformer_key
    folder = tmp_path / "standard" / "transformer"
    folder.mkdir(parents=True)
    config = {"num_attention_heads": 2, "attention_head_dim": 16, "rope_freq_dim": 2,
              "rope_theta": 10000, "num_layers": 4, "hidden_size": 32}
    source = {}
    for key, tensor in tiny_native_state.items():
        source.update(convert_transformer_key(key, tensor, config))
    save_file(source, str(folder / "diffusion_pytorch_model.safetensors"))
    (folder / "config.json").write_text(json.dumps(config))
    (folder.parent / "inference_config.json").write_text(json.dumps({"variant": "standard", "steps": 4}))
    return folder.parent


class TinyClip:
    def __init__(self):
        self.requests = []

    def tokenize(self, prompt, **kwargs):
        self.requests.append((prompt, kwargs))
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.ones(1, 3, 16) * 0.02, {"minimax_token_tags": torch.ones(3, dtype=torch.long)}]]


class ShapeVAE:
    """Media-boundary fixture only; DiT/sampler tests run actual native ComfyUI."""
    def __init__(self):
        self.encoded = []

    def encode(self, images):
        from comfy_extras.nodes_minimax_h3 import temporal_shape
        self.encoded.append(images)
        t = 1 if len(images) == 1 else temporal_shape(len(images))[1]
        return torch.zeros(1, 24, t, images.shape[1] // 16, images.shape[2] // 16)

    def decode(self, latent):
        t = latent.shape[2]
        n = 1 if t == 1 else (t - 2) // 5 * 17 + 5
        return torch.arange(n).float()[:, None, None, None].expand(n, latent.shape[3] * 16, latent.shape[4] * 16, 3) / 255


@pytest.fixture
def clip():
    return TinyClip()


@pytest.fixture
def vae():
    return ShapeVAE()
