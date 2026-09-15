import ast
import json
from pathlib import Path
import inspect

import pytest
import torch


def test_no_external_inference_or_diffusers_dependency():
    root = Path(__file__).resolve().parents[1]
    for file in root.glob("*.py"):
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split('.')[0] in {"subprocess", "requests", "httpx", "diffusers"} for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split('.')[0] not in {"subprocess", "requests", "httpx", "diffusers"}


def test_registered_nodes_match_callable_sockets(plugin):
    for name, cls in plugin.NODE_CLASS_MAPPINGS.items():
        schema = cls.INPUT_TYPES()
        method = getattr(cls, cls.FUNCTION)
        parameters = inspect.signature(method).parameters
        for socket in {**schema.get("required", {}), **schema.get("optional", {})}:
            assert socket in parameters, (name, socket)
        assert name in plugin.NODE_DISPLAY_NAME_MAPPINGS


def test_upstream_to_native_conversion_roundtrip(plugin, tiny_bundle, tiny_native_state):
    from lynnreal_test.weights import converted_state
    converted, _ = converted_state(tiny_bundle / "transformer", "transformer")
    assert converted.keys() == tiny_native_state.keys()
    for key in converted:
        torch.testing.assert_close(converted[key], tiny_native_state[key], rtol=0, atol=0)
    # A second load reuses the complete conversion, without rewriting blocks.
    files = list((tiny_bundle / "transformer/.comfyui-cache").rglob("*.safetensors"))
    stamps = [p.stat().st_mtime_ns for p in files]
    converted_state(tiny_bundle / "transformer", "transformer")
    assert stamps == [p.stat().st_mtime_ns for p in files]


def test_quantized_qkv_and_swiglu_scales(plugin):
    from lynnreal_test.weights import convert_dit_group
    source = {}
    for i, name in enumerate(("q", "k", "v")):
        source[f"transformer_blocks.0.attn.to_{name}.weight_int8"] = torch.full((4, 8), i + 1, dtype=torch.int8)
        source[f"transformer_blocks.0.attn.to_{name}.weight_scale"] = torch.full((4, 1), (i + 1) / 10)
    source["transformer_blocks.0.ff.net.0.proj.weight_int8"] = torch.arange(32, dtype=torch.int8).reshape(8, 4)
    source["transformer_blocks.0.ff.net.0.proj.weight_scale"] = torch.arange(1, 9).float().unsqueeze(1)
    out = convert_dit_group(source, list(source))
    assert out["blocks.0.attn.qkv_proj.weight"].dtype == torch.int8
    assert out["blocks.0.attn.qkv_proj.weight"].shape == (12, 8)
    assert out["blocks.0.mlp.fc1.weight_scale"].flatten().tolist() == [5, 6, 7, 8, 1, 2, 3, 4]
    assert json.loads(bytes(out["blocks.0.mlp.fc1.comfy_quant"].tolist()))["format"] == "int8_tensorwise"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="INT8 kernel validation needs CUDA")
@pytest.mark.parametrize("rows", [1, 32, 35])
def test_native_int8_matches_released_per_token_math(plugin, rows):
    import comfy.ops
    from lynnreal_test.quantization import flash_operations
    operations = flash_operations()
    layer = operations.Linear(128, 64, bias=False, device="cpu")
    generator = torch.Generator().manual_seed(321)
    packed = torch.randint(-127, 128, (64, 128), generator=generator, dtype=torch.int8)
    scales = torch.rand(64, 1, generator=generator) * 0.02
    state = {"weight": packed, "weight_scale": scales,
             "comfy_quant": torch.tensor(list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8)}
    layer.load_state_dict(state, strict=False)
    layer.to("cuda")
    value = torch.randn(rows, 128, generator=generator).cuda().to(torch.bfloat16)
    scale = value.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    quantized = (value.float() / scale).round().clamp(-127, 127).to(torch.int8)
    padding = max(32, (rows + 7) // 8 * 8) - rows
    accumulator = torch._int_mm(torch.nn.functional.pad(quantized, (0, 0, 0, padding)), packed.cuda().t())[:rows]
    expected = (accumulator.float() * scale * scales.cuda().t()).to(torch.bfloat16)
    with torch.inference_mode():
        actual = layer(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_flash_loader_quantized_native_sampling(plugin, tiny_bundle, monkeypatch, clip):
    from safetensors.torch import load_file, save_file
    import comfy.quant_ops
    from lynnreal_test import nodes, runtime
    path = tiny_bundle / "transformer" / "diffusion_pytorch_model.safetensors"
    state = load_file(str(path))
    for key in list(state):
        if key.startswith("transformer_blocks.") and key.endswith(".weight") and state[key].ndim == 2 and (".attn." in key or ".ff." in key):
            value = state.pop(key).float()
            scale = value.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
            state[key + "_int8"] = (value / scale).round().clamp(-127, 127).to(torch.int8)
            state[key.removesuffix("weight") + "weight_scale"] = scale
    save_file(state, str(path))
    config = {"variant": "flash", "steps": 3, "num_layers": 4, "token_compression": {
        "full_prefix_blocks": 1, "full_suffix_blocks": 1, "spatial_stride": 2, "reduction": "select",
        "preserve_text_tokens": True, "preserve_audio_tokens": True}}
    (tiny_bundle / "inference_config.json").write_text(json.dumps(config))
    monkeypatch.setattr(nodes, "bundle_path", lambda name: tiny_bundle)
    model, = nodes.LynnRealModelLoader().load("flash")
    assert isinstance(model.model.diffusion_model.blocks[0].attn.qkv_proj.weight, comfy.quant_ops.QuantizedTensor)
    assert len(model.object_patches) == 4
    cond, latent = runtime.keyframe_conditioning(clip, None, "Test", 32, 32, 22)
    result = runtime.sample(model, cond, latent, 123)
    assert all(torch.isfinite(t).all() for t in result["samples"].unbind())


def test_examples_validate_in_native_comfy(plugin, tmp_path, monkeypatch):
    import asyncio
    import nodes as core_nodes
    import execution
    import folder_paths
    from comfy_extras.nodes_video import SaveVideo, LoadVideo, GetVideoComponents
    from PIL import Image
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path))
    Image.new("RGB", (32, 32)).save(tmp_path / "upload_your_image.png")
    # Prompt validation checks existence; actual media decoding is tested above.
    (tmp_path / "upload_your_video.mp4").touch()
    for key, cls in plugin.NODE_CLASS_MAPPINGS.items():
        monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS, key, cls)
    for cls in (SaveVideo, LoadVideo, GetVideoComponents):
        schema = cls.GET_SCHEMA()
        monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS, schema.node_id, cls)
    root = Path(__file__).resolve().parents[1] / "examples"
    for file in sorted((root / "api").glob("*.json")):
        result = asyncio.run(execution.validate_prompt("lynnreal-example", json.loads(file.read_text()), None))
        assert result[0], (file.name, result)
        ui = json.loads((root / file.name).read_text())
        by_id = {node["id"]: node for node in ui["nodes"]}
        for link, source, output, target, socket, typ in ui["links"]:
            assert link in by_id[source]["outputs"][output]["links"]
            assert by_id[target]["inputs"][socket]["link"] == link


def test_video_vae_per_head_qkv_order(plugin):
    from lynnreal_test.weights import convert_vae_group
    source = {f"decoder.transformer_blocks.0.attn.to_{p}.bias": torch.arange(i * 4, (i + 1) * 4).float()
              for i, p in enumerate(("q", "k", "v"))}
    out = convert_vae_group(source, list(source), heads=2)
    assert out["decoder.transformer_blocks.0.attn.to_qkv.bias"].tolist() == [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]


def test_text_layers_are_truncated_and_renamed(plugin):
    from lynnreal_test.weights import text_name
    assert text_name("model.language_model.layers.49.self_attn.q_proj.weight") == "model.layers.49.self_attn.q_proj.weight"
    assert text_name("model.language_model.layers.50.self_attn.q_proj.weight") is None
    assert text_name("model.visual.blocks.0.attn.qkv.weight") == "visual.blocks.0.attn.qkv.weight"


def test_rejects_missing_shards(plugin, tmp_path):
    from lynnreal_test.weights import Shards
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="invalid"):
        Shards(tmp_path)


def test_precise_four_and_three_step_schedules(plugin):
    from lynnreal_test.runtime import sigma_schedule
    torch.testing.assert_close(sigma_schedule(4), torch.tensor([1, 36/37, 12/13, 0.8, 0]), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(sigma_schedule(3), torch.tensor([1, 24/25, 12/14, 0]), rtol=1e-6, atol=1e-7)


def test_reference_order_and_aligned_padding(plugin, clip, vae):
    from lynnreal_test.runtime import Reference, reference_conditioning
    refs = (Reference("image", torch.zeros(1, 64, 64, 3), "actor A"),
            Reference("video", torch.rand(8, 32, 64, 3), "pose", fps=12, aligned=True),
            Reference("image", torch.ones(1, 64, 64, 3), "actor B"))
    positive, latent = reference_conditioning(clip, vae, "Move together", refs, 64, 64, 22, image_short_edge=64)
    assert [item["type"] for item in clip.requests[-1][1]["minimax_ref_items"]] == ["image", "video", "image"]
    assert "<Picture 2>: actor B" in clip.requests[-1][0]
    assert vae.encoded[1].shape == (22, 64, 64, 3)
    torch.testing.assert_close(vae.encoded[1][-1], vae.encoded[1][-2])
    assert positive[0][1]["minimax_refs"][1]["latent_t"] == latent["samples"].unbind()[0].shape[2]


def test_keyframes_anchor_native_timeline_and_trim(plugin, clip, vae):
    from lynnreal_test.runtime import keyframe_conditioning, decode
    positive, latent = keyframe_conditioning(clip, vae, "Turn around", 64, 64, 120,
                                            torch.zeros(1, 64, 64, 3), torch.ones(1, 64, 64, 3))
    assert [k["resolved_frame_index"] for k in positive[0][1]["minimax_keyframes"]] == [0, 123]
    images, audio = decode(vae, latent)
    assert len(images) == 120 and audio is None


def test_native_model_loader_returns_managed_model(plugin, tiny_bundle, monkeypatch):
    monkeypatch.setattr(plugin.nodes, "bundle_path", lambda _: tiny_bundle)
    model = plugin.nodes.LynnRealModelLoader().load("standard")[0]
    assert hasattr(model, "load_device")
    assert len(model.model.diffusion_model.blocks) == 4
    assert model.model_options["lynnreal"]["steps"] == 4


@pytest.mark.parametrize("variant,steps", [("standard", 4), ("flash", 3)])
def test_actual_comfy_sampling_forward_counts(plugin, tiny_model, clip, variant, steps):
    from lynnreal_test import runtime as rt
    from lynnreal_test.flash import apply_flash
    model = tiny_model
    if variant == "flash":
        model = apply_flash(model, {"num_layers": 4, "token_compression": {
            "full_prefix_blocks": 1, "full_suffix_blocks": 1, "spatial_stride": 2,
            "reduction": "select", "preserve_text_tokens": True, "preserve_audio_tokens": True}})
    model.model_options["lynnreal"] = {"variant": variant, "steps": steps}
    positive, latent = rt.keyframe_conditioning(clip, None, "A small test scene", 96, 96, 22)
    calls = []
    handle = model.model.diffusion_model.register_forward_hook(lambda *args: calls.append(1))
    try:
        with torch.inference_mode():
            generated = rt.sample(model, positive, latent, 7)
        assert len(calls) == steps
        assert all(torch.isfinite(t).all() for t in generated["samples"].unbind())
    finally:
        handle.remove()


def test_continuation_preserves_time_phase_and_future_only_decode(plugin, clip, vae, tiny_model):
    from lynnreal_test import runtime as rt
    prefix = torch.randn(1, 24, 7, 4, 4) * 0.1
    positive, latent = rt.continuation_conditioning(clip, "Keep moving", prefix, 22, 17)
    assert latent["samples"].unbind()[0].shape[2] == 5
    layout = rt.continuation_layout(3, 5, 4, 4, 28, (22, prefix))
    video_start = next(a for a, b, kind in layout.segments if kind == "video")
    audio_start = next(a for a, b, kind in layout.segments if kind == "audio")
    assert layout.position_ids[video_start, 0] == pytest.approx(3 + 22 * 5 / 3)
    assert layout.position_ids[audio_start, 0] == 40
    with torch.inference_mode():
        generated = rt.sample(tiny_model, positive, latent, 11)
    images, _ = rt.decode(vae, generated)
    assert len(images) == 17
    assert images[0, 0, 0, 0].item() == pytest.approx(22 / 255)


def test_flash_refinement_preserves_audio(plugin, tiny_model, clip):
    from lynnreal_test import runtime as rt
    tiny_model.model_options["lynnreal"] = {"variant": "flash", "steps": 3}
    positive, latent = rt.keyframe_conditioning(clip, None, "Scene", 64, 64, 22)
    video, audio = latent["samples"].unbind()
    audio.fill_(0.12)
    with torch.inference_mode():
        result = plugin.nodes.LynnRealFlashRefine().generate(tiny_model, positive, latent, 96, 96, 7)[0]
    assert result["samples"].unbind()[0].shape[-2:] == (6, 6)
    torch.testing.assert_close(result["samples"].unbind()[1], audio, rtol=0, atol=0)


def test_stream_and_repair_use_native_sampler(plugin, tiny_model, clip, vae):
    with torch.inference_mode():
        images, fps, video = plugin.nodes.LynnRealStream().generate(
            tiny_model, clip, vae, torch.zeros(1, 64, 64, 3), "Walk", '["Take another step"]', 64, 64, 39, 7)
    assert images.shape == (39, 64, 64, 3) and fps == 24
    assert video.get_components().audio is None
    with torch.inference_mode():
        repaired, fps, _ = plugin.nodes.LynnRealFrameRepair().generate(
            tiny_model, clip, vae, torch.zeros(2, 32, 32, 3), 24, "Correct the shape", 11, 7)
    assert len(repaired) == 2


def test_cancel_before_conditioning(plugin, clip, vae, monkeypatch):
    from lynnreal_test import runtime as rt
    def interrupted():
        raise RuntimeError("cancelled")
    monkeypatch.setattr(rt.mm, "throw_exception_if_processing_interrupted", interrupted)
    with pytest.raises(RuntimeError, match="cancelled"):
        rt.reference_conditioning(clip, vae, "Edit", [rt.Reference("image", torch.zeros(1, 64, 64, 3), "subject")], 64, 64, 22)
    assert not vae.encoded


def test_native_video_and_audio_roundtrip(plugin, tmp_path):
    from comfy_api.latest import InputImpl, Types
    from lynnreal_test.runtime import as_video
    images = torch.rand(24, 32, 32, 3)
    audio = {"waveform": torch.sin(torch.arange(32000).float() * 0.05).reshape(1, 1, -1).repeat(1, 2, 1) * 0.1,
             "sample_rate": 32000}
    destination = tmp_path / "clip.mp4"
    as_video(images, audio).save_to(str(destination), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
    result = InputImpl.VideoFromFile(str(destination)).get_components()
    assert len(result.images) == 24 and float(result.frame_rate) == 24
    assert result.audio["waveform"].shape[:2] == (1, 2)
    assert result.audio["sample_rate"] == 32000


def test_light_vae_depth_and_single_image_phase(plugin):
    from lynnreal_test.light_vae import LightVideoModel
    settings = {"image_context_tokens": 5, "image_context_output_phase": 3}
    with torch.device("meta"):
        model = LightVideoModel(26, settings)
    assert len(model.decoder.transformer_blocks) == 26
    assert model.image_context_tokens == 5 and model.image_output_phase == 3
