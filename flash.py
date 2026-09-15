"""Flash token selection expressed as ComfyUI ModelPatcher block replacements.

Selection/restoration follows LynnReal-Omni model/flash.py. Scratch tensors live
only in one diffusion forward and are never retained on the model or patcher.
"""
import torch
import comfy.patcher_extension


def spatial_layout(positions, tags, stride=2):
    video = tags == 0
    coordinates = positions[video]
    spatial = torch.ones_like(video)
    for axis in (1, 2):
        alphabet = torch.unique(coordinates[:, axis], sorted=True)
        retained = alphabet[::stride]
        if retained[-1] != alphabet[-1]:
            retained = torch.cat((retained, alphabet[-1:]))
        spatial &= torch.isin(positions[:, axis], retained)
    mask = ~video | spatial
    keep = mask.nonzero().flatten()
    inverse = torch.empty_like(tags, dtype=torch.long)
    inverse[keep] = torch.arange(keep.numel(), device=tags.device)
    anchors = (video & mask).nonzero().flatten()
    target = positions[anchors].float()
    for rows in (video & ~mask).nonzero().flatten().split(1024):
        source = positions[rows].float()
        distance = torch.cdist(source[:, 1:], target[:, 1:])
        distance.masked_fill_(source[:, None, 0] != target[None, :, 0], float("inf"))
        inverse[rows] = inverse[anchors[distance.argmin(dim=1)]]
    return keep, inverse


def reduced_segments(segments, keep):
    result = []
    for start, end, row in segments:
        left = int(torch.searchsorted(keep, start))
        right = int(torch.searchsorted(keep, end))
        if right > left:
            if isinstance(row, torch.Tensor):
                row = row.index_select(0, keep[left:right].to(row.device) - start)
            result.append((left, right, row))
    return result


def forward_scope(executor, x, timestep, context, transformer_options, **kwargs):
    options = transformer_options.copy()
    options["lynnreal_flash_state"] = {"text_tags": (kwargs.get("minimax_payload") or {}).get("text_token_tags")}
    try:
        return executor(x, timestep, context, options, **kwargs)
    finally:
        options["lynnreal_flash_state"].clear()


class FlashBlock:
    def __init__(self, first, last, gain):
        self.first, self.last, self.gain = first, last, gain

    def __call__(self, args, extra):
        state = args["transformer_options"]["lynnreal_flash_state"]
        if self.first:
            layout = args["layout"]
            tags = torch.empty(layout.seq_len, dtype=torch.long)
            for start, end, kind in layout.segments:
                tags[start:end] = 1 if kind == "text" else 0 if kind in ("video", "cond", "ref_img") else 2
                if kind == "text" and state["text_tags"] is not None:
                    tags[start:end] = state["text_tags"].flatten().cpu()
            keep, inverse = spatial_layout(layout.position_ids.cpu(), tags)
            keep, inverse = keep.to(args["img"].device), inverse.to(args["img"].device)
            full = args["img"]
            selected = full.index_select(0, keep)
            state.update(full=full, initial=selected, keep=keep, inverse=inverse,
                         segments=reduced_segments(args["mod_segments"], keep),
                         rope=args["rope_freqs"].index_select(1, keep))
            hidden = selected
        else:
            hidden = args["img"]
        inputs = dict(args, img=hidden, mod_segments=state["segments"], rope_freqs=state["rope"])
        hidden = extra["original_block"](inputs)["img"]
        if self.last:
            hidden = state["full"] + ((hidden - state["initial"]) * self.gain).index_select(0, state["inverse"])
        return {"img": hidden}


def apply_flash(model, config):
    compression = config["token_compression"]
    depth = len(model.model.diffusion_model.blocks)
    start, end = compression["full_prefix_blocks"], depth - compression["full_suffix_blocks"]
    if (depth != config["num_layers"] or not 0 < start < end < depth
            or compression["reduction"] != "select" or compression["spatial_stride"] != 2
            or not compression["preserve_text_tokens"] or not compression["preserve_audio_tokens"]
            or compression.get("full_refresh_blocks") or compression.get("text_stride", 1) != 1
            or compression.get("audio_stride", 1) != 1):
        raise ValueError("Unsupported LynnReal Flash compression configuration.")
    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 "lynnreal_flash", forward_scope)
    for i in range(start, end):
        patched.set_model_patch_replace(FlashBlock(i == start, i == end - 1, compression.get("residual_gain", 1.0)),
                                         "dit", "double_block", i)
    return patched
