"""Regenerate the UI workflows and native ComfyUI API prompts for review."""
import json
from pathlib import Path

from conftest import MODULE, ROOT


MEDIA = {
    "LoadImage": ({"required": {"image": (["upload_your_image.png"],)}}, ("IMAGE", "MASK")),
    "LoadVideo": ({"required": {"file": (["upload_your_video.mp4"],)}}, ("VIDEO",)),
    "GetVideoComponents": ({"required": {"video": ("VIDEO",)}}, ("IMAGE", "AUDIO", "FLOAT", "COMBO", "COMBO")),
    "SaveVideo": ({"required": {"video": ("VIDEO",), "filename_prefix": ("STRING", {"default": "video/LynnReal"}),
                                      "format": (["auto", "mp4"],), "format.codec": (["auto"],)}}, ("VIDEO",)),
    "SaveImage": ({"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "LynnReal"})}}, ()),
}


def schema(kind):
    if kind in MEDIA:
        return MEDIA[kind]
    node = MODULE.NODE_CLASS_MAPPINGS[kind]
    return node.INPUT_TYPES(), node.RETURN_TYPES


class Graph:
    def __init__(self, variant="standard"):
        self.api = {}
        self.variant = variant
        self.model = self.add("LynnRealModelLoader", bundle=variant)
        self.clip = self.add("LynnRealTextEncoderLoader", bundle=variant)
        self.vae = self.add("LynnRealVAELoader", bundle=variant)

    def add(self, kind, **values):
        definition, _ = schema(kind)
        inputs = {}
        for name, spec in definition["required"].items():
            typ, options = spec[0], spec[1] if len(spec) > 1 else {}
            if name in values:
                inputs[name] = values[name]
            elif isinstance(typ, list):
                inputs[name] = options.get("default", typ[0])
            elif typ in ("INT", "FLOAT", "BOOLEAN", "STRING"):
                inputs[name] = options.get("default", "")
            else:
                raise ValueError((kind, name))
        inputs.update(values)
        key = str(len(self.api) + 1)
        self.api[key] = {"class_type": kind, "inputs": inputs}
        return key

    @staticmethod
    def link(key, index=0):
        return [key, index]

    def condition(self, kind="LynnRealTextToVideo", **values):
        common = dict(clip=self.link(self.clip), prompt="A paper boat drifts on a quiet stream. Soft water sounds.",
                      width=768, height=448, frames=124)
        if kind != "LynnRealTextToVideo":
            common["vae"] = self.link(self.vae)
        if kind == "LynnRealImageEdit":
            common.pop("frames")
        common.update(values)
        return self.add(kind, **common)

    def sample(self, condition):
        return self.add("LynnRealSampler", model=self.link(self.model), positive=self.link(condition),
                        latent=self.link(condition, 1), seed=7)

    def finish(self, sampled, image=False):
        decoded = self.add("LynnRealDecode", vae=self.link(self.vae), audio_vae=self.link(self.vae, 1),
                           latent=self.link(sampled))
        if image:
            selected = self.add("LynnRealSelectEditedImage", images=self.link(decoded), frame_index=11)
            self.add("SaveImage", images=self.link(selected), filename_prefix="LynnReal/edit")
        else:
            self.save_video(decoded, 3)

    def save_video(self, source, index=0):
        self.add("SaveVideo", video=self.link(source, index), filename_prefix="video/LynnReal",
                 **{"format": "auto", "format.codec": "auto"})

    def image(self):
        return self.add("LoadImage", image="upload_your_image.png")

    def video(self):
        loaded = self.add("LoadVideo", file="upload_your_video.mp4")
        return self.add("GetVideoComponents", video=self.link(loaded))

    def write(self, name):
        target = ROOT / "examples"
        (target / "api").mkdir(exist_ok=True)
        (target / "api" / f"{name}.json").write_text(json.dumps(self.api, ensure_ascii=False, indent=2) + "\n")
        nodes, edges, slots, levels = [], [], {}, {}
        column_rows = {}
        for key, item in self.api.items():
            definition, outputs = schema(item["class_type"])
            specs = {**definition["required"], **definition.get("optional", {})}
            links = [v for v in item["inputs"].values() if isinstance(v, list)]
            level = max((levels[v[0]] + 1 for v in links), default=0)
            levels[key] = level
            row = column_rows.get(level, 0)
            column_rows[level] = row + 1
            inputs, widgets = [], []
            for field, spec in specs.items():
                typ, opts = spec[0], spec[1] if len(spec) > 1 else {}
                primitive = isinstance(typ, list) or typ in ("INT", "FLOAT", "BOOLEAN", "STRING")
                value = item["inputs"].get(field)
                linked = isinstance(value, list)
                if not primitive or linked:
                    socket = {"name": field, "type": "COMBO" if isinstance(typ, list) else typ, "link": None}
                    if primitive:
                        socket["widget"] = {"name": field}
                    slots[(key, field)] = len(inputs)
                    inputs.append(socket)
                if primitive:
                    fallback = opts.get("default", typ[0] if isinstance(typ, list) else 0)
                    widgets.append(fallback if linked or value is None else value)
                    if opts.get("control_after_generate"):
                        widgets.append("fixed")
            nodes.append({"id": int(key), "type": item["class_type"], "pos": [level * 410, row * 650],
                          "size": [340, 530 if any(isinstance(v, str) and len(v) > 60 for v in widgets) else 350],
                          "flags": {}, "order": int(key) - 1, "mode": 0, "inputs": inputs,
                          "outputs": [{"name": typ, "type": typ, "links": [], "slot_index": i} for i, typ in enumerate(outputs)],
                          "properties": {"Node name for S&R": item["class_type"]}, "widgets_values": widgets})
        by_id = {str(n["id"]): n for n in nodes}
        for key, item in self.api.items():
            for field, value in item["inputs"].items():
                if not isinstance(value, list):
                    continue
                origin, output = value
                index = slots[(key, field)]
                edge = len(edges) + 1
                typ = by_id[origin]["outputs"][output]["type"]
                edges.append([edge, int(origin), output, int(key), index, typ])
                by_id[origin]["outputs"][output]["links"].append(edge)
                by_id[key]["inputs"][index]["link"] = edge
        document = {"last_node_id": len(nodes), "last_link_id": len(edges), "nodes": nodes, "links": edges,
                    "groups": [], "config": {}, "extra": {"ds": {"scale": 0.6, "offset": [40, 40]}}, "version": 0.4}
        (target / f"{name}.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")


def main():
    for variant in ("standard", "flash"):
        graph = Graph(variant)
        graph.finish(graph.sample(graph.condition()))
        graph.write(f"01_{variant}_text_to_video")

    graph = Graph()
    graph.finish(graph.sample(graph.condition("LynnRealImageToVideo", first_image=graph.link(graph.image()),
                                              last_image=graph.link(graph.image()))))
    graph.write("02_first_last_frames")

    graph = Graph()
    references = graph.add("LynnRealReferenceImages", images=graph.link(graph.image()))
    references = graph.add("LynnRealReferenceImages", images=graph.link(graph.image()), references=graph.link(references))
    video = graph.video()
    references = graph.add("LynnRealReferenceVideo", images=graph.link(video), fps=graph.link(video, 2),
                           aligned=False, references=graph.link(references))
    graph.finish(graph.sample(graph.condition("LynnRealReferenceToVideo", references=graph.link(references))))
    graph.write("03_multiple_references")

    graph = Graph()
    video = graph.video()
    graph.finish(graph.sample(graph.condition("LynnRealVideoControl", control_frames=graph.link(video),
                                              fps=graph.link(video, 2), appearance=graph.link(graph.image()), task="body pose")))
    graph.write("04_pose_game_mesh_video_edit")

    graph = Graph()
    graph.finish(graph.sample(graph.condition("LynnRealImageEdit", image=graph.link(graph.image()),
                                              prompt="Change the jacket to red. Preserve identity and the background.")), image=True)
    graph.write("05_image_edit")

    graph = Graph()
    video = graph.video()
    graph.finish(graph.sample(graph.condition("LynnRealContinueVideo", source_frames=graph.link(video),
                                              fps=graph.link(video, 2), frames=102)))
    graph.write("06_video_continuation")

    graph = Graph()
    streamed = graph.add("LynnRealStream", model=graph.link(graph.model), clip=graph.link(graph.clip),
                         vae=graph.link(graph.vae), first_image=graph.link(graph.image()),
                         initial_prompt="The paper boat floats downstream.",
                         captions_json=json.dumps(["The paper boat continues floating downstream."] * 6),
                         width=768, height=448, frames=124)
    graph.save_video(streamed, 2)
    graph.write("07_segmented_stream")

    graph = Graph()
    video = graph.video()
    repaired = graph.add("LynnRealFrameRepair", model=graph.link(graph.model), clip=graph.link(graph.clip),
                         vae=graph.link(graph.vae), source_frames=graph.link(video), fps=graph.link(video, 2),
                         prompt="Remove rendering artifacts while preserving the composition and identity.")
    graph.save_video(repaired, 2)
    graph.write("08_frame_repair")

    graph = Graph("flash")
    condition = graph.condition(width=672, height=384)
    sampled = graph.sample(condition)
    refined = graph.add("LynnRealFlashRefine", model=graph.link(graph.model), positive=graph.link(condition),
                        latent=graph.link(sampled), width=1344, height=768, seed=8)
    graph.finish(refined)
    graph.write("09_flash_refinement")

    graph = Graph()
    light = graph.add("LynnRealLightVAELoader", folder="light-vae")
    sampled = graph.sample(graph.condition())
    decoded = graph.add("LynnRealDecode", vae=graph.link(light), audio_vae=graph.link(graph.vae, 1), latent=graph.link(sampled))
    graph.save_video(decoded, 3)
    graph.write("10_lightweight_vae")


if __name__ == "__main__":
    main()
