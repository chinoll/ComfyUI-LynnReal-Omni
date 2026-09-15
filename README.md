# ComfyUI-LynnReal-Omni

把 [LynnReal-Omni](https://github.com/LynnReal-AI/LynnReal-Omni) 的生成、参考控制、编辑和续写接入 ComfyUI，提供 **19 个节点、11 份可导入工作流**。

**模型在当前 ComfyUI 进程内运行。** 使用 ComfyUI 的原生 MiniMax H3 模型、`MODEL / CLIP / VAE / CONDITIONING / LATENT` 接口、采样器和显存管理。节点没有 HTTP 推理调用、外部进程、Diffusers pipeline 或独立推理服务。

当前版本为适配初版：已验证节点加载、原生小模型采样、INT8 计算、工作流参数和视频保存；**尚未使用完整预训练权重验证生成质量、峰值内存和性能**。分段长视频的实现范围见下文。

## 安装

1. 使用包含原生 MiniMax H3 支持的 ComfyUI。本适配核验版本为 [ComfyUI `a7b1d39`](https://github.com/Comfy-Org/ComfyUI/tree/a7b1d39d342d102f305797fb5ba12dc304d9c1f5)，需要 `comfy_extras/nodes_minimax_h3.py`、原生 H3 模型和当前的混合精度接口。旧版 ComfyUI 需要先更新，并在原有 Python 环境中更新它自己的依赖。
2. 把本目录放到 `ComfyUI/custom_nodes/ComfyUI-LynnReal-Omni/`，确保 `__init__.py` 位于该目录第一层。
3. 按下面的结构放入模型文件，重启 ComfyUI。
4. 将 `examples/` 第一层的 JSON 拖入 ComfyUI，选择模型和自己的图片/视频，提交工作流。

本节点没有额外运行依赖，沿用 ComfyUI 的 PyTorch、Safetensors、视频编解码和相关组件。无需安装上游项目或启动其他服务。安装包不包含模型权重。

### 模型目录

保留官方模型包的组件目录、配置、分片索引和全部 Safetensors 文件：

```text
ComfyUI/
├── custom_nodes/ComfyUI-LynnReal-Omni/
└── models/lynnreal/
    ├── standard/
    │   ├── inference_config.json
    │   ├── transformer/   # config.json + 完整权重及分片索引
    │   ├── text_encoder/  # config.json + 完整权重及分片索引
    │   ├── vae/           # config.json + 完整权重及分片索引
    │   └── audio_vae/     # config.json + 完整权重及分片索引
    ├── flash/
    │   ├── inference_config.json
    │   ├── transformer/
    │   ├── text_encoder/
    │   ├── vae/
    │   └── audio_vae/
    └── light-vae/         # 可选的轻量视频 VAE
        ├── config.json
        ├── decode_config.json
        └── ...safetensors
```

上游指定的模型地址：[Standard](https://huggingface.co/stdstu123/LynnReal-Onmi-beta-0.1)、[Flash](https://huggingface.co/stdstu123/LynnReal-Onmi-flash-beta-0.1)、[轻量 VAE](https://huggingface.co/stdstu123/LynnReal-Onmi-light-vae)。包是否完整应以对应仓库实际文件为准；只有配置或索引文件不足以运行。只需安装自己要使用的变体。共享的编码器和 VAE 可以使用目录符号链接。

默认从 `models/lynnreal` 查找，也支持 ComfyUI `extra_model_paths.yaml` 中名为 `lynnreal` 的模型路径。模型包目录可以改名，在各加载节点中选择对应名称。

### 首次加载与资源需求

首次加载会把官方张量名称、QKV 排列和 SwiGLU 排列转换为 ComfyUI 原生格式，并在各组件的 `.comfyui-cache/` 下保存分块缓存。后续加载复用缓存。Flash 保留原始 INT8 权重与每输出通道的 scale，不重新量化权重。转换阶段逐块处理，支持 ComfyUI 中断。

缓存会额外占用接近转换后权重大小的磁盘空间，模型目录需可写。更新来源配置或权重后会生成新的缓存目录；不用的旧缓存可在停止生成后自行删除。权重以本地 Safetensors 读取，节点不会自动下载模型。

大型 DiT 和 Qwen3-VL 编码器仍有较高内存需求。ComfyUI 负责加载和卸载；本适配不保证在 16 GB 显存或 16 GB 系统内存下完成全尺寸生成。Flash 降低的是 DiT 的计算与存储成本，文本编码器和视频解码仍会消耗资源。

## 节点与功能

在节点搜索中输入 `LynnReal`。

| 节点 | 功能 / 接口 |
| --- | --- |
| Model Loader | 加载 Standard 四步或 Flash 三步模型，输出原生 `MODEL` |
| Text Encoder | 原生 Qwen3-VL 文本与视觉编码器，输出 `CLIP` |
| Video & Audio VAE | 输出视频 `VAE` 和音频 `VAE` |
| Lightweight VAE | 可选 26 层轻量视频解码器，输出标准 `VAE` |
| Prompt | 组合场景、对白、环境音和背景音乐提示词，输出 `STRING` |
| Reference Images | 添加单张或批量主体参考图，可以串联 |
| Reference Video | 添加参考视频或逐帧对齐的控制视频，可以串联 |
| Text to Video | 文生音视频条件和空潜空间 |
| Image to Video | 首帧图生视频；可选连接尾帧 |
| Reference Conditioning | 多主体、多图片、混合图片与视频参考 |
| Pose / Game / Mesh / Video Edit | 身体姿态、手部姿态、游戏画面、网格渲染、视频外观编辑；可选外观参考图 |
| Image Edit | 指令图片编辑，构造 22 帧的编辑片段 |
| Sampler | ComfyUI 原生 Euler 联合音视频采样 |
| Decode | 输出 `IMAGE` 帧、`AUDIO`、24 fps 和原生 `VIDEO` |
| Flash Refine | Flash 文生视频第二阶段放大和两步细化，保留第一阶段音频 |
| Select Image | 从编辑片段选出一张图，默认第 11 帧（从 0 开始） |
| Continue Video | 使用源视频最后 22 帧续写，输出新增部分 |
| Stream | 根据分段提示词连续生成较长的无声视频 |
| Frame Repair | 根据指令逐帧修复视频；每个源帧执行一次四步生成 |

### 基本连接

1. `Text Encoder → Text to Video.clip`。
2. `Text to Video` 的 `CONDITIONING / LATENT` 分别连接 `Sampler.positive / latent`。
3. `Model Loader → Sampler.model`。
4. `Sampler → Decode.latent`；视频和音频 VAE 分别连接 `Decode.vae / audio_vae`。
5. `Decode.video → ComfyUI Save Video`。图片编辑则连接 `Decode.images → Select Image → Save Image`。

需要独立处理声音时，可连接 `Decode.audio` 到 ComfyUI 的音频节点。断开 `audio_vae` 后，Decode 返回无声视频，`audio` 输出为 `None`，此时不要将其连到要求实际音频的节点。

其他条件节点替换 `Text to Video` 即可复用加载、采样、解码部分。自定义的 `LYNNREAL_REFERENCES` 只用于串联参考项；模型、采样结果和媒体均使用 ComfyUI 标准类型。

### 变体与采样

- **Standard**：四次去噪前向，支持上述参考、编辑、控制、续写和逐帧修复功能。
- **Flash**：三次去噪前向，42 层网络、原始 W8A8 权重、空间 token 选择与残差恢复；支持文生视频和首尾帧条件。参考、编辑、续写节点会提示改用 Standard。
- **Flash Refine**：仅支持 Flash 文生视频，第一阶段结果放大后再执行两步。它需要相同的文本条件。
- Sampler 固定 CFG 1、Euler 和发布时的时间步；普通 KSampler 的默认调度不等价。外部标准节点虽能连接同类型接口，但没有本节点的帧数、续写等元数据处理。
- Flash 的矩阵运算使用 ComfyUI 管理的混合精度层和进程内 PyTorch INT8 运算，激活量化按上游参考公式使用 FP32 scale。未移植上游所有融合与性能优化，不宣称相同速度或逐像素一致。

### 输入、提示词和时间

- 输出固定 **24 fps**，宽高为 **32 的倍数**。
- H3 视频时间网格是 `17 × k + 5` 帧；普通生成至少内部运行 22 帧。请求其他帧数时，内部向上补齐，解码后裁剪为请求长度。
- 首尾帧节点的尾帧锚定在内部完整片段的末尾。希望导出包含准确的尾帧时，使用 `22 / 39 / 56 / 73 / 90 / 107 / 124 …` 帧；示例使用 124 帧。
- 图片输入是 ComfyUI `IMAGE`，视频输入先用 `Load Video → Get Video Components` 拆成图片批次和 fps。输入视频按 fps 重采样至 24 fps。
- 对齐控制会调整为目标宽高，并用最后一帧补齐较短的控制片段。普通参考视频保留其参考性质，不要求与输出逐帧对齐。
- 身体/手部姿态节点接收**已经制作好的姿态视频**，不包含人体/手部检测器。游戏和网格控制同样需要输入已有的渲染序列。
- 普通生成提示词会补成上游三段结构；参考生成会补成六段结构。完整的结构化提示词可直接输入。参考标签按类型分别编号为 `<Picture 1>`、`<Video 1>`，顺序由参考节点的串联顺序决定。
- 用 Prompt 节点的 `scene_and_dialogue` 描述画面和对白、`soundscape` 描述环境音、`music` 描述背景音乐。参考/编辑任务需要自定义声音时，可直接输入包含 `subject_definitions:` 和声音段落的完整六段提示词。

## 续写、分段视频与修复的边界

**Continue Video**：取输入的最后 22 帧，编码成 7 个时间潜变量。新增片段接着这个时间网格生成；解码时先合并前缀和新增潜变量，再裁掉前缀。输出只含新增视频及新增音频，不包含原视频。输入至少需要重采样后的 22 帧。

**Stream**：当前适配使用原生稠密潜空间续写：首段 22 帧，后续每段新增 17 帧，每段四步，保留最近 7 个时间潜变量作为上下文，输出无声视频。`captions_json` 是 JSON 字符串数组，至少需要 `ceil((frames - 22) / 17)` 条提示词。例如 124 帧需要 6 条。节点完成后统一返回全部帧，ComfyUI 生成过程中可以中断。

**这不是上游实验性 FramePack streaming 的逐算子移植。** 本版没有实现其 bootstrap、压缩历史、后续块四步加两步细化，也不在生成中向前端逐块推送可播放片段。长视频功能采用上面明确列出的原生续写策略，时序质量和长时间稳定性仍需完整模型实测。

**Frame Repair**：每个源帧独立构造 22 帧编辑片段，四步采样后选出指定帧。输出无声；源帧宽高必须为 32 的倍数。它没有跨帧一致性约束，可能闪烁，成本随输入帧数线性增加。

**Lightweight VAE**：加载已发布配置中的 26 层解码器和单图上下文/输出相位规则，使用原生 H3 的分块解码路径；未移植上游针对硬件的 tile 优化。轻量 VAE 不替代音频 VAE，也不减少 DiT 或文本编码器的大小。

## 示例工作流

`examples/` 第一层文件可拖入界面；`examples/api/` 为对应的 **ComfyUI 自身** API prompt 格式，供已有 ComfyUI 客户端使用。

| 文件 | 用途 |
| --- | --- |
| `01_standard_text_to_video.json` | Standard 文生音视频 |
| `01_flash_text_to_video.json` | Flash 文生音视频 |
| `02_first_last_frames.json` | 首尾帧控制；仅首帧时断开尾帧输入 |
| `03_multiple_references.json` | 两张主体图和一个普通参考视频 |
| `04_pose_game_mesh_video_edit.json` | 在 task 中切换身体/手部/游戏/网格/外观编辑 |
| `05_image_edit.json` | 图片编辑并保存选定的图 |
| `06_video_continuation.json` | 续写并保存新增片段 |
| `07_segmented_stream.json` | 多段提示词生成无声长视频 |
| `08_frame_repair.json` | 逐帧修复 |
| `09_flash_refinement.json` | Flash 两阶段文生视频 |
| `10_lightweight_vae.json` | 用轻量 VAE 解码视频 |

素材节点里的 `upload_your_image.png` / `upload_your_video.mp4` 是占位文件名，请在节点中选择自己的输入。示例不是模型生成结果。

## 验证记录

核验于 2026-09-15：Python 3.12、PyTorch 2.11 / CUDA 13.0、RTX 5060 Ti，ComfyUI 版本见安装部分。

- **24 项测试通过**：官方转换函数的张量往返、QKV 和 FFN/scale 排列、缓存复用、参考顺序、帧数处理、取消处理、续写时间坐标、Flash 音频保留、原生音视频保存/重新读取等。
- 使用小尺寸的**真实原生 H3 网络与 ComfyUI 采样器**验证 Standard 四次/Flash 三次前向，另验证量化 Flash 从加载到采样的完整小模型路径。文本编码器和 VAE 在条件/续写测试中使用边界替身；不能据此声称完整模型已跑通。
- CUDA INT8 层针对多种 token 数量与上游 FP32 动态 scale / INT32 累加公式逐值一致；不代表整个模型与上游生成结果逐值一致。
- 11 份示例均通过 ComfyUI 原生 prompt 校验，界面 JSON 的连接同步检查通过。
- 单独启动 ComfyUI，在 `/object_info` 中确认全部 19 个节点成功注册。
- **未验证**：完整预训练模型的画质、音画同步品质、参考保真度、完整 VAE 输出数值一致性、长视频稳定性、性能和峰值内存。

开发环境执行：

```bash
COMFYUI_PATH=/path/to/ComfyUI python -m pytest tests -q
```

测试需要支持的 ComfyUI 及其完整依赖，另安装 pytest。CUDA 不可用时量化层的 CUDA 数值测试会跳过。开发用虚拟环境不随压缩包分发。

## 来源与许可

本实现对照 [LynnReal-Omni `f6ee8b8`](https://github.com/LynnReal-AI/LynnReal-Omni/tree/f6ee8b8ecfeafa354a56d91b27a530ac8897d8d8) 和 [Diffusers 官方 H3 转换器](https://github.com/huggingface/diffusers/blob/abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc/scripts/convert_minimax_h3_to_diffusers.py)。运行时使用已安装 ComfyUI 的原生实现，不附带上游完整推理框架或模型。

保留上游 [MiniMax H3 社区许可](LICENSE) 和 [NOTICE](NOTICE)。测试中的官方转换函数片段另按 [Apache-2.0](LICENSES/Apache-2.0.txt) 分发。各模型和 ComfyUI 自身的许可仍分别适用。
