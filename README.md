# CPM OCR Studio

一个基于 **Go + Wails3** 的桌面应用，通过 **ONNX Runtime** 统一管理并运行：

- **HuggingFace 模型管理** —— 搜索、下载、删除本地模型（ONNX / Paddle / LLM）
- **PaddlePaddleOCR** —— 用 PP-OCRv6 检测 + 识别 ONNX 模型做文字识别
- **MiniCPM 小模型推理** —— 加载 optimum 导出的 Qwen2/MiniCPM 架构 ONNX 模型做对话生成

推理全部走 ONNX Runtime 共享库，不依赖 Python 运行时。

## 功能

| 模块 | 说明 |
| --- | --- |
| 模型管理 | 搜索 HuggingFace Hub、查看仓库文件、整库/按文件下载、本地模型列表/删除/打开目录、自定义模型根目录 |
| PaddleOCR | PP-OCRv6 三档模型便捷切换（tiny/small/medium）+ 文档方向分类，识别前自动矫正旋转、重建结果图、托盘/全局快捷键一键 OCR |
| 语音识别 | FunASR SenseVoiceSmall（ONNX 主引擎，中文优、内置标点）+ whisper.cpp（Metal 备用），支持文件转写与麦克风录音识别 |
| MiniCPM 对话 | 加载 `model.onnx + config.json + tokenizer.json`、ChatML 模板、流式输出、温度/Top-K/Top-P/重复惩罚/停止、**上传图片自动 OCR 后作为上下文发送**、推理模型思考过程折叠显示 |
| 运行环境 | 自动下载对应平台的 ONNX Runtime 共享库、显示全局配置 |

## 技术架构

```
main.go                     Wails3 入口（注册 4 个 Service）
state.go                    共享状态（配置 / OCR 引擎 / LLM 引擎 / 事件）
runtime_service.go          ONNX Runtime 库下载与初始化
hf_service.go               HuggingFace Hub 客户端 + 本地模型管理
ocr_service.go              PaddleOCR 引擎服务
llm_service.go              MiniCPM/Qwen ONNX 推理服务
internal/
  config/                   应用配置与目录
  hfhub/                    HF Hub API（搜索/文件树/下载）、本地模型扫描
  onnxmeta/                 极简 ONNX protobuf 解析器（读取输入输出名与形状）
  ort/                      onnxruntime_go 封装 + 运行时下载解压
  paddleocr/                DB 检测 + CTC 识别 + 方向分类，纯 Go 图像处理
  llm/                      纯 Go Qwen ByteLevel BPE 分词器 + KV Cache 采样循环
frontend/                   Vanilla JS + Vite 界面（深色主题，四个 Tab）
```

## 构建与运行

要求：Go 1.27+（最新稳定版）、Node.js 20+、`wails3` CLI、macOS / Linux / Windows。

```bash
# 1. 安装 wails3
go install github.com/wailsapp/wails/v3/cmd/wails3@latest

# 2. 安装前端依赖并生成绑定
cd frontend && npm install && cd ..
wails3 generate bindings

# 3. 构建前端（生成 frontend/dist）
cd frontend && npm run build && cd ..

# 4. 构建并打包
wails3 build              # 产物在 bin/（先放 build/appicon.png 图标；PATH 需含 ~/go/bin）

# 开发模式（热重载）
wails3 task dev
```

> 网络受限时请先配置 HTTP 代理（例如 `export https_proxy=http://127.0.0.1:7890`），
> 模型下载依赖 huggingface.co，运行时下载依赖 GitHub Releases。

## 首次使用

1. 打开应用，进入 **运行环境** 页，点击「下载 / 更新运行时」获取 ONNX Runtime 共享库
   （macOS `libonnxruntime.dylib`，Linux `libonnxruntime.so`，Windows `onnxruntime.dll`）。
   下载完成后应用启动时自动初始化。
2. **PaddleOCR** 页点击「安装中文模型」安装 PP-OCRv6 模型，再点「加载默认模型」即可识别图片（v6 单模型覆盖 50 种语言）。
3. **MiniCPM 对话** 页填入模型目录（见下），点「加载」后对话。

> 本仓库首次运行已预置：
> - ONNX Runtime 共享库（`~/.cpm_orc/lib`）
> - PP-OCRv6 small 检测/识别模型（`~/.cpm_orc/models/paddleocr/ch`，单模型支持 50 语言）
> - MiniCPM5-1B ONNX 模型（`~/.cpm_orc/models/llm/minicpm5-1b`，带 thinking/response 推理）
> - Qwen3-0.6B-Instruct int8 ONNX 模型（`~/.cpm_orc/models/llm/qwen3-0.6b-instruct`，带推理）
> - Qwen2.5-0.5B-Instruct int8 ONNX 模型（`~/.cpm_orc/models/llm/qwen2.5-0.5b-instruct`）

> 注：早期开发用的 `tiny-qwen2-test` 是随机权重测试模型，输出为乱码，仅用于验证推理管线，已移除。

## MiniCPM / Qwen ONNX 模型要求

LLM 引擎运行的是 **HF Optimum 导出的、带逐层 KV Cache 的 ONNX 模型**，目录结构：

```
model.onnx          # 解码器（含过去键值输入）
config.json         # qwen2 / llama / mistral 等架构
tokenizer.json      # ByteLevel BPE
```

输入约定：`input_ids` / `attention_mask` / `position_ids`，以及 `past_key_values.N.key` /
`past_key_values.N.value`（每层一对）；输出 `logits` 与 `present.N.key` / `present.N.value`。
这同时也是 transformers.js 仓库（如 `onnx-community/*`）的格式。

从 transformers 权重导出：

```bash
pip install optimum[exporters] onnxruntime
optimum-cli export onnx --model Qwen/Qwen2.5-0.5B-Instruct \
    --task text-generation-with-past ./qwen2.5-0.5b-onnx/
```

MiniCPM 系列（`openbmb/MiniCPM2.5-2.4B` 等，qwen2 架构）同样可以这样导出。

> 若导出结果使用「合并 KV Cache」或独立 encoder/decoder 文件，目前不受支持，会给出明确报错。

## 测试

```bash
go test ./internal/...        # 单元测试（OCR 后处理、分词器、ONNX 元数据、运行时下载）

# OCR 端到端（需联网下载约 15MB 模型 + 本机 onnxruntime 库）
CPMM_OCR_TEST=1 CPMM_ORT_LIB=/path/to/libonnxruntime.dylib go test ./internal/paddleocr/ -run TestRecogniseEndToEnd -v
```

## 对话

- **LLM 对话**：侧栏常规 tab，可加载 MiniCPM5-1B / Qwen3-0.6B / Qwen2.5-0.5B，聊天内容在**对话区内滚动**（内部滚动条），不随页面全局滚动。

## 系统集成

- **系统托盘**：菜单栏图标，支持「打开主窗口 / OCR 剪贴板图片 / 截图识别 / 退出」；关闭窗口隐藏到托盘，后台继续运行。
- **全局快捷键**（macOS/Linux/Windows 尽力支持）：
  - `CmdOrCtrl+Alt+O` — OCR 剪贴板图片
  - `CmdOrCtrl+Alt+S` — 交互截图后 OCR
  - `CmdOrCtrl+Alt+M` — 显示主窗口
- **一键识别**：识别结果通过事件推送到前端（自动切到 OCR 页展示），无需手动操作界面。

> 截图用系统命令：macOS `screencapture -i`；剪贴板图片用 `osascript` 提取。非 macOS 平台暂返回不支持提示。

## 加速与语音识别

Apple Silicon 加速 + 语音识别的调研、实现与实测结论见
[docs/ACCELERATION-ASR.md](docs/ACCELERATION-ASR.md)：CoreML EP 对小 CNN OCR 实测更慢（默认关闭）；
MLX 无 Go 绑定不可行；FunASR SenseVoiceSmall（ONNX）已落地为主引擎，whisper.cpp Metal 为备用，支持录音识别。

## 模型加载经验

详见 [docs/MODEL-GUIDE.md](docs/MODEL-GUIDE.md)：MiniCPM5-1B 与 Qwen3-0.6B 的加载/转换/踩坑/加新模型方法。

## 已知限制

- 当前内置模型为 Qwen2.5-0.5B、Qwen3-0.6B 与 MiniCPM5-1B。Qwen3 系列还有 1.7B / 4B 的 onnx-community 导出（int8/q4），可直接通过模型管理页搜索下载放入 `models/llm/`。Qwen3.8 系列最小 27B（BF16 约 55GB），对 CPU + ONNX 桌面应用不现实；如需更大模型建议配合 GPU 推理框架（vLLM/Ollama）。
- OCR 后处理用纯 Go 实现（连通域 + 最小外接矩形），效果与 OpenCV 版本接近，但极端场景可能有差异。
- LLM 推理在 CPU 上运行，适合 0.5B~4B 小模型；大模型建议换用 GPU/更专业的推理框架。
- 分词器针对 Qwen/MiniCPM 的 ByteLevel BPE 实现，其它 tokenizer 风格（Unigram 等）不受支持。