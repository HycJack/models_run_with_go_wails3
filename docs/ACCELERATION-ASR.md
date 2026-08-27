# 加速方案与语音识别（CoreML / whisper.cpp Metal）记录

记录本次「Apple Silicon 加速」与「语音识别」的调研、实现与实测结论，
以及 MLX 的可行性判断。

## 1. 背景

用户希望：
- 让现有 OCR / LLM 的 ONNX 模型走 Apple GPU / ANE
- 支持语音识别（Qwen3-ASR 或类似）
- 确认能否支持 MLX

## 2. CoreML Execution Provider —— 实测结论：对我们的 OCR 反而更慢

**做了什么**：
- 现有 ONNX Runtime（1.29.0 macOS arm64）已内置 CoreML EP（`strings libonnxruntime.dylib` 可确认）
- `internal/ort.NewSession` 增加可选的 CoreML EP 注册（`AppendExecutionProviderCoreMLV2`，
  `ModelFormat: MLProgram` 以获得最佳 ANE 覆盖）
- OCR 引擎的 det/rec/cls/ori 四个会话可启用；`Config.CoreML` 控制（默认 **关闭**）

**实测（PP-OCRv6 small，1224×1584 样张）**：

| 模式 | 耗时 |
| --- | --- |
| CPU | ~1.0 s |
| CoreML EP（首次，含编译） | ~3.3 s |
| CoreML EP（预热后） | ~2.9 s |

**结论**：CoreML EP 对我们的**小 CNN OCR 模型没有加速**，反而更慢（CoreML 每调用有开销、
动态 shape 的检测模型大量算子回退 CPU + 分图开销）。**默认关闭**，保留开关供大模型场景尝试。

> LLM（KV 缓存 transformer + int4）CoreML EP 支持差，同样不回落到有用路径，保持 CPU。

## 3. MLX —— Go 生态不可行

- MLX（Apple 的 ML 框架）主要面向 **Python / Swift / C++**，**没有 Go 绑定**
- 在 Go + Wails 里直接"支持 MLX"不现实
- 结论：Apple Silicon 加速的现实路径是 **whisper.cpp / llama.cpp 的 Metal 后端**（本方案）

## 4. whisper.cpp Metal 语音识别 —— 已落地

**架构**：whisper.cpp 编译为独立 `whisper-cli`（Metal，EMBED_LIBRARY），
应用通过 `AsrService` 调起并解析转写结果。

**实测（Apple Silicon M 系列）**：
- 英文："Hello world, this is a speech recognition test using Whisper with metal acceleration."
  → 完全正确，**159 ms**
- 中文："今天天气很好…" → 正确转写（简体→繁体等 base 模型的轻微差异属正常），毫秒级

**实现**：
- `AsrService`（`asr_service.go`）：
  - `Status()` / `SetBin` / `SetModel` / `DownloadModel`（HF `ggerganov/whisper.cpp` 的 ggml-base.bin）
  - `Transcribe(audioPath, lang)` 与 `TranscribeBase64(b64, lang, filename)`
- 前端「语音识别」页签：引擎/模型状态、下载模型、选音频、语言选择、转写结果
- 配置：`config.json` 的 `whisperBin` / `whisperModel`（默认 `~/.cpm_orc/whisper/whisper-cli` 与 `~/.cpm_orc/models/whisper/ggml-base.bin`）

**whisper-cli 构建（macOS / Apple Silicon）**：
```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DWHISPER_METAL=ON
cmake --build build --config Release --target whisper-cli -j4
cp build/bin/whisper-cli ~/.cpm_orc/whisper/
```

**模型下载**（也可在应用内「语音识别 → 下载模型」）：
```bash
curl -L -o ~/.cpm_orc/models/whisper/ggml-base.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

## 5. Qwen3-ASR 现状（未采用）

- Qwen3-ASR-0.6B/1.7B 是**多模态**模型（AuT 音频编码器 + 投影 + Qwen3 LLM），无官方 ONNX
- 官方只提供 Python（transformers / vLLM）推理；CPU 实时率差
- 在 Go + ONNX Runtime 架构里完整支持需自研：音频解码 + Fbank 特征提取（纯 Go DSP）、
  音频编码器 ONNX 导出、LLM 解码适配 —— 数周级工作量
- **结论**：采用 FunASR SenseVoiceSmall（ONNX）作为主引擎（见下），whisper.cpp Metal 为备用

## 6. FunASR SenseVoiceSmall —— 已落地（主引擎）

FunASR（阿里达摩院）的 SenseVoiceSmall 是**官方 ONNX** 模型，正好走我们的 ONNX Runtime。

**模型**（`~/.cpm_orc/models/sensevoice/`）：
- `model.onnx`（241MB，fp32，输入 `speech` [B,T,560] 等 4 个张量）
- `am.mvn`（Kaldi CMVN 均值和缩放）
- `chn_jpn_yue_eng_ko_spectok.bpe.model`（SentencePiece 分词）

**纯 Go 实现的管线**（`internal/sensevoice/`）：
- `wav.go`：WAV 解析 + 重采样到 16k
- `fft.go`：基数 2 复数 FFT + mel 尺度换算
- `fbank.go`：Kaldi Fbank（hamming 窗 / 512 FFT / 80 mel 三角滤波器 / log floor）+ LFR（m=7,n=6，560 维拼接）+ CMVN；`ParseCMVN` 解析 am.mvn
- `sentencepiece.go`：极简 protobuf 解析 .model + BPE 解码（剔除 `<|zh|>` 等标签）
- `sensevoice.go`：`Transcribe(audio, lang, textnorm)` → argmax + CTC 折叠去空

**实测**（Apple Silicon）：
- 英文 72ms、中文 110ms，转写正确，自带标点与语言/情绪/事件标签（已剔除）

**应用集成**：
- `AsrService` 支持双后端：`sensevoice`（默认，进程内 ONNX）/ `whisper`（外部 whisper-cli Metal）
- 前端「语音识别」页：后端切换、SenseVoice 目录、whisper 备用配置

**踩坑**：Kaldi mel 滤波器组当 `right==center` 时会 0/0 → NaN，需钳制；否则特征 NaN 导致输出全空。

## 7. 录音识别

- 前端「🎙 录音识别」按钮：开始/停止录音
- 后端用 **ffmpeg（AVFoundation）** 采集麦克风（`ffmpeg -f avfoundation -i :0 -ac 1 -ar 16000`），停止后直接转写
- 需本机安装 ffmpeg（`brew install ffmpeg`）；录音中按钮红色脉冲、显示秒数
- 说明：macOS 麦克风权限给到运行应用的终端/应用

## 8. 验证

```bash
# ASR 端到端（需 whisper-cli + 模型已就位）
CPMM_ASR_TEST=1 go test . -run TestAsrTranscribe -v
```
- 全部单元/集成测试通过
- `wails3 build` 产物 `bin/cpm-orc` 启动正常