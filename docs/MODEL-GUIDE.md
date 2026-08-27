# 模型加载与使用经验总结（MiniCPM5-1B / Qwen3-0.6B）

本文总结在本应用中成功加载并跑通 **MiniCPM5-1B** 与 **Qwen3-0.6B-Instruct** 两个
ONNX 模型的过程、踩过的坑与可复用的方法。

## 1. 两个模型的基本情况

| 模型 | 来源 | 文件 | 架构 | 大小 | CPU 速度 |
| --- | --- | --- | --- | --- | --- |
| MiniCPM5-1B | `Mike0021/MiniCPM5-1B-ONNX-Web` | `onnx/model_q4.onnx`（q4，fp16）→ **fp32 转换后** | `llama` | 902MB → 1.35GB | ≈20 tokens/s |
| Qwen3-0.6B-Instruct | `onnx-community/Qwen3-0.6B-ONNX` | `onnx/model_int8.onnx` | `qwen3` | 617MB | ≈25 tokens/s |

两者都能直接对话（Qwen3-0.6B、MiniCPM5-1B 为推理模型，带 thinking/response 思考流程）。

## 2. 引擎对模型格式的要求（决定"能不能跑"）

一个可运行的模型目录必须包含：
```
model.onnx          # 解码器
config.json         # 架构与维度
tokenizer.json      # ByteLevel BPE
```

**必须满足的 ONNX 格式**（transformers.js / Optimum 逐层 KV Cache 格式）：
- 输入：`input_ids`、`attention_mask`、`position_ids` + 每层一对 `past_key_values.N.key/.value`
- 输出：`logits` + 每层 `present.N.key/.value`
- 激活精度 **fp32 或 int8 动态量化**（fp16 需转成 fp32）

**不受支持的格式**（会给出明确报错）：
- 合并 KV Cache（单一 `past_key_values` 张量，onnxruntime-genai 风格）
- 混合架构（Qwen3.5 的 Gated DeltaNet：`past_conv.N` / `past_recurrent.N` 状态输入 + 3D position_ids）
- fp16 模型（yalue/onnxruntime_go 绑定无法创建 fp16 张量）
- 非 ByteLevel BPE 的 tokenizer

## 3. 加载模型踩过的坑

### 3.1 fp16 模型需要转成 fp32
MiniCPM5 官方 q4 版是 **fp16 激活 + int4 权重**，CPU ORT 能跑，但 Go 绑定创建不了 fp16
张量。解决：用 Python + onnx 把模型转成 fp32（int4 权重保留，仅把 fp16 常量和 Cast 转 fp32）：

```python
import onnx, numpy as np
from onnx import numpy_helper, TensorProto
m = onnx.load("model.onnx")
for i in m.graph.initializer:
    if i.data_type == TensorProto.FLOAT16:
        i.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(i).astype(np.float32), i.name))
for n in m.graph.node:
    if n.op_type == "Cast":
        for a in n.attribute:
            if a.name == "to" and a.i == TensorProto.FLOAT16: a.i = TensorProto.FLOAT
for v in list(m.graph.input) + list(m.graph.output):
    t = v.type.tensor_type
    if t.elem_type == TensorProto.FLOAT16: t.elem_type = TensorProto.FLOAT
onnx.save(m, "model_fp32.onnx")
```

### 3.2 config.json 的坑
- `eos_token_id` 可能是**数组**（`[1, 130073]`）——用 `IntOrSlice` 兼容 int / 数组。
- `model_type` 必须进白名单（已支持 `qwen2 / qwen3 / llama / mistral / gemma / gpt2 / qwen / minicpm / baichuan / stablelm`），否则直接拒绝。
- `head_dim` 可能缺失——按 `hidden_size / num_attention_heads` 推导。
- MiniCPM5 的 `bos_token_id=0`（`<s>`），llama 家族解码前必须前置 BOS，否则输出乱码。

### 3.3 tokenizer.json 的坑
- merges 可能是 `"a b"` 字符串，也可能是 `["a","b"]` 数组——两种都要兼容。
- **重复 id**：部分 tokenizer 的 vocab 里多个 token 映射到同一个 id，`idToToken` 用 Go map
  会随机覆盖。解决：以 added_tokens 内容为准覆盖。
- **推理标记**：Qwen3/MiniCPM5 用 ` thinking` / ` response`（带前导空格的普通 token，
  special=False），解码时会出现。`<|im_start|>` 等 `special=True` 的 token 解码时跳过。
- **ByteLevel 字母表 + byte_fallback** 必须实现；纯 ASCII 的 `?` 空格会经 ByteLevel 映射成
  `Ġ`（U+0120），否则中英文分词全错。
- 预分词正则含**零宽断言**（`\s+(?!\S)`），Go 标准库 regexp（RE2）不支持，需用
  `github.com/dlclark/regexp2`。
- 文本需 **NFC 规范化**（tokenizer.json 里 `normalizer: NFC`）。

### 3.4 一个容易被渲染误导的坑
` thinking` / ` response` 这类"HTML 标签"字符串在终端/文档渲染里会被当成标签吃掉，
看起来像 ` thinking`/` response`。排查时务必看**字节**（`[]byte(...)`）而不是打印的文本。

## 4. 使用经验

### 4.1 对话模板
- 统一 ChatML：`<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n`
- llama 家族模板前要加 `<s>`（BOS）。
- Qwen3/MiniCPM 是推理模型：思考段（` thinking` 到 ` response`）单独走 `llm:reasoning`
  事件、前端折叠展示；引擎只返回最终回答。

### 4.2 参数建议
- 推理模型温度略低（0.6~0.7），Top-P 0.9，Top-K 40，`use_chat_template` 打开。
- 对话用图片时，先走 PP-OCRv6 提取文字再拼进 prompt（本应用的做法）。

### 4.3 如何加一个新模型
1. 在 HF 找 `onnx-community/*` 或 transformers.js 格式的 ONNX 导出（标准逐层 KV Cache）。
2. 下载 `config.json` + `tokenizer.json` + `model.onnx`（选 int8 优先，其次 fp32；fp16 需先转 fp32）。
3. 放到 `~/.cpm_orc/models/llm/<id>/`。
4. 打开应用「MiniCPM 对话」页 → 下拉选模型 → 加载 → 对话。

## 5. 验证方法
- 分词一致性：与 HuggingFace `AutoTokenizer` 对比若干句子的 token id（本项目实现与 HF 逐 token 一致）。
- 端到端：加载模型后直接 `Generate`，看输出是否为有意义文本（随机权重/格式错误会输出乱码）。
- 单元测试：`go test ./internal/...`（含分词器、OCR 后处理、ONNX 元数据、运行时下载）。

## 6. 关于 Qwen3.5
Qwen3.5 是**混合架构**（Gated DeltaNet + 稀疏 MoE），官方 ONNX 导出为
`decoder_model_merged`（onnxruntime-genai 风格），需要 `past_conv`/`past_recurrent`
状态 + 3D position_ids + `num_logits_to_keep`，且只有 3GB fp32 或 fp16 量化两种文件。
当前引擎（通用纯注意力逐层 KV 循环）不支持；要支持需新增混合状态管理（约 150 行改造），
暂未执行。当前内置模型已覆盖日常需求。