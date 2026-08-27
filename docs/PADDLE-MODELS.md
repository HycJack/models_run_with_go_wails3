# PaddlePaddle 模型全指南（PaddleOCR / PaddleX 家族）

本指南整理 CPM OCR Studio 可用的 PaddlePaddle 系列模型：从 OCR 基础档位
（PP-OCRv6 tiny/small/medium）到方向分类、文档矫正、版面分析、表格、公式、
印章等垂类模型。所有模型均为 HuggingFace `PaddlePaddle/*` 官方仓库，优先使用
**ONNX** 格式（本应用通过 ONNX Runtime 直接加载，不依赖 Python）。

---

## 1. OCR 基础档位：PP-OCRv6 三档对比

PP-OCRv6 是最新一代通用 OCR 模型族，统一骨干 PPLCNetV4，按参数量分三档：

| 档位 | 定位 | 检测参数 | 识别参数 | 总参数 | 语言 | ONNX 仓库 |
| --- | --- | --- | --- | --- | --- | --- |
| **tiny** | 端侧 / IoT | 0.44M | 1.11M | 1.5M | 49 种（不含日文） | `PP-OCRv6_tiny_det/rec_onnx` |
| **small** | 移动端 / 桌面 | 2.48M | 5.29M | ~8M | 50 种 | `PP-OCRv6_small_det/rec_onnx` |
| **medium** | 服务端 | 21.99M | 19.18M | 34.5M | 50 种 | `PP-OCRv6_medium_det/rec_onnx` |

50 种语言 = 简体中文 + 繁体中文 + 英文 + 日文 + 46 种拉丁语系语言，单一模型覆盖，
无需按语言切换模型。

### 1.1 精度（内部多场景基准，检测 Hmean % / 识别准确率 %）

| 档位 | 检测 Hmean | 识别 W-Avg | 说明 |
| --- | --- | --- | --- |
| medium | **86.2** | **83.2** | 超 PP-OCRv5_server +4.6/+5.1，34.5M 参数精度超越 Qwen3-VL-235B、GPT-5.5 |
| small | 84.1 | 81.3 | 速度与 PP-OCRv5_mobile 持平，精度更高 |
| tiny | 80.6 | 73.5 | 最快档，Apple M4 推理比 small 快约 4 倍 |

medium 在日文、古籍、旋转文本、工业字符场景提升尤其显著；tiny 对表情符号/特殊
字符场景表现反而最好。

### 1.2 速度参考（ONNX Runtime CPU，单张）

| 硬件 | medium | small | tiny |
| --- | --- | --- | --- |
| Apple M4 | 5.55s | 1.29s | **0.35s** |
| Intel Xeon 8350C | 3.31s | 0.61s | 0.22s |
| NVIDIA V100 | 0.67s | 0.53s | 0.29s |

### 1.3 文件清单与大小

| 仓库 | 文件 | 大小 | 用途 |
| --- | --- | --- | --- |
| `*_det_onnx` | `inference.onnx` | tiny 1.8MB / small 9.9MB / medium 62MB | 文本检测 |
| `*_rec_onnx` | `inference.onnx` | tiny 4.5MB / small 21MB / medium 76MB | 文字识别 |
| `*_rec_onnx` | `inference.yml` | ~150KB | 含 `character_dict`，用于生成字典 |

**字典注意事项**：
- small 与 medium 共用同一字典（18708 字符，`PP-OCRv6_dict.txt`）。
- **tiny 字典不同**（6904 字符），换用 tiny 时必须用其自身 `inference.yml`
  生成字典，否则解码错乱。

### 1.4 下载

```bash
# medium（本应用模型目录）
cd ~/.cpm_orc/models/paddleocr/ch
curl -L -x http://127.0.0.1:7890 -o PP-OCRv6_medium_det.onnx \
  https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/resolve/main/inference.onnx
curl -L -x http://127.0.0.1:7890 -o PP-OCRv6_medium_rec.onnx \
  https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.onnx

# tiny（如需更快）
curl -L -x http://127.0.0.1:7890 -o PP-OCRv6_tiny_det.onnx \
  https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/main/inference.onnx
curl -L -x http://127.0.0.1:7890 -o PP-OCRv6_tiny_rec.onnx \
  https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.onnx
curl -L -x http://127.0.0.1:7890 -o PP-OCRv6_tiny_dict.txt \
  https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/resolve/main/ppocr_keys_v1.txt
```

---

## 2. 方向分类模型

### 2.1 文档方向分类（本应用已内置）

| 项目 | 值 |
| --- | --- |
| 模型 | PP-LCNet_x1_0_doc_ori |
| 仓库 | `monkt/paddleocr-onnx`（文件 `preprocessing/doc-orientation/PP-LCNet_x1_0_doc_ori.onnx`） |
| 大小 | 6.8MB |
| 功能 | 判断整页文档 0°/90°/180°/270°，识别前自动旋转矫正 |

引擎位置：`internal/paddleocr/ori.go`，对应 `Models.OriPath`。已在
`InstallDefaults` 中随默认模型一起下载。

### 2.2 文本行方向分类（180°）

| 项目 | 值 |
| --- | --- |
| 模型 | PP-LCNet_x1_0_textline_ori |
| 仓库 | `PaddlePaddle/PP-LCNet_x1_0_textline_ori_onnx` |
| 文件 | `inference.onnx` 6.8MB + `inference.yml` |
| 功能 | 判断**单行文字**是否倒置（180°），比文档级更细，适合竖屏/旋转截图 |

注意：该模型归一化用 ImageNet 均值（mean `0.485/0.456/0.406`、std
`0.229/0.224/0.225`），与本应用 det/rec 的 PaddleOCR 归一化
（mean 0.5、std 0.5、scale 1/255）**不同**。接入时需为它单独配置归一化参数。

对应引擎：`internal/paddleocr/cls.go`，`Models.ClsPath`（现引擎仅支持 180° 二分类）。

### 2.3 文本行方向分类（四方向）

| 项目 | 值 |
| --- | --- |
| 模型 | PP-LCNet_x0_25_textline_ori / PP-LCNet_x1_0_textline_ori |
| 仓库 | `PaddlePaddle/PP-LCNet_x1_0_textline_ori_onnx` |
| 说明 | 同一仓库还包含 0/90/180/270° 四方向判定能力（yaml 中标注），用于识别前旋转单行文本 |

---

## 3. 文档矫正：UVDoc

| 项目 | 值 |
| --- | --- |
| 模型 | UVDoc（Unwarping Document） |
| 仓库 | `PaddlePaddle/UVDoc_onnx` |
| 文件 | `inference.onnx` 31.7MB |
| 功能 | 翻拍书籍/弯曲页面的曲面矫正，去透视与卷曲，还原平整文档 |

适用场景：手机拍书、扫描仪弯折纸页。输出一张矫正后的平面文档图，再走 OCR。
可作为 det 之前的预处理步骤接入。

---

## 4. 版面分析（Layout）

| 模型 | 仓库 | ONNX 大小 | 说明 |
| --- | --- | --- | --- |
| PP-DocLayoutV3 | `PaddlePaddle/PP-DocLayoutV3_onnx` | 130.5MB | 最新版面模型，标题/正文/表格/图片/页眉页脚等区域检测 |
| PP-DocLayout_plus-L | `PaddlePaddle/PP-DocLayout_plus-L_onnx` | 129.7MB | 增强版版面分析 |
| PicoDet-S/L layout | `PaddlePaddle/PicoDet-S_layout_3cls_onnx` | — | 轻量版面，3 类（文本/标题/图表） |
| PicoDet_layout_1x_table | `PaddlePaddle/PicoDet_layout_1x_table` | — | 表格区域检测 |

用途：先做版面分析切出区域，再对每个区域做 OCR（避免多栏文档被逐行横切），
或定位表格/图片区域做进一步结构化。接入后需实现版面后处理（阈值 + NMS），
与本应用 det 后处理类似。

---

## 5. 表格识别

| 模型 | 仓库 | ONNX 大小 | 功能 |
| --- | --- | --- | --- |
| RT-DETR-L_wired_table_cell_det | `PaddlePaddle/RT-DETR-L_wired_table_cell_det_onnx` | 129.4MB | 有线表格单元格检测 |
| RT-DETR-L_wireless_table_cell_det | `PaddlePaddle/RT-DETR-L_wireless_table_cell_det_onnx` | 129.4MB | 无线表格单元格检测 |
| PP-LCNet_x1_0_table_cls | `PaddlePaddle/PP-LCNet_x1_0_table_cls_onnx` | 6.8MB | 表格分类（有线/无线/无表） |

典型流程：`PP-LCNet_x1_0_table_cls` 判断表格类型 → 对应 RT-DETR 检测单元格
→ 逐格 OCR → 按行列重建 CSV / Excel。RT-DETR 为检测器（输出框 + 类别），
后处理与 PaddleOCR det 的 DB 后处理不同（需解码 RT-DETR 输出）。

---

## 6. 公式识别

| 项目 | 值 |
| --- | --- |
| 模型 | PP-FormulaNet_plus-L / -M / -S |
| 仓库 | `PaddlePaddle/PP-FormulaNet_plus-M`（**注意**：当前仓库为 Paddle 推理格式 `inference.pdiparams` 617MB，**无官方 ONNX**） |
| 功能 | 公式图片 → LaTeX 序列 |

`PP-FormulaNet_plus-L_onnx` 仓库当前为空壳（仅 README）。**暂无官方 ONNX
导出**，需自行用 Paddle2ONNX 转换，或使用 `safetensors` 版本 + Transformers
后端。接入成本较高，暂不建议在本应用（纯 ONNX）中使用。

---

## 7. 印章检测

| 模型 | 仓库 | 说明 |
| --- | --- | --- |
| PP-OCRv4_server_seal_det | `PaddlePaddle/PP-OCRv4_server_seal_det` | 服务端印章检测 |
| PP-OCRv4_mobile_seal_det | `PaddlePaddle/PP-OCRv4_mobile_seal_det` | 移动端印章检测 |

功能：定位红色印章区域，可配合 OCR 做印章文字提取或盖章文档校验。

---

## 8. 与本应用引擎的兼容性总表

| 模型 | 引擎支持 | 接入难度 | 建议 |
| --- | --- | --- | --- |
| PP-OCRv6_medium det/rec | ✅ 直接可用（已下载） | 无 | 精度优先时选用 |
| PP-OCRv6_tiny det/rec | ✅ 直接可用 | 低（需 tiny 独立字典） | 速度优先时选用 |
| PP-LCNet doc_ori | ✅ 已内置 | 无 | 已随默认安装 |
| PP-LCNet textline_ori | ⚠️ ClsPath 可加载，需调归一化参数 | 中 | 值得接入，改善旋转截图 |
| UVDoc | ❌ 需新增预处理步骤 | 中高 | 拍书场景可选 |
| PP-DocLayoutV3 | ❌ 需新引擎 | 高 | 多栏文档可选 |
| RT-DETR 表格/公式/印章 | ❌ 需新引擎 + 新后处理 | 高 | 按需扩展 |

**引擎加载方式**（`internal/paddleocr/engine.go`）：

```go
eng.Load(paddleocr.Models{
    DetPath:  ".../PP-OCRv6_medium_det.onnx",
    RecPath:  ".../PP-OCRv6_medium_rec.onnx",
    ClsPath:  ".../PP-LCNet_x1_0_textline_ori.onnx", // 可选
    OriPath:  ".../PP-LCNet_x1_0_doc_ori.onnx",      // 可选
    DictPath: ".../PP-OCRv6_dict.txt",
})
```

换用其他档位只需替换 Det/Rec/Dict 路径，识别输入高度引擎会自动从 ONNX 图读取
（v3=32px / v4/v6=48px）。

---

## 9. 补充说明

- **镜像/代理**：HuggingFace 下载需走本机代理（Clash Verge 端口 7890）。
  仓库文件实际重定向到 `us.aws.cdn.hf.co`（Xet CDN），curl 必须加 `-L`。
- **模型格式**：优先选 `*_onnx` 后缀官方仓库；`*_safetensors` 需 PyTorch/
  Transformers 后端；裸名仓库（如 `PP-OCRv6_medium_rec`）是 Paddle 推理格式。
- **许可证**：以上模型均为 Apache-2.0，可自由用于商业项目。
- **实测情况**：本应用已实测 medium det/rec 与 small 共用同一字典
  （18708 字符），可直接切换。