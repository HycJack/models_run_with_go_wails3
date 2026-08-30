# LaTeX 修复方案说明（内部 tex 包）

本文详细说明「公式助手」中 LaTeX 的生成 → 清洗 → 校验 → 修复 → 渲染整条链路，
重点讲解 `internal/tex` 包的**确定性修复（Repair）**与**校验（Validate）**的设计与实现，
并记录代码审核中发现的问题与改进建议。

## 1. 背景

小型本地模型（MiniCPM5-1B / Qwen3-0.6B 等）输出 LaTeX 时经常出现：

- 花括号 / `\begin{}` 环境不闭合
- 把数学公式错误地包进 `\text{...}`（KaTeX 中 `\text{}` 内的 `\frac`、`^`、`=` 等不生效）
- `\text{}` 内混入 `$...$`、运算符、命令
- 输出带 markdown 代码块、`\[ \]` / `$$` 包裹或前后散文

目标：**不引入第二个模型**，用确定性的字符串/栈工具把上述问题修到「能渲染 / 基本正确」，
并对修复结果给出结构化校验信息，交给用户或前端展示。

## 2. 整体管线

```
模型输出
  │
  ▼
cleanLatex / cleanProblemLatex   # 去代码块、取最后的 $..$ 跨度、去散文
  │
  ▼
tex.Validate(...)                # 7 项结构/启发式检查 → Result{Valid, Checks}
  │
  ▼
tex.Repair(...)                  # 确定性修复 → RepairResult{Original, Latex, Changed, Valid, Log}
  │
  ▼
前端 renderProblem / renderLatex # KaTeX 渲染（含 \text{}/SVG/配图的后处理）
```

各环节所在文件：

| 环节 | 文件 |
| --- | --- |
| 单公式清洗 `cleanLatex` / `tidy` / `lastSpan` | `internal/app/math_service.go` |
| 整题清洗 `cleanProblemLatex` | `internal/app/ollama_service.go` |
| 校验 `Validate` | `internal/tex/validator.go` |
| 修复 `Repair` | `internal/tex/repair.go` |
| `\text{}` 内容提取算法 | `internal/tex/text.go` |
| 前端渲染 | `frontend/src/pages/MathPage.tsx` |

## 3. 校验（Validate）

`Validate(latex)` 返回 `Result{Valid, Checks}`，`Valid` 为所有 `Checks` 都 OK 才为真。
共 7 项检查（`validator.go`）：

1. **brace** `checkBraceBalance` — 逐字节数 `{`/`}`，`}` 多于 `{` 或最终未闭合都判失败。
2. **env** `checkEnvBalance` — 用栈匹配 `\begin{...}` / `\end{...}`（合法的嵌套匹配实现）。
3. **dollar** `checkDollarBalance` — 逐字符计数 `$`（跳过转义 `\$`），奇数个 `$` 判为未闭合。
4. **leftright** `checkLeftRightBalance` — 计数 `\left` 与 `\right`，不配对则失败。
5. **content** `checkFormulaContent` — 启发式：出现数学信号（`\`、`$`、运算符、数字、括号）
   判为公式；否则把 LaTeX 命令名去掉后数英文单词，≥3 个判为「普通文字而非公式」
   （避免 `\text`、`\frac` 等命令名被误计为英文单词）。
6. **structure** `checkEmptyStructures` — 正则探测空结构：`\frac{}{}`、`\sqrt{}`、`_{}`、`^{}`；
   另有 `danglingOpRe` 检测尾随悬挂运算符/上下标（`x^2+`、`x^`、`x_`、`x > `）。
7. **text** `checkTextContent` — 校验 `\text{...}` 内**只能有纯文本**，含 LaTeX/数学内容即失败
   （与修复步骤 2 共享 `text.go` 的判定逻辑）。

> 注意：`checkEnvBalance` 是唯一正确的环境匹配实现（栈 + 嵌套 + 顺序）。
> `Repair` 中的环境补全没有复用这套逻辑，是当前一个 bug（见 §6.1）。

## 4. 修复（Repair）

`Repair(latex)` 是一趟**本地、确定、无模型**的修复，按顺序执行 8 步，并记录修改日志：

```
work = latex
① 花括号配平   —— 删除多余的 '}' + 补尾部缺的 '}'
② \text 内容提取 —— textifyAll：把 \text{} 里的数学内容挪出来成行内公式
③ $ 定界符清理 —— normalizeDollarSpans：去掉 $...$ / $$...$$ 内嵌套的 $
④ 环境修复     —— fixEnvironments：改错配 \end{b}→\end{a}、删孤立 \end{}、补缺失 \end{}
⑤ 空结构删除   —— \frac{}{}、\sqrt{}、_{}、^{}、\frac{}{x}→x、\frac{x}{}→x
⑥ \left/\right 配平 —— 补缺失的 \right. 或删多余的 \right
⑦ 末尾清理     —— 去掉孤立 '\' 或 '$ '（保留 \\ 换行）+ 截断悬挂运算符 (x^2+→x^2)
⑧ TrimSpace    —— 去首尾空白
```

每一步都是可解释的：`RepairResult.Log` 会记录「补齐 N 个右花括号」「从 \text{} 内提取 N 处
LaTeX 内容为行内公式」等，前端「自动修复」按钮点击后可在 toast / 校验区看到结果。

### 4.1 核心：\text{} 内容提取（text.go）

这是本方案最核心的一步。规则：**`\text{}` 参数里只允许纯文本**，数学内容必须提取到 `$...$`。

`text.go` 提供：

- `textSpanIndex(latex, i)` — 从 `i` 开始找下一个 `\text{...}` 的参数区间（跳过 `\textcolor`
  等 `\text` 前缀命令），返回内容起止、`}` 后位置。
- `splitTextContent(s)` — 把 `\text{}` 参数切成交替的「文本 / 数学」token：
  - 遇到 `$...$`：整体算一个数学 token；
  - 遇到数学字符（ASCII 字母/数字/运算符/括号/`\`/`$`）：吞掉后续连续数学字符**及空格**，
    得到一个「run」，再经 `isTextMathRun` 判定是否真是数学；
  - 其余（中文等非数学字节）：作为文本 token。
- `isTextMathRun(s)` — run 是否为数学：
  - 含 `\` 或 `$` → 数学；
  - 不含运算符 `=+-*/^_<>&|` → 文本；
  - 含运算符**且**含字母/数字 → 数学；
  - 否则（如中文之间单独的 `-`，例「第一问-第二问」）→ 文本。
- `textify(content)` — 把参数重排：文本留在 `\text{...}`，数学移出为 `$...$`；
  若整个参数只有数学则直接输出 `$...$`，不带 `\text{}`。
- `textifyAll(latex)` — 全串扫描，对每个 `\text{...}` 调用 `textify`。
- `textHasMath(latex)` — 供 `checkTextContent` 复用，报告第一个含 LaTeX 的 `\text{}` 片段。

**设计取舍**：

- 「`\text{}` 只能含纯文本」是 KaTeX / LaTeX 渲染的现实约束；把数学挪出 `\text{}` 是**无损、可逆、
  确定性**的，比让模型重出更可靠。
- 单个运算符不构成数学（中文连字符不是公式），避免把「第一问-第二问」误提取。
- 嵌套 `\text` 直接放弃处理（不是现实中的模型输出形态）。

## 5. 生成端清洗（cleanLatex / cleanProblemLatex）

- `cleanLatex`（`math_service.go`）：用于**单公式**。去掉 markdown 代码块；再按
  `\[ \]` → `$$` → `\( \)` → `$` 顺序取**最后**一个完整跨度（小模型常把真正公式放在结尾）；
  最后 `tidy` 去掉包裹分隔符并折叠空白。
- `cleanProblemLatex`（`ollama_service.go`）：用于**整题**。只剥 markdown 代码块与
  开头 ` ```latex ` 等，保留 `\text{}` / `$` / 环境等结构（整题不能只取最后一个跨度）。

## 6. 代码审核发现的问题

> 状态标记：✅ 已修复（2026-08-30） / ⚠️ 已知限制（未改，避免误伤）
> 以下问题均在修复前复现验证过（`internal/tex` 原有单测未覆盖到）。

### 6.1 ✅ [Bug] Repair 环境配平会把已闭合环境再加一个 \end{}（repair.go）

原实现：

```go
var stack []string
for _, m := range envRepairRe.FindAllStringSubmatch(work, -1) {
	stack = append(stack, m[2])          // 只统计 \begin，不看是否已 \end
}
if len(stack) > 0 {
	for i := len(stack) - 1; i >= 0; i-- {
		work += "\\end{" + stack[i] + "}" // 无条件补 end
	}
}
```

- 复现：`Repair(\begin{aligned} a+b \end{aligned})`
  → `\begin{aligned} a+b \end{aligned}\end{aligned}`，`Valid=false`（多出一个 end）。
- 根因：没有复用 `checkEnvBalance` 的**栈匹配**逻辑，只是「每个 begin 补一个 end」。
- **修复**：改为栈匹配（`\begin` 入栈、匹配的 `\end` 出栈），只对末尾仍未闭合的环境按逆嵌套序补 `\end{}`。

### 6.2 ✅ [Bug] \text{} 内未闭合 `$...$` 会产出 `$$...$` 垃圾（text.go）

- 复现：`Repair(\text{当 $x>0)` → `\text{当 }$$x>0$`
  （`splitTextContent` 遇到无配对 `$` 时把 `s[i:]` 整个当 token，`textify` 因不以 `$` 结尾又包了一层）。
- **修复**：新增 `dollarSpan` 统一识别 `$...$` / `$$...$$`（含未闭合情况），
  `textify` 改用 `wrapMath`：已带定界符的原样输出，只缺闭合定界符的补一个尾 `$`/`$$`。
  现在 `\text{当 $x>0` → `\text{当 }$x>0$`，不再有三美元垃圾。

### 6.3 ✅ [Bug] \text{} 内 `$...$` 前面紧跟字母时会被整体吞成一个数学 run（text.go）

- 复现：`Repair(\text{a $x$ b})` → `$a $x$ b$`（run 收集把 `$` 当普通数学字符吞进去）。
- 对照：`\text{已知 $x^2$}` 正常，是因为「已知」是非数学字符才让 `$` 成为 token 起点。
- **修复**：run 收集循环遇到 `$` 即停止（`s[j] != '$'`），`$` 分支永远接管，
  现在 `\text{a $x$ b}` → `\text{a }$x$\text{ b}`。

### 6.4 ✅ [质量] 英文单词会与相邻数学连成一个 run 被整段数学化（text.go）

- 复现：`Repair(\text{note 3x+1})` → `$note 3x+1$`（"note" 被并进数学）。
- **修复**：新增 `splitMathRun` + `isProseWord` + `combineMathChunks`：
  - 含空格的数学 run 先按空白切块；
  - **全小写 ≥2 字母的单词**（非数学函数名，如 sin/cos/tan/log/lim…）判为散文，移出为文本；
    大写单词（AB、CD… 线段标签）与单字母（x、A 变量）保持数学；
  - 相邻数学块重新合并为一个 `$...$`（`x > 0` 不会拆成三个公式）。
  - 例：`note 3x+1` → `\text{note }$3x+1$`；`已知 AB = CD` → `\text{已知 }$AB = CD$`；
    `已知 tan A = 3` → `\text{已知 }$tan A = 3$`。
  - 判定口径：含至少一个小写字母的 ≥2 字母词（`note`/`where`/`Given`）判为散文；
    **全大写**词（AB、CD… 线段标签）保持数学；单字母（x、A）保持数学变量。
    数学函数名（sin/cos/tan/log/lim…）即使小写也保持数学。

### 6.5 ✅ [一致性] content 检查被花括号绕过（validator.go）

`mathSignalRe` 含 `{`、`}` 与 `\`，所以 `\text{hello this is some prose words}` 这类纯散文
也能通过 content 检查（因为带 `\text{}`），而不带花括号的 `hello this is some prose words`
却判为「普通文字」。`content` 与 `text` 两项检查口径不一致。
**修复**：`checkFormulaContent` 先用新增的 `stripTextSpans` 剥掉所有 `\text{...}` 跨度，
再判数学信号——`\text{}` 内的散文不再充当公式信号；单词检测仍基于完整输入，
使 `\text{hello this is some prose words}` 与裸散文一样被判为「普通文字」。
现在 `\text{已知 }$x>0$\text{ 时}`（剥离后剩 `$x>0$`）正常通过。

### 6.6 ✅ [一致性/顺序] 修复顺序问题

环境配平（旧②）发生在 `\text{}` 提取（旧③）之前，且旧②本身有 6.1 的 bug。
**修复**：调整为 ① 花括号 → ② `\text{}` 提取 → ③ 环境配平 → ④ 尾部清理，
避免 `\text{}` 内的 `\begin{}` 被误计数。

### 6.7 ✅ [前端/安全] 模型生成的 HTML/SVG 直接进 dangerouslySetInnerHTML

`MathPage.tsx` 用 `dangerouslySetInnerHTML` 渲染 `renderProblem` 结果，其中包含模型生成的
`<svg>`、`<img src="data:...">`（`recognizeFigures` 拼进 LaTeX 串），`tryGenerateSVG` 只做
`<svg>...</svg>` 截取、未做任何 sanitize。
**修复**：`renderProblem` 输出前统一过 `sanitizeHtml`（去 `<script>`、`on*` 事件属性、
`javascript:` 链接）。本地模型风险低，作为纵深防御。

### 6.8 ✅ [小] 其它

- `Repair` 不再误删末尾 `\\` 换行（仅剥孤立单个 `\`）。
- `structure` 检查新增**尾随悬挂运算符/上下标**检测（`x^2+`、`x^`、`x_`、`x > ` 判为结构错误，
  `danglingOpRe`），`x^2+` 不再静默判合法（仅提示，不做自动修复）。

### 6.9 ✅ [Bug] $$...$$ 内嵌套单 $ 无法渲染（KaTeX 报 "Can't use function '$' in math mode"）

- 复现：`$$x = $y$ + 1$$` → KaTeX 抛错、前端显示红色错误。
- 根因：模型常在显示公式里再套单 `$`；KaTeX 数学模式下 `$` 不是合法字符。
- **后端修复**（`repair.go` 新增步骤 ③）：`normalizeDollarSpans` 在 `\text{}` 提取后运行，
  只处理**闭合**的 `$...$`/`$$...$$` 跨度，剥掉内部嵌套 `$`（`$$x = $y$ + 1$$` → `$$x = y + 1$$`），
  未闭合跨度与转义 `\$` 原样保留。
- **前端修复**（`MathPage.tsx` `renderLatex`）：渲染前把段内剩余 `$` 剥掉（`\$` 转义保留），
  兜底保证任何形态都能渲染。

## 7. 测试现状

`internal/tex/tex_test.go` 覆盖：

- 合法/非法基础用例（花括号不平衡）
- `Repair` 补花括号
- 散文判定（`TestValidateProse`）
- `\text{}` 校验（`TestValidateTextContent`）与修复（`TestRepairTextContent`）
- 环境配平：已闭合环境不变、未闭合环境补 `\end{}`（`TestRepairBalancedEnv`）
- `$` 定界符：`$...$`/`$$...$$`、未闭合 `$`、`\text{a $x$ b}`（`TestRepairDollarSpans`）
- 散文词拆分：`note 3x+1`、`where x > 0`、`Given x > 0`、`AB = CD`、`tan A = 3`（`TestRepairProseWords`）
- 尾部 `\\` 换行保留、孤立 `\` 移除（`TestRepairTrailingBreak`）
- content 口径一致性：`\text{}` 内纯散文判为「非公式」（`TestValidateContentConsistency`）
- 尾随悬挂运算符/上下标检测（`TestValidateDanglingOperator`）
- `$` 定界符配对检测：未闭合 `$` 正确检出、转义 `\$` 不误计（`TestValidateDollarBalance`）
- `$$...$$` 内嵌套 `$` 清理、未闭合 `$`/`$$` 保留、转义 `\$` 保留（`TestRepairNestedDollars`）
- 多 `\text{}` 跨度时命令名 "text" 不被误计为英文单词（`TestValidateContentConsistency` 数学用例）
- 多余 `}` 删除：`\frac{a}{b}}` → `\frac{a}{b}`、`}x = 1` → `x = 1`（`TestRepairExtraBraces`）
- 空结构修复：`\frac{}{x}` → `x`、`\frac{x}{}` → `x`、`\sqrt{}` → 删除、`_{}`/`^{}` → 删除（`TestRepairEmptyStructures`）
- 环境错配修复：`\begin{aligned} x \end{cases}` → `\end{aligned}`（`TestRepairEnvMismatch`）
- 多余 `\end{}` 删除：无对应 `\begin{}` 的 `\end{}` 被移除（`TestRepairExtraEnd`）
- `\left`/`\right` 配对修复：`\left( x` → 补 `\right.`、`\left( x \right] \left(` → 补 `\right.`（`TestRepairLeftRight`）
- 悬挂运算符截断：`x^2 +` → `x^2`、`x^` → `x`、`x_` → `x`（`TestRepairDanglingOps`）

> 附：用随机输入（含新增的 `\frac{}{`、`\sqrt{`、`\begin{`、`\left(` 等种子）对 `Repair`/`Validate` 做了 3 万次模糊测试，未发现 panic。

## 8. 后续改进建议（按优先级）

§6 审核发现的问题已全部修复。可选的低优先级方向：

- 顶层未闭合 `$`：validator 已能检出（dollar FAIL），repair 不自动闭合（避免吞内容风险），
  用户可手动补充或依赖前端 `renderLatex` 剥离 `$` 兜底渲染。
- CJK 纯文本（无 `[A-Za-z]` 词）无法通过 wordRe 检出为散文——这是启发式方法的已知边界，
  实际模型输出极少出现纯 CJK 无任何数学符号的情况。
- 更细的英文数学环境（`\begin{aligned}` 内 `&&`/`\\` 对齐结构的语义校验）
- 空结构 / 悬挂运算符之外的更多结构错误（如孤立的 `\end`、`\text{}` 嵌套）检测
- 前端 `sanitizeHtml` 后续可换用更完整的白名单清洗（DOMPurify）

## 9. 快速验证

```bash
go test ./internal/tex/ -v        # 单元测试（含修复回归用例）
go vet ./internal/tex/            # 静态检查
cd frontend && npm run build      # 前端（含 sanitizeHtml 变更）构建
```
