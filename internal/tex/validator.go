package tex

import (
	"regexp"
	"strings"
)

// Check is a single validation check result.
type Check struct {
	OK     bool   `json:"ok"`
	Type   string `json:"type"`
	Detail string `json:"detail"`
}

// Result is the full validation output.
type Result struct {
	Valid  bool    `json:"valid"`
	Checks []Check `json:"checks"`
}

var (
	envRe        = regexp.MustCompile(`\\(begin|end)\{([^{}]+)\}`)
	emptyFracRe  = regexp.MustCompile(`\\(?:d?frac|tfrac)\s*\{\s*\}\s*\{`)
	emptyFracDen = regexp.MustCompile(`\\(?:d?frac|tfrac)\s*\{[^{}]*\}\s*\{\s*\}`)
	emptySqrtRe  = regexp.MustCompile(`\\sqrt(?:\s*\[[^\]]*\])?\s*\{\s*\}`)
	emptySubRe   = regexp.MustCompile(`_\s*\{\s*\}`)
	emptySupRe   = regexp.MustCompile(`\^\s*\{\s*\}`)
	mathSignalRe = regexp.MustCompile(`[\\$=+\-*/^_<>&|()[\]{}]|\d`)
	wordRe       = regexp.MustCompile(`[A-Za-z]{2,}`)
	cmdRe        = regexp.MustCompile(`\\[A-Za-z]+`)
	danglingOpRe = regexp.MustCompile(`(?:\^|_|[=+\-*/<>&|])\s*$`)
)

// Validate checks a LaTeX formula structurally and heuristically.
func Validate(latex string) Result {
	if strings.TrimSpace(latex) == "" {
		return Result{Valid: false, Checks: []Check{{OK: false, Type: "empty", Detail: "公式为空"}}}
	}
	checks := []Check{
		checkBraceBalance(latex),
		checkEnvBalance(latex),
		checkDollarBalance(latex),
		checkLeftRightBalance(latex),
		checkFormulaContent(latex),
		checkEmptyStructures(latex),
		checkTextContent(latex),
	}
	valid := true
	for _, c := range checks {
		if !c.OK {
			valid = false
		}
	}
	return Result{Valid: valid, Checks: checks}
}

func checkBraceBalance(latex string) Check {
	depth := 0
	firstBad := -1
	for i, ch := range latex {
		switch ch {
		case '{':
			depth++
		case '}':
			depth--
			if depth < 0 && firstBad < 0 {
				firstBad = i
			}
		}
	}
	if firstBad >= 0 {
		return Check{OK: false, Type: "brace", Detail: "存在多余的右花括号 }"}
	}
	if depth > 0 {
		return Check{OK: false, Type: "brace", Detail: "存在未闭合的花括号（缺 " + itoa(depth) + " 个 }）"}
	}
	return Check{OK: true, Type: "brace", Detail: "花括号平衡"}
}

func checkDollarBalance(latex string) Check {
	count := 0
	i := 0
	for i < len(latex) {
		if latex[i] == '\\' && i+1 < len(latex) {
			i += 2 // skip \$ and any other escaped char
			continue
		}
		if latex[i] == '$' {
			count++
		}
		i++
	}
	if count%2 != 0 {
		return Check{OK: false, Type: "dollar", Detail: "存在未闭合的 $ 定界符"}
	}
	return Check{OK: true, Type: "dollar", Detail: "$ 定界符配对"}
}

var (
	leftRightLeftRe  = regexp.MustCompile(`\\left[\(\[\{\\.|]`)
	leftRightRightRe = regexp.MustCompile(`\\right[\)\]\}\\.|]`)
)

func checkLeftRightBalance(latex string) Check {
	lc := len(leftRightLeftRe.FindAllString(latex, -1))
	rc := len(leftRightRightRe.FindAllString(latex, -1))
	if lc > rc {
		return Check{OK: false, Type: "leftright", Detail: "缺少 " + itoa(lc-rc) + " 个 \\right"}
	}
	if rc > lc {
		return Check{OK: false, Type: "leftright", Detail: "多余的 \\right（多 " + itoa(rc-lc) + " 个）"}
	}
	return Check{OK: true, Type: "leftright", Detail: "\\left/\\right 配对"}
}

func checkEnvBalance(latex string) Check {
	stack := []string{}
	bad := false
	detail := ""
	for _, m := range envRe.FindAllStringSubmatch(latex, -1) {
		kind, name := m[1], m[2]
		if kind == "begin" {
			stack = append(stack, name)
		} else {
			if len(stack) == 0 {
				bad = true
				detail = "多余的 \\end{" + name + "}"
				break
			}
			if stack[len(stack)-1] != name {
				bad = true
				detail = "环境不匹配：\\begin{" + stack[len(stack)-1] + "} 需要 \\end{" + stack[len(stack)-1] + "}"
				break
			}
			stack = stack[:len(stack)-1]
		}
	}
	if bad {
		return Check{OK: false, Type: "env", Detail: detail}
	}
	if len(stack) > 0 {
		return Check{OK: false, Type: "env", Detail: "环境 \\begin{" + stack[len(stack)-1] + "} 未闭合"}
	}
	return Check{OK: true, Type: "env", Detail: "环境平衡"}
}

func checkFormulaContent(latex string) Check {
	// Prose inside \text{...} must not count as a math signal (braces and the
	// \text command would otherwise make any \text{}-wrapped prose pass this
	// check). Strip the spans, then look for real math elsewhere.
	if mathSignalRe.MatchString(stripTextSpans(latex)) {
		return Check{OK: true, Type: "content", Detail: "包含公式内容"}
	}
	// Count prose words on the input with LaTeX commands removed, so command
	// names (\text, \frac ...) are not mistaken for English words.
	words := wordRe.FindAllString(cmdRe.ReplaceAllString(latex, ""), -1)
	if len(words) >= 3 {
		return Check{OK: false, Type: "content", Detail: "看起来像普通文字而非公式"}
	}
	return Check{OK: true, Type: "content", Detail: "包含公式内容"}
}

func checkEmptyStructures(latex string) Check {
	for _, re := range []*regexp.Regexp{emptyFracRe, emptyFracDen, emptySqrtRe, emptySubRe, emptySupRe} {
		if re.MatchString(latex) {
			return Check{OK: false, Type: "structure", Detail: "存在空结构（如 \\frac{}{} / \\sqrt{} / _{}/^{}）"}
		}
	}
	if danglingOpRe.MatchString(latex) {
		return Check{OK: false, Type: "structure", Detail: "末尾存在悬挂的运算符/上下标（如 x^2+ / x^）"}
	}
	return Check{OK: true, Type: "structure", Detail: "结构完整"}
}

// checkTextContent verifies that \text{...} arguments contain only plain text,
// not LaTeX/math content (formulas must live in $...$, not inside \text{}).
func checkTextContent(latex string) Check {
	if bad, found := textHasMath(latex); found {
		return Check{OK: false, Type: "text", Detail: "\\text{} 内不应包含 LaTeX 内容：" + bad}
	}
	return Check{OK: true, Type: "text", Detail: "\\text{} 内仅包含文本"}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
