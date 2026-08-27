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
	mathSignalRe = regexp.MustCompile(`[\\=+\-*/^_<>&|()[\]{}]|\d`)
	wordRe       = regexp.MustCompile(`[A-Za-z]{2,}`)
)

// Validate checks a LaTeX formula structurally and heuristically.
func Validate(latex string) Result {
	if strings.TrimSpace(latex) == "" {
		return Result{Valid: false, Checks: []Check{{OK: false, Type: "empty", Detail: "公式为空"}}}
	}
	checks := []Check{
		checkBraceBalance(latex),
		checkEnvBalance(latex),
		checkFormulaContent(latex),
		checkEmptyStructures(latex),
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
	if mathSignalRe.MatchString(latex) {
		return Check{OK: true, Type: "content", Detail: "包含公式内容"}
	}
	words := wordRe.FindAllString(latex, -1)
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
	return Check{OK: true, Type: "structure", Detail: "结构完整"}
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