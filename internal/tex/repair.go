package tex

import (
	"regexp"
	"strings"
)

// RepairResult is the outcome of a deterministic repair pass.
type RepairResult struct {
	Original string `json:"original"`
	Latex    string `json:"latex"`
	Changed  bool   `json:"changed"`
	Valid    bool   `json:"valid"`
	Log      []string `json:"log"`
}

var envRepairRe = regexp.MustCompile(`\\(begin)\{([^{}]+)\}`)

// Repair applies deterministic fixes (close unbalanced braces/environments)
// and returns the repaired LaTeX. It is intentionally a small, local, syntax
// tool — no model involved.
func Repair(latex string) RepairResult {
	res := RepairResult{Original: latex, Latex: latex}
	work := latex

	// 1. Balance braces.
	open := 0
	for _, ch := range work {
		if ch == '{' {
			open++
		} else if ch == '}' {
			if open > 0 {
				open--
			}
		}
	}
	if open > 0 {
		work += stringsRepeat("}", open)
		res.Log = append(res.Log, "补齐 " + itoa(open) + " 个右花括号")
	}

	// 2. Balance environments.
	var stack []string
	for _, m := range envRepairRe.FindAllStringSubmatch(work, -1) {
		stack = append(stack, m[2])
	}
	if len(stack) > 0 {
		for i := len(stack) - 1; i >= 0; i-- {
			work += "\\end{" + stack[i] + "}"
		}
		res.Log = append(res.Log, "补齐 " + itoa(len(stack)) + " 个 \\end{...}")
	}

	// 3. Remove trailing dangling command separators.
	for strings.HasSuffix(work, "\\") || strings.HasSuffix(work, "$ ") {
		if strings.HasSuffix(work, "\\") {
			work = work[:len(work)-1]
			res.Log = append(res.Log, "移除末尾孤立的反斜杠")
		} else {
			work = strings.TrimSuffix(work, "$ ")
		}
	}

	res.Latex = work
	res.Changed = work != latex
	v := Validate(work)
	res.Valid = v.Valid
	return res
}

func stringsRepeat(s string, n int) string {
	if n <= 0 {
		return ""
	}
	out := make([]byte, 0, len(s)*n)
	for i := 0; i < n; i++ {
		out = append(out, s...)
	}
	return string(out)
}