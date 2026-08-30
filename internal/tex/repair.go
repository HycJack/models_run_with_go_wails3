package tex

import (
	"regexp"
	"strings"
)

// RepairResult is the outcome of a deterministic repair pass.
type RepairResult struct {
	Original string   `json:"original"`
	Latex    string   `json:"latex"`
	Changed  bool     `json:"changed"`
	Valid    bool     `json:"valid"`
	Log      []string `json:"log"`
}

var (
	reFracEmpty    = regexp.MustCompile(`\\(?:d?frac|tfrac)\s*\{\s*\}\s*\{\s*\}`)
	reFracEmptyNum = regexp.MustCompile(`\\(?:d?frac|tfrac)\s*\{\s*\}\s*\{([^{}]*)\}`)
	reFracEmptyDen = regexp.MustCompile(`\\(?:d?frac|tfrac)\s*\{([^{}]*)\}\s*\{\s*\}`)
	reSqrtEmpty    = regexp.MustCompile(`\\sqrt(?:\s*\[[^\]]*\])?\s*\{\s*\}`)
	reSubEmpty     = regexp.MustCompile(`_\s*\{\s*\}`)
	reSupEmpty     = regexp.MustCompile(`\^\s*\{\s*\}`)
)

// Repair applies deterministic fixes and returns the repaired LaTeX.
func Repair(latex string) RepairResult {
	res := RepairResult{Original: latex, Latex: latex}
	work := latex

	// ── 1. Balance braces ──────────────────────────────────────────────
	work, braceLogs := fixBraces(work)
	res.Log = append(res.Log, braceLogs...)

	// ── 2. Extract math from \text{} ───────────────────────────────────
	if fixed, n := textifyAll(work); n > 0 {
		work = fixed
		res.Log = append(res.Log, "从 \\text{} 内提取 "+itoa(n)+" 处 LaTeX 内容为行内公式")
	}

	// ── 3. Normalize nested $ inside spans ─────────────────────────────
	if norm := normalizeDollarSpans(work); norm != work {
		work = norm
		res.Log = append(res.Log, "清理 $ 定界符内的嵌套 $ 符号")
	}

	// ── 4. Fix environments (mismatch + extra \end + missing \end) ────
	if fixed, logs := fixEnvironments(work); fixed != work {
		work = fixed
		res.Log = append(res.Log, logs...)
	}

	// ── 5. Remove empty structures ─────────────────────────────────────
	if fixed, n := removeEmptyStructures(work); n > 0 {
		work = fixed
		res.Log = append(res.Log, "移除 "+itoa(n)+" 个空结构")
	}

	// ── 6. Balance \left / \right ──────────────────────────────────────
	if fixed, logs := fixLeftRight(work); len(logs) > 0 {
		work = fixed
		res.Log = append(res.Log, logs...)
	}

	// ── 7. Remove trailing dangling separators (keep \\ line break) ────
	work, sepLogs := trimTrailingSeparators(work)
	res.Log = append(res.Log, sepLogs...)

	// ── 8. Trim trailing dangling operators (x^2+ → x^2) ──────────────
	if fixed := trimDanglingOps(work); fixed != work {
		work = fixed
		res.Log = append(res.Log, "截断末尾悬挂运算符")
	}

	// ── final cleanup ──────────────────────────────────────────────────
	work = strings.TrimSpace(work)

	res.Latex = work
	res.Changed = work != latex
	v := Validate(work)
	res.Valid = v.Valid
	return res
}

func fixBraces(s string) (string, []string) {
	var balanced []byte
	open := 0
	removed := 0
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if ch == '{' {
			open++
			balanced = append(balanced, ch)
		} else if ch == '}' {
			if open > 0 {
				open--
				balanced = append(balanced, ch)
			} else {
				removed++
			}
		} else {
			balanced = append(balanced, ch)
		}
	}
	var logs []string
	work := s
	if removed > 0 {
		work = string(balanced)
		logs = append(logs, "移除 "+itoa(removed)+" 个多余的右花括号")
	}
	if open > 0 {
		work += stringsRepeat("}", open)
		logs = append(logs, "补齐 "+itoa(open)+" 个右花括号")
	}
	return work, logs
}

func fixEnvironments(latex string) (string, []string) {
	type match struct {
		name string
		pos  int
		end  int
	}
	var begins, ends []match
	for _, m := range envRe.FindAllStringSubmatchIndex(latex, -1) {
		name := latex[m[4]:m[5]]
		if latex[m[2]:m[3]] == "begin" {
			begins = append(begins, match{name, m[0], m[1]})
		} else {
			ends = append(ends, match{name, m[0], m[1]})
		}
	}

	type marker struct {
		isBegin  bool
		name     string
		pos, end int
	}
	var markers []marker
	for _, b := range begins {
		markers = append(markers, marker{true, b.name, b.pos, b.end})
	}
	for _, e := range ends {
		markers = append(markers, marker{false, e.name, e.pos, e.end})
	}
	for i := 1; i < len(markers); i++ {
		for j := i; j > 0 && markers[j].pos < markers[j-1].pos; j-- {
			markers[j], markers[j-1] = markers[j-1], markers[j]
		}
	}

	stack := []string{}
	var logs []string
	// Build result by scanning markers in order.
	var out []byte
	lastIdx := 0
	for _, mk := range markers {
		// Copy text before this marker.
		out = append(out, latex[lastIdx:mk.pos]...)
		lastIdx = mk.pos

		if mk.isBegin {
			stack = append(stack, mk.name)
			out = append(out, latex[mk.pos:mk.end]...)
			lastIdx = mk.end
		} else if len(stack) == 0 {
			logs = append(logs, "移除孤立的 \\end{"+mk.name+"}")
			lastIdx = mk.end
		} else if stack[len(stack)-1] == mk.name {
			stack = stack[:len(stack)-1]
			out = append(out, latex[mk.pos:mk.end]...)
			lastIdx = mk.end
		} else {
			top := stack[len(stack)-1]
			out = append(out, "\\end{"+top+"}"...)
			stack = stack[:len(stack)-1]
			logs = append(logs, "修正 \\end{"+mk.name+"} → \\end{"+top+"}")
			lastIdx = mk.end
		}
	}
	// Copy remaining text after last marker.
	out = append(out, latex[lastIdx:]...)

	if len(stack) > 0 {
		for i := len(stack) - 1; i >= 0; i-- {
			out = append(out, "\\end{"+stack[i]+"}"...)
		}
		logs = append(logs, "补齐 "+itoa(len(stack))+" 个 \\end{...}")
	}

	return string(out), logs
}

func removeEmptyStructures(s string) (string, int) {
	count := 0
	for _, re := range []*regexp.Regexp{reFracEmpty, reFracEmptyNum, reFracEmptyDen, reSqrtEmpty, reSubEmpty, reSupEmpty} {
		count += len(re.FindAllString(s, -1))
	}
	if count == 0 {
		return s, 0
	}
	s = reFracEmpty.ReplaceAllString(s, "")
	s = reFracEmptyNum.ReplaceAllString(s, "$1")
	s = reFracEmptyDen.ReplaceAllString(s, "$1")
	s = reSqrtEmpty.ReplaceAllString(s, "")
	s = reSubEmpty.ReplaceAllString(s, "")
	s = reSupEmpty.ReplaceAllString(s, "")
	return s, count
}

func trimDanglingOps(s string) string {
	// Trim trailing operators: x^2+ → x^2, x = → x =
	reOp := regexp.MustCompile(`[=+\-*/<>&|]\s*$`)
	orig := s
	s = reOp.ReplaceAllString(s, "")

	// Trim trailing ^ or _ with no content after them (x^ → x, x_ → x).
	trimmed := []byte(s)
	i := len(trimmed)
	for i > 0 && trimmed[i-1] == ' ' {
		i--
	}
	if i > 0 && (trimmed[i-1] == '^' || trimmed[i-1] == '_') {
		rest := string(trimmed[i:])
		if !strings.HasPrefix(rest, "{") {
			trimmed = trimmed[:i-1]
			s = string(trimmed)
		}
	}

	s = strings.TrimRight(s, " ")
	if s == orig {
		return orig
	}
	return s
}

func fixLeftRight(s string) (string, []string) {
	lc := len(leftRightLeftRe.FindAllString(s, -1))
	rc := len(leftRightRightRe.FindAllString(s, -1))
	if lc == rc {
		return s, nil
	}
	var logs []string
	work := s
	if lc > rc {
		n := lc - rc
		for i := 0; i < n; i++ {
			work += "\\right."
		}
		logs = append(logs, "补齐 "+itoa(n)+" 个 \\right.")
	} else {
		n := rc - lc
		for i := 0; i < n; i++ {
			idx := strings.LastIndex(work, "\\right")
			if idx >= 0 {
				end := idx + 6
				if end < len(work) && (work[end] == '.' || work[end] == ')' || work[end] == ']' || work[end] == '}' || work[end] == '|') {
					end++
				}
				work = work[:idx] + work[end:]
			}
		}
		logs = append(logs, "移除 "+itoa(n)+" 个多余的 \\right")
	}
	return work, logs
}

func trimTrailingSeparators(s string) (string, []string) {
	var logs []string
	for strings.HasSuffix(s, "\\") || strings.HasSuffix(s, "$ ") {
		if strings.HasSuffix(s, "\\") {
			if strings.HasSuffix(s, "\\\\") {
				break
			}
			s = s[:len(s)-1]
			logs = append(logs, "移除末尾孤立的反斜杠")
		} else {
			s = strings.TrimSuffix(s, "$ ")
		}
	}
	return s, logs
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
