package tex

import "strings"

// Rule: inside \text{...} only plain text is allowed — no LaTeX/math content.
// These files share the detection logic between Validate and Repair.

// textMathChars are characters that can appear inside a math run within a
// \text{} argument. Spaces are handled separately by the tokenizer.
const textMathChars = "=+-*/^_<>&|()[]{},.;:!?\\$"

// textTok is a segment of a \text{} argument.
type textTok struct {
	Math bool   // true if the segment is LaTeX/math content
	S    string // raw segment text
}

func isTextMathChar(b byte) bool {
	if b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' {
		return true
	}
	return strings.ContainsRune(textMathChars, rune(b))
}

// isTextMathRun reports whether a whitespace-free token is math content. A
// bare operator between CJK characters (e.g. the hyphen in 第一问-第二问) is
// punctuation, not math — so an operator alone does not qualify unless the run
// also contains a letter/digit (or a command / $ marker).
func isTextMathRun(s string) bool {
	if strings.ContainsAny(s, "\\$") {
		return true
	}
	if !strings.ContainsAny(s, "=+-*/^_<>&|") {
		return false
	}
	for i := 0; i < len(s); i++ {
		b := s[i]
		if b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' {
			return true
		}
	}
	return false
}

// splitTextContent splits a \text{} argument into alternating text and math
// tokens. A math token is either an existing $...$ / $$...$$ span (dollarSpan)
// or a run of characters containing a math signal (operator / command). Runs
// are split at whitespace; consecutive math chunks are merged back later, while
// pure lowercase prose words are kept as text.
func splitTextContent(s string) []textTok {
	var toks []textTok
	i := 0
	for i < len(s) {
		c := s[i]
		if c == '$' {
			n := dollarSpan(s, i)
			toks = append(toks, textTok{Math: true, S: s[i : i+n]})
			i += n
			continue
		}
		if isTextMathChar(c) {
			j := i
			for j < len(s) && s[j] != '$' && (isTextMathChar(s[j]) || s[j] == ' ') {
				j++
			}
			end := j
			for end > i && s[end-1] == ' ' {
				end--
			}
			run := s[i:end]
			if isTextMathRun(run) {
				if mt := splitMathRun(run); len(mt) > 0 {
					toks = append(toks, mt...)
				} else {
					toks = append(toks, textTok{Math: true, S: run})
				}
			} else if run != "" {
				toks = append(toks, textTok{Math: false, S: run})
			}
			if j > end {
				toks = append(toks, textTok{Math: false, S: s[end:j]})
			}
			i = j
			continue
		}
		j := i
		for j < len(s) && !isTextMathChar(s[j]) && s[j] != '$' {
			j++
		}
		toks = append(toks, textTok{Math: false, S: s[i:j]})
		i = j
	}
	return combineMathChunks(toks)
}

// dollarSpan returns the length of a $...$ / $$...$$ span starting at i, or
// the length to the end of s when the span is unclosed.
func dollarSpan(s string, i int) int {
	if i+1 < len(s) && s[i+1] == '$' {
		if j := strings.Index(s[i+2:], "$$"); j >= 0 {
			return j + 4
		}
		return len(s) - i
	}
	if j := strings.IndexByte(s[i+1:], '$'); j >= 0 {
		return j + 2
	}
	return len(s) - i
}

// mathFuncWords are lowercase math function names that must stay inside math
// mode even though they look like prose words.
var mathFuncWords = map[string]bool{
	"sin": true, "cos": true, "tan": true, "cot": true, "sec": true, "csc": true,
	"arcsin": true, "arccos": true, "arctan": true,
	"log": true, "ln": true, "lim": true, "exp": true,
	"min": true, "max": true, "sum": true, "prod": true, "mod": true,
	"det": true, "gcd": true, "lcm": true, "arg": true, "inf": true, "sup": true,
}

// isProseWord reports whether a whitespace-free chunk is prose text rather than
// math: >=2 letters, containing at least one lowercase letter (so "note",
// "where", "Given" qualify), and not a math function name. All-uppercase words
// (AB, CD ...) are treated as math segment labels, and single letters (x, A)
// stay math variables.
func isProseWord(s string) bool {
	if len(s) < 2 {
		return false
	}
	end := len(s)
	if strings.ContainsRune(".,;:!?", rune(s[end-1])) {
		end--
	}
	if end < 2 {
		return false
	}
	hasLower := false
	for i := 0; i < end; i++ {
		c := s[i]
		if !isAsciiLetter(c) {
			return false
		}
		if c >= 'a' && c <= 'z' {
			hasLower = true
		}
	}
	return hasLower && !mathFuncWords[s[:end]]
}

// splitMathRun splits a math run that contains spaces, pulling prose words out
// as text. It returns nil when the run should stay a single math token.
func splitMathRun(run string) []textTok {
	if !strings.ContainsAny(run, " ") {
		return nil
	}
	chunks := strings.Fields(run)
	if len(chunks) < 2 {
		return nil
	}
	prose := make([]bool, len(chunks))
	hasProse := false
	for i, ch := range chunks {
		prose[i] = isProseWord(ch)
		if prose[i] {
			hasProse = true
		}
	}
	if !hasProse {
		return nil
	}
	var toks []textTok
	for i, ch := range chunks {
		if prose[i] {
			toks = append(toks, textTok{Math: false, S: ch})
		} else {
			toks = append(toks, textTok{Math: true, S: ch})
		}
		if i < len(chunks)-1 {
			toks = append(toks, textTok{Math: false, S: " "})
		}
	}
	return toks
}

// combineMathChunks folds "math space math" sequences into a single math token
// so that e.g. "x > 0" stays one $...$ expression instead of three separate ones.
func combineMathChunks(toks []textTok) []textTok {
	var out []textTok
	for i := 0; i < len(toks); i++ {
		if toks[i].Math && i+2 < len(toks) && toks[i+1].S == " " && toks[i+2].Math {
			var b strings.Builder
			b.WriteString(toks[i].S)
			for i+2 < len(toks) && toks[i+1].S == " " && toks[i+2].Math {
				b.WriteString(" ")
				b.WriteString(toks[i+2].S)
				i += 2
			}
			out = append(out, textTok{Math: true, S: b.String()})
			continue
		}
		out = append(out, toks[i])
	}
	return out
}

// textify rebuilds a \text{} argument: text stays in \text{}, math is moved
// out into inline $...$ math. It reports whether any math was found. Nested
// \text{} spans are left untouched (not a realistic model output).
func textify(content string) (string, bool) {
	if strings.Contains(content, "\\text") {
		return content, false
	}
	toks := splitTextContent(content)
	var merged []textTok
	for _, tk := range toks {
		if len(merged) > 0 && !merged[len(merged)-1].Math && !tk.Math {
			merged[len(merged)-1].S += tk.S
		} else {
			merged = append(merged, tk)
		}
	}
	hasMath := false
	for _, tk := range merged {
		if tk.Math {
			hasMath = true
			break
		}
	}
	if !hasMath {
		return content, false
	}
	var sb strings.Builder
	for _, tk := range merged {
		if tk.Math {
			sb.WriteString(wrapMath(tk.S))
		} else if strings.TrimSpace(tk.S) != "" {
			sb.WriteString("\\text{")
			sb.WriteString(tk.S)
			sb.WriteString("}")
		}
	}
	return sb.String(), true
}

// wrapMath wraps a math token in $...$ / $$...$$ delimiters. Already-delimited
// tokens pass through; a token that opens a delimiter but is missing its
// closing one gets it appended (e.g. unclosed "$x>0" -> "$x>0$").
func wrapMath(s string) string {
	if strings.HasPrefix(s, "$$") {
		if strings.HasSuffix(s, "$$") {
			return s
		}
		return s + "$$"
	}
	if strings.HasPrefix(s, "$") {
		if strings.HasSuffix(s, "$") {
			return s
		}
		return s + "$"
	}
	return "$" + s + "$"
}

func isAsciiLetter(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z'
}

func hasPrefixAt(s string, i int, prefix string) bool {
	if i+len(prefix) > len(s) {
		return false
	}
	return s[i:i+len(prefix)] == prefix
}

// matchBrace returns the index of the '}' matching the '{' at open.
func matchBrace(s string, open int) (int, bool) {
	depth := 0
	for i := open; i < len(s); i++ {
		switch s[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return i, true
			}
		}
	}
	return 0, false
}

// textSpanIndex finds the content of the next \text{...} at or after i.
// It returns the content offsets (start/end of the argument) and the index
// after the closing brace. ok is false when no \text command remains.
func textSpanIndex(latex string, i int) (contentStart, contentEnd, next int, ok bool) {
	if latex[i] != '\\' || !hasPrefixAt(latex, i, "\\text") {
		return 0, 0, i, false
	}
	after := i + 5
	if after < len(latex) && isAsciiLetter(latex[after]) {
		return 0, 0, i + 1, false
	}
	k := after
	for k < len(latex) && (latex[k] == ' ' || latex[k] == '\t') {
		k++
	}
	if k >= len(latex) || latex[k] != '{' {
		return 0, 0, i + 1, false
	}
	end, matched := matchBrace(latex, k)
	if !matched {
		return 0, 0, i + 1, false
	}
	return k + 1, end, end + 1, true
}

// textifyAll scans the whole latex for \text{...} and moves any LaTeX content
// out into inline math. It returns the fixed latex and the number of spans
// that were repaired.
func textifyAll(latex string) (string, int) {
	var sb strings.Builder
	i, fixed := 0, 0
	for i < len(latex) {
		if start, end, next, ok := textSpanIndex(latex, i); ok {
			out, changed := textify(latex[start:end])
			if changed {
				sb.WriteString(out)
				fixed++
				i = next
				continue
			}
		}
		sb.WriteByte(latex[i])
		i++
	}
	return sb.String(), fixed
}

// textHasMath reports whether any \text{...} argument contains LaTeX content.
func textHasMath(latex string) (string, bool) {
	i := 0
	for i < len(latex) {
		if start, end, next, ok := textSpanIndex(latex, i); ok {
			if _, has := textify(latex[start:end]); has {
				return truncate(latex[start:end], 40), true
			}
			i = next
			continue
		}
		i++
	}
	return "", false
}

// stripTextSpans removes every \text{...} span (command + braces + content)
// from latex. Used by checkFormulaContent so that prose inside \text{} does
// not count as a math signal.
func stripTextSpans(latex string) string {
	var sb strings.Builder
	i := 0
	for i < len(latex) {
		if _, _, next, ok := textSpanIndex(latex, i); ok {
			i = next
			continue
		}
		sb.WriteByte(latex[i])
		i++
	}
	return sb.String()
}

// normalizeDollarSpans removes stray $ delimiters nested inside a closed
// $...$ / $$...$$ span (e.g. "$$x = $y$ + 1$$" -> "$$x = y + 1$$") so KaTeX
// does not choke on a '$' in math mode. Unclosed spans are left untouched, and
// an escaped literal dollar (\$) is preserved. It runs after textify so that
// \text{} content has already been extracted into well-formed spans.
func normalizeDollarSpans(latex string) string {
	const esc = "\x00"
	latex = strings.ReplaceAll(latex, `\$`, esc)
	var sb strings.Builder
	i := 0
	for i < len(latex) {
		if latex[i] != '$' {
			sb.WriteByte(latex[i])
			i++
			continue
		}
		if i+1 < len(latex) && latex[i+1] == '$' {
			if j := strings.Index(latex[i+2:], "$$"); j >= 0 {
				content := latex[i+2 : i+2+j]
				sb.WriteString("$$")
				sb.WriteString(strings.ReplaceAll(content, "$", ""))
				sb.WriteString("$$")
				i += j + 4
				continue
			}
			sb.WriteString("$$")
			i += 2
			continue
		}
		if j := strings.IndexByte(latex[i+1:], '$'); j >= 0 {
			content := latex[i+1 : i+1+j]
			sb.WriteString("$")
			sb.WriteString(strings.ReplaceAll(content, "$", ""))
			sb.WriteString("$")
			i += j + 2
			continue
		}
		sb.WriteByte('$')
		i++
	}
	return strings.ReplaceAll(sb.String(), esc, `\$`)
}

func truncate(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "…"
}
