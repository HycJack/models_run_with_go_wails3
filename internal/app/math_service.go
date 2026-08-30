package app

import (
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"cpm_orc/internal/llm"
	"cpm_orc/internal/paddleocr"
	"cpm_orc/internal/tex"
)

// MathService is a local LaTeX copilot: natural language / formula image to
// LaTeX, plus deterministic validation and repair.
type MathService struct {
	state *State
}

// NewMathService creates the math assistant service.
func NewMathService(s *State) *MathService { return &MathService{state: s} }

var codeFenceRe = regexp.MustCompile("(?s)```[a-zA-Z]*\\s*(.*?)```")

// cleanLatex extracts the pure LaTeX from a model response, stripping markdown
// fences, display/environment delimiters and any surrounding prose.
func cleanLatex(s string) string {
	if m := codeFenceRe.FindStringSubmatch(s); m != nil {
		s = m[1]
	}
	// The small model pads its answer with a step-by-step explanation; the
	// formula it really wants to return usually sits in the final display span
	// (e.g. "最终表达式：\[ \boxed{...} \]"). Take the last complete span.
	for _, pair := range [][2]string{{"\\[", "\\]"}, {"$$", "$$"}, {"\\(", "\\)"}, {"$", "$"}} {
		if inner, ok := lastSpan(s, pair[0], pair[1]); ok {
			return tidy(inner)
		}
	}
	// Fallback: drop prose before the first LaTeX command.
	if k := strings.Index(s, "\\"); k >= 0 {
		s = s[k:]
	}
	return tidy(s)
}

// lastSpan returns the last substring delimited by open/close.
func lastSpan(s, open, close string) (string, bool) {
	best, found := "", false
	idx := 0
	for {
		i := strings.Index(s[idx:], open)
		if i < 0 {
			break
		}
		i += idx
		j := strings.Index(s[i+len(open):], close)
		if j < 0 {
			break
		}
		j += i + len(open)
		best = s[i+len(open) : j]
		found = true
		idx = i + len(open)
	}
	return best, found
}

func tidy(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "\\[")
	s = strings.TrimSuffix(s, "\\]")
	s = strings.TrimPrefix(s, "$$")
	s = strings.TrimSuffix(s, "$$")
	s = strings.TrimPrefix(s, "$")
	s = strings.TrimSuffix(s, "$")
	return strings.Join(strings.Fields(strings.TrimSpace(s)), " ")
}

// ToLatex converts a natural-language description into LaTeX using the loaded
// LLM.
func (s *MathService) ToLatex(text string) (string, error) {
	prompt := "将下面这段自然语言描述转换为一个 LaTeX 数学公式。\n描述：" + text
	out, err := s.generateLLM(prompt)
	if err != nil {
		return "", err
	}
	return cleanLatex(out), nil
}

// OcrToLatex recognises a formula image (PP-OCR) and converts the text into
// LaTeX with the LLM.
func (s *MathService) OcrToLatex(imageB64 string) (string, error) {
	if !s.state.Orc().Loaded() {
		if err := s.loadDefaultOcr(); err != nil {
			return "", err
		}
	}
	res, err := s.state.Orc().RecogniseBase64(imageB64)
	if err != nil {
		return "", fmt.Errorf("图片识别失败: %w", err)
	}
	var parts []string
	for _, l := range res.Lines {
		if strings.TrimSpace(l.Text) != "" {
			parts = append(parts, l.Text)
		}
	}
	ocrText := strings.Join(parts, " ")
	if strings.TrimSpace(ocrText) == "" {
		return "", fmt.Errorf("图片中未识别到文字（PP-OCR 适合印刷体公式，手写公式建议直接在文本区输入）")
	}
	prompt := "下面是公式图片的 OCR 文本，可能含噪声。请整理成正确的 LaTeX 数学公式。\nOCR 文本：" + ocrText
	out, err := s.generateLLM(prompt)
	if err != nil {
		return "", err
	}
	return cleanLatex(out), nil
}

// loadDefaultOcr loads the small PP-OCRv6 tier from the default model dir.
func (s *MathService) loadDefaultOcr() error {
	base := filepath.Join(s.state.cfg.OcrDir, "ch")
	m := paddleocr.Models{
		DetPath:  filepath.Join(base, "PP-OCRv6_small_det.onnx"),
		RecPath:  filepath.Join(base, "PP-OCRv6_small_rec.onnx"),
		OriPath:  filepath.Join(base, "PP-LCNet_x1_0_doc_ori.onnx"),
		DictPath: filepath.Join(base, "PP-OCRv6_dict.txt"),
	}
	if err := s.state.Orc().Load(m); err != nil {
		return fmt.Errorf("OCR 引擎加载失败（请先在 PaddleOCR 页加载模型）: %w", err)
	}
	return nil
}

func (s *MathService) generateLLM(prompt string) (string, error) {
	eng := s.state.LLM()
	if !eng.Loaded() {
		return "", fmt.Errorf("尚未加载对话模型，请先在「LLM 对话」页加载一个模型")
	}
	opts := llm.DefaultGenOptions()
	opts.MaxNewTokens = 1024
	opts.Temperature = 0.2
	opts.TopK = 10
	opts.TopP = 0.9
	opts.UseChatTemplate = true
	opts.SystemPrompt = "你是 LaTeX 公式助手。用户描述后，你严格只输出一个 LaTeX 数学公式本身，不要任何解释、不要 markdown 代码块、不要用 \\[、$ 或 \\begin 包裹。"
	text, err := eng.Generate(prompt, opts, nil)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("模型未返回结果")
	}
	return text, nil
}

// Validate runs deterministic LaTeX checks.
func (s *MathService) Validate(latex string) tex.Result {
	return tex.Validate(latex)
}

// Repair deterministically fixes common LaTeX syntax issues.
func (s *MathService) Repair(latex string) tex.RepairResult {
	return tex.Repair(latex)
}

// CopyText copies text to the system clipboard.
func (s *MathService) CopyText(text string) error {
	cmd := exec.Command("pbcopy")
	in, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	if _, err := in.Write([]byte(text)); err != nil {
		return err
	}
	in.Close()
	return cmd.Wait()
}

// InsertAtCursor copies the text and types it at the system cursor (macOS).
func (s *MathService) InsertAtCursor(text string) error {
	if err := s.CopyText(text); err != nil {
		return err
	}
	// Bring our window to the front first so System Events sends the
	// keystroke to the correct application.
	s.state.ShowMainWindow()
	// Brief pause to let the window focus before sending the keystroke.
	time.Sleep(150 * time.Millisecond)
	script := `tell application "System Events" to keystroke "v" using command down`
	if err := exec.Command("osascript", "-e", script).Run(); err != nil {
		return fmt.Errorf("粘贴失败，请确认已授予辅助功能权限（系统设置 → 隐私与安全性 → 辅助功能）: %w", err)
	}
	return nil
}

// CaptureScreenshot interactively captures a screen region and returns the
// image as a base64 string. The caller is responsible for sending it to a
// vision model for recognition.
func (s *MathService) CaptureScreenshot() (string, error) {
	tmp := filepath.Join(os.TempDir(), "cpm-screenshot.png")
	os.Remove(tmp)
	if err := captureScreenSelection(tmp, s.state); err != nil {
		return "", fmt.Errorf("截图失败: %w", err)
	}
	data, err := os.ReadFile(tmp)
	if err != nil {
		return "", fmt.Errorf("读取截图失败: %w", err)
	}
	return base64.StdEncoding.EncodeToString(data), nil
}