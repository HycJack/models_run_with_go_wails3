package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// OllamaService talks to a local Ollama server over its native chat API. It
// provides an alternative text backend (MiniCPM5-1B etc.) and, crucially,
// a vision backend (MiniCPM-V 4.6) that reads a formula image directly —
// the same approach TeXada uses, replacing the PP-OCR text pipeline.
type OllamaService struct {
	state *State
	http  *http.Client

	mu     sync.Mutex
	cancel context.CancelFunc
}

// NewOllamaService creates the Ollama client service.
func NewOllamaService(s *State) *OllamaService {
	return &OllamaService{
		state: s,
		http:  &http.Client{Timeout: 120 * time.Second},
	}
}

// OllamaModel is one installed model returned by the server.
type OllamaModel struct {
	Name    string `json:"name"`
	Size    int64  `json:"size"`
	Details struct {
		Family string `json:"family"`
		Format string `json:"format"`
		Param  string `json:"parameter_size"`
	} `json:"details"`
}

// Host returns the configured Ollama base URL.
func (s *OllamaService) Host() string { return s.state.cfg.OllamaHost }

// SetHost updates the Ollama base URL and persists it.
func (s *OllamaService) SetHost(host string) {
	s.state.cfg.OllamaHost = strings.TrimRight(host, "/")
	_ = s.state.SaveConfig()
}

// Ping checks that the Ollama server is reachable.
func (s *OllamaService) Ping() (bool, error) {
	resp, err := s.http.Get(s.state.cfg.OllamaHost + "/api/tags")
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK, nil
}

// ListModels lists models installed in Ollama.
func (s *OllamaService) ListModels() ([]OllamaModel, error) {
	resp, err := s.http.Get(s.state.cfg.OllamaHost + "/api/tags")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var out struct {
		Models []OllamaModel `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Models, nil
}

// Stop cancels the in-flight Ollama request, if any.
func (s *OllamaService) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cancel != nil {
		s.cancel()
		s.cancel = nil
	}
}

// ollamaMessage mirrors the /api/chat message shape.
type ollamaMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

// chat performs a streaming chat completion and returns the full text.
func (s *OllamaService) chat(model string, messages []ollamaMessage, options map[string]any) (string, error) {
	ctx, cancel := context.WithCancel(context.Background())
	s.mu.Lock()
	s.cancel = cancel
	s.mu.Unlock()
	defer cancel()

	body, _ := json.Marshal(map[string]any{
		"model":    model,
		"messages": messages,
		"stream":   true,
		// MiniCPM5 is a reasoning model: disable thinking so the token budget
		// is spent on the answer, not internal monologue.
		"think":   false,
		"options": options,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.state.cfg.OllamaHost+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return "", fmt.Errorf("已停止")
		}
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}

	var full strings.Builder
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var chunk struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Done bool `json:"done"`
		}
		if err := json.Unmarshal(line, &chunk); err != nil {
			continue
		}
		if chunk.Message.Content != "" {
			full.WriteString(chunk.Message.Content)
		}
		if chunk.Done {
			break
		}
	}
	if err := sc.Err(); err != nil {
		if ctx.Err() != nil {
			return "", fmt.Errorf("已停止")
		}
		return "", err
	}
	return full.String(), nil
}

// Generate runs a plain text chat completion via Ollama.
func (s *OllamaService) Generate(model, prompt, system string, temperature float64, maxTokens int) (string, error) {
	msgs := []ollamaMessage{}
	if strings.TrimSpace(system) != "" {
		msgs = append(msgs, ollamaMessage{Role: "system", Content: system})
	}
	msgs = append(msgs, ollamaMessage{Role: "user", Content: prompt})
	return s.chat(model, msgs, map[string]any{
		"temperature": temperature,
		"num_predict": maxTokens,
	})
}

// VisionToLatex sends a formula image to a multimodal Ollama model (e.g.
// MiniCPM-V 4.6) and returns the recognised LaTeX, cleaned of prose.
func (s *OllamaService) VisionToLatex(model, imageB64 string) (string, error) {
	msgs := []ollamaMessage{
		{Role: "system", Content: "你是公式识别助手。识别图片中的数学公式，只输出 LaTeX 公式本身，不要任何解释。"},
		{Role: "user", Content: "识别图片中的数学公式", Images: []string{imageB64}},
	}
	out, err := s.chat(model, msgs, map[string]any{"temperature": 0.05, "num_predict": 512})
	if err != nil {
		return "", err
	}
	return cleanLatex(out), nil
}

// VisionToProblem sends a problem image to the vision model and asks it to
// reproduce the full content as LaTeX with inline SVG for figures.
func (s *OllamaService) VisionToProblem(model, imageB64 string) (string, error) {
	prompt := `你是数学题目识别助手。请完整识别图片中的所有内容，输出为 LaTeX 格式。

规则：
1. 题目文字用 \text{} 包裹，保持中文
2. 数学公式用 $...$ 或 $$...$$ 表示
3. 配图/几何图形用 <svg> 代码绘制（不是 \includegraphics）
4. 保持题目的原始结构和排版顺序
5. 不要添加解释，只输出内容

SVG 绘图要求：
- 使用 <svg viewBox="..." xmlns="http://www.w3.org/2000/svg">
- 用 <line> 画线段，<circle> 画圆，<path> 画曲线，<rect> 画矩形
- 用 <text> 标注顶点字母和数值（font-size=14, text-anchor=middle）
- 直角用小正方形标注（两条短线段）
- viewBox 大小合适（通常 0 0 300 200）
- stroke="black" stroke-width="1.5"
- 虚线用 stroke-dasharray="5,3"

示例输出：
\text{（1）如图所示，}
<svg viewBox="0 0 200 150" xmlns="http://www.w3.org/2000/svg">
  <line x1="20" y1="130" x2="180" y2="130" stroke="black" stroke-width="1.5"/>
  <line x1="20" y1="130" x2="100" y2="20" stroke="black" stroke-width="1.5"/>
  <line x1="100" y1="20" x2="180" y2="130" stroke="black" stroke-width="1.5"/>
  <text x="15" y="145" font-size="14">A</text>
  <text x="100" y="15" font-size="14">B</text>
  <text x="185" y="145" font-size="14">C</text>
</svg>
\text{，已知 AB = 3，BC = 4，求 AC 的长度。}
$AC = \sqrt{AB^2 + BC^2} = 5$`

	msgs := []ollamaMessage{
		{Role: "system", Content: prompt},
		{Role: "user", Content: "完整识别并输出这道题目的所有内容为 LaTeX + SVG 格式", Images: []string{imageB64}},
	}
	out, err := s.chat(model, msgs, map[string]any{"temperature": 0.05, "num_predict": 4096})
	if err != nil {
		return "", err
	}
	return cleanProblemLatex(out), nil
}

// cleanProblemLatex strips markdown fences and leading/trailing prose from a
// full-problem LaTeX block (unlike cleanLatex which expects a single formula).
func cleanProblemLatex(s string) string {
	s = strings.TrimSpace(s)
	// strip markdown code fences
	if i := strings.Index(s, "```"); i >= 0 {
		j := strings.Index(s[i+3:], "```")
		if j >= 0 {
			s = s[i+3 : i+3+j]
		}
	}
	s = strings.TrimSpace(s)
	// strip leading "```latex" if still present
	for _, prefix := range []string{"```latex", "```tex", "```"} {
		if strings.HasPrefix(s, prefix) {
			s = strings.TrimPrefix(s, prefix)
		}
	}
	return strings.TrimSpace(s)
}

// ToLatex converts natural language to LaTeX via an Ollama text model.
func (s *OllamaService) ToLatex(model, text string) (string, error) {
	out, err := s.Generate(model, "将下面这段自然语言描述转换为一个 LaTeX 数学公式。\n描述："+text,
		"你是 LaTeX 公式助手。严格只输出一个 LaTeX 数学公式本身，不要解释、不要 markdown 代码块、不要用 \\[、$ 或 \\begin 包裹。",
		0.1, 512)
	if err != nil {
		return "", err
	}
	return cleanLatex(out), nil
}