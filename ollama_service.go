package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
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

// VisionToProblem sends a problem image to the vision model and returns the
// text + formulas as LaTeX (figures are NOT included — call VisionFiguresSVG
// separately to get SVG code for the figures).
func (s *OllamaService) VisionToProblem(model, imageB64 string) (string, error) {
	prompt := `你是数学题目识别助手。请完整识别图片中的文字和公式，输出为 LaTeX 格式。

规则：
1. 题目文字用 \text{} 包裹，保持中文
2. 数学公式用 $...$ 或 $$...$$ 表示
3. 保持题目的原始结构和排版顺序
4. 不要添加解释，只输出内容
5. 不要尝试画图或输出 SVG
6. 使用 align* 环境对齐多行公式（如有）`

	msgs := []ollamaMessage{
		{Role: "system", Content: prompt},
		{Role: "user", Content: "完整识别并输出这道题目的文字和公式为 LaTeX 格式", Images: []string{imageB64}},
	}
	out, err := s.chat(model, msgs, map[string]any{"temperature": 0.05, "num_predict": 2048})
	if err != nil {
		return "", err
	}
	return cleanProblemLatex(out), nil
}

// VisionFiguresSVG detects figure regions in the problem image, crops them,
// and for each one: embeds the cropped image (accurate) + asks the model to
// generate SVG code (editable reference).
func (s *OllamaService) VisionFiguresSVG(model, imageB64 string) (string, error) {
	figures, err := detectFigures(imageB64)
	if err != nil || len(figures) == 0 {
		return `\text{（无配图）}`, nil
	}

	var result strings.Builder
	for i, figB64 := range figures {
		fmt.Fprintf(&result, `\text{图 %d：}`, i+1)
		result.WriteString("\n\n")

		// Always embed the cropped image (accurate)
		result.WriteString(fmt.Sprintf(
			`<img src="data:image/png;base64,%s" style="max-width:60%%;height:auto;display:block;margin:8px auto;" />`, figB64))
		result.WriteString("\n\n")

		// Try to generate SVG (best-effort, may be inaccurate)
		svg := s.tryGenerateSVG(model, figB64)
		if svg != "" {
			result.WriteString(`<details><summary>SVG 代码（可编辑）</summary>`)
			result.WriteString("\n")
			result.WriteString(svg)
			result.WriteString("\n</details>")
			result.WriteString("\n\n")
		}
	}
	return result.String(), nil
}

// tryGenerateSVG sends a cropped figure image to the model and asks for SVG.
func (s *OllamaService) tryGenerateSVG(model, imgB64 string) string {
	prompt := `请为这个几何图形生成 SVG 绘图代码。
要求：<svg viewBox="..." xmlns="http://www.w3.org/2000/svg"> 格式，用 <line> 画线段，<text> 标注字母，stroke="black" stroke-width="1.5"。只输出 SVG 代码。`
	msgs := []ollamaMessage{
		{Role: "system", Content: prompt},
		{Role: "user", Content: "生成 SVG", Images: []string{imgB64}},
	}
	out, err := s.chat(model, msgs, map[string]any{"temperature": 0.1, "num_predict": 1024})
	if err != nil || out == "" {
		return ""
	}
	// extract <svg>...</svg>
	start := strings.Index(out, "<svg")
	if start < 0 {
		return ""
	}
	end := strings.Index(out[start:], "</svg>")
	if end < 0 {
		return ""
	}
	return out[start : start+end+6]
}

// detectFigures finds large non-text blobs in a base64 image and returns
// their base64-encoded PNG crops.
func detectFigures(imgB64 string) ([]string, error) {
	raw, err := base64.StdEncoding.DecodeString(imgB64)
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	bin := make([]bool, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			gray := (r*299 + g*587 + b*114) / 1000
			bin[y*w+x] = gray < 48000
		}
	}

	visited := make([]bool, w*h)
	type box struct{ x0, y0, x1, y1 int }
	var blobs []box

	for y := 0; y < h; y += 2 {
		for x := 0; x < w; x += 2 {
			if bin[y*w+x] && !visited[y*w+x] {
				b := box{x, y, x, y}
				queue := [][2]int{{x, y}}
				visited[y*w+x] = true
				area := 0
				for len(queue) > 0 && area < 500000 {
					cx, cy := queue[0][0], queue[0][1]
					queue = queue[1:]
					area++
					if cx < b.x0 { b.x0 = cx }
					if cx > b.x1 { b.x1 = cx }
					if cy < b.y0 { b.y0 = cy }
					if cy > b.y1 { b.y1 = cy }
					for _, d := range [][2]int{{-2, 0}, {2, 0}, {0, -2}, {0, 2}, {-2, -2}, {2, -2}, {-2, 2}, {2, 2}} {
						nx, ny := cx+d[0], cy+d[1]
						if nx >= 0 && nx < w && ny >= 0 && ny < h && bin[ny*w+nx] && !visited[ny*w+nx] {
							visited[ny*w+nx] = true
							queue = append(queue, [2]int{nx, ny})
						}
					}
				}
				blobs = append(blobs, b)
			}
		}
	}

	var figures []string
	minArea := w * h / 100
	for _, b := range blobs {
		bw, bh := b.x1-b.x0, b.y1-b.y0
		if bw*bh < minArea || bw < w/8 || bh < h/10 {
			continue
		}
		aspect := float64(bw) / float64(bh)
		if aspect < 0.1 || aspect > 10.0 {
			continue
		}
		// Expand bounding box to include nearby text labels (A, B, C, etc.)
		// Use generous padding: max of 40px or 15% of figure dimension
		padX := bw * 15 / 100; if padX < 40 { padX = 40 }
		padY := bh * 15 / 100; if padY < 40 { padY = 40 }
		cx0 := b.x0 - padX; if cx0 < 0 { cx0 = 0 }
		cy0 := b.y0 - padY; if cy0 < 0 { cy0 = 0 }
		cx1 := b.x1 + padX; if cx1 > w { cx1 = w }
		cy1 := b.y1 + padY; if cy1 > h { cy1 = h }

		crop := image.NewRGBA(image.Rect(0, 0, cx1-cx0, cy1-cy0))
		for cy := cy0; cy < cy1; cy++ {
			for cx := cx0; cx < cx1; cx++ {
				crop.Set(cx-cx0, cy-cy0, img.At(cx+bounds.Min.X, cy+bounds.Min.Y))
			}
		}
		var buf bytes.Buffer
		if err := png.Encode(&buf, crop); err == nil {
			figures = append(figures, base64.StdEncoding.EncodeToString(buf.Bytes()))
		}
	}
	return figures, nil
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