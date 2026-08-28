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

// VisionToProblem sends a problem image (containing text + formulas + figures)
// to a multimodal Ollama model and returns the full problem reproduced in
// LaTeX — text as \text{}, formulas in math mode, figures embedded as base64.
func (s *OllamaService) VisionToProblem(model, imageB64 string) (string, error) {
	// Step 1: detect figure regions in the original image
	figures, err := detectFigures(imageB64)
	if err != nil {
		figures = nil // degrade gracefully
	}

	// Step 2: ask the model to output text + formulas, marking figures
	figureHint := ""
	if len(figures) > 0 {
		figureHint = fmt.Sprintf("\n\n图片中检测到 %d 个配图区域。请在对应位置用 [FIGURE_0]、[FIGURE_1] 等标记占位。", len(figures))
	}
	prompt := fmt.Sprintf(`你是数学题目识别助手。请完整识别图片中的所有内容，包括：
1. 题目文字（题号、题干、选项等），用 \\text{{}} 包裹
2. 数学公式，用 $...$ 或 $$...$$ 表示
3. 配图/图形用 [FIGURE_N] 标记占位（N 从0开始）

输出要求：
- 保持题目的原始结构和排版顺序
- 文字部分用 \\text{{}} 包裹，保持中文
- 公式用标准 LaTeX 数学模式
- 配图位置用 [FIGURE_N] 标记，不要尝试用 \\includegraphics
- 不要添加解释，只输出 LaTeX 内容
- 使用 align* 环境对齐多行公式（如有）%s`, figureHint)

	msgs := []ollamaMessage{
		{Role: "system", Content: prompt},
		{Role: "user", Content: "完整识别并输出这道题目的所有内容为 LaTeX 格式", Images: []string{imageB64}},
	}
	out, err := s.chat(model, msgs, map[string]any{"temperature": 0.05, "num_predict": 2048})
	if err != nil {
		return "", err
	}
	result := cleanProblemLatex(out)

	// Step 3: replace [FIGURE_N] markers with embedded base64 images
	for i, fig := range figures {
		tag := fmt.Sprintf("[FIGURE_%d]", i)
		if i < len(figures) {
			dataURI := fmt.Sprintf(`<img src="data:image/png;base64,%s" style="max-width:100%%;height:auto;" />`, fig)
			result = strings.ReplaceAll(result, tag, dataURI)
		}
	}
	// also replace any remaining markers the model didn't use
	for i := len(figures); i < 10; i++ {
		tag := fmt.Sprintf("[FIGURE_%d]", i)
		result = strings.ReplaceAll(result, tag, "")
	}

	return result, nil
}

// detectFigures finds non-text figure regions in a base64-encoded image and
// returns their base64-encoded PNG crops. Uses simple connected-component
// analysis: large blobs that aren't text lines are treated as figures.
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

	// Convert to binary (non-white = black)
	bin := make([]bool, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			gray := (r*299 + g*587 + b*114) / 1000
			bin[y*w+x] = gray < 48000 // threshold: non-white
		}
	}

	// Simple connected-component labeling via flood fill
	visited := make([]bool, w*h)
	type box struct{ x0, y0, x1, y1 int }
	var blobs []box

	for y := 0; y < h; y += 3 { // sample every 3rd pixel for speed
		for x := 0; x < w; x += 3 {
			if bin[y*w+x] && !visited[y*w+x] {
				// flood fill
				b := box{x, y, x, y}
				queue := [][2]int{{x, y}}
				visited[y*w+x] = true
				area := 0
				for len(queue) > 0 && area < 200000 {
					cx, cy := queue[0][0], queue[0][1]
					queue = queue[1:]
					area++
					if cx < b.x0 { b.x0 = cx }
					if cx > b.x1 { b.x1 = cx }
					if cy < b.y0 { b.y0 = cy }
					if cy > b.y1 { b.y1 = cy }
					for _, d := range [][2]int{{-3,0},{3,0},{0,-3},{0,3}} {
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

	// Filter: keep blobs that are large enough and roughly figure-shaped
	// (wide + tall, not just a text line)
	var figures []string
	minArea := w * h / 40 // at least 2.5% of image
	for _, b := range blobs {
		bw, bh := b.x1-b.x0, b.y1-b.y0
		area := bw * bh
		if area < minArea { continue }
		// figure should be reasonably wide and tall (not just a line)
		if bw < w/6 || bh < h/8 { continue }
		// aspect ratio: not too skinny
		aspect := float64(bw) / float64(bh)
		if aspect < 0.15 || aspect > 7.0 { continue }

		// crop with padding
		pad := 10
		cx0 := b.x0 - pad; if cx0 < 0 { cx0 = 0 }
		cy0 := b.y0 - pad; if cy0 < 0 { cy0 = 0 }
		cx1 := b.x1 + pad; if cx1 > w { cx1 = w }
		cy1 := b.y1 + pad; if cy1 > h { cy1 = h }

		// create cropped image
		crop := image.NewRGBA(image.Rect(0, 0, cx1-cx0, cy1-cy0))
		for cy := cy0; cy < cy1; cy++ {
			for cx := cx0; cx < cx1; cx++ {
				crop.Set(cx-cx0, cy-cy0, img.At(cx+bounds.Min.X, cy+bounds.Min.Y))
			}
		}

		// encode to PNG base64
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