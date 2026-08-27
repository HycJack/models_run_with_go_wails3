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