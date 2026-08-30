package app

import (
	"encoding/json"
	"os"
	"path/filepath"

	"cpm_orc/internal/llm"
)

// LlmStatus describes the current LLM engine state.
type LlmStatus struct {
	Loaded     bool   `json:"loaded"`
	Dir        string `json:"dir"`
	ModelType  string `json:"modelType"`
	Generating bool   `json:"generating"`
	Threads    int    `json:"threads"`
	// Vision reports whether the loaded model can understand images directly
	// (has a vision encoder / image token config).
	Vision bool `json:"vision"`
}

// detectVision reports whether a model directory's config.json declares a
// vision encoder.
func detectVision(dir string) bool {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return false
	}
	for _, k := range []string{"vision_config", "vision_tower", "image_token_index", "image_token_id", "mm_resampler_config", "video_config"} {
		if _, ok := m[k]; ok {
			return true
		}
	}
	return false
}

// LlmService manages ONNX small-model inference (Qwen / MiniCPM families).
type LlmService struct {
	state *State
}

// NewLlmService creates the LLM service.
func NewLlmService(s *State) *LlmService { return &LlmService{state: s} }

// Status returns the current LLM engine state.
func (s *LlmService) Status() LlmStatus {
	eng := s.state.LLM()
	st := LlmStatus{
		Loaded:     eng.Loaded(),
		Generating: eng.IsGenerating(),
		Threads:    4,
	}
	if st.Loaded {
		dir, modelType, _ := eng.Status()
		st.Dir = dir
		st.ModelType = modelType
		st.Vision = detectVision(dir)
	}
	return st
}

// Load loads an ONNX model directory (model.onnx + config.json +
// tokenizer.json). The directory may also be a model ID under the model root.
func (s *LlmService) Load(dirOrID string) error {
	if err := s.state.EnsureOrt(); err != nil {
		return err
	}
	dir := dirOrID
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(s.state.ModelRoot(), filepath.FromSlash(dir))
	}
	if _, err := os.Stat(dir); err != nil {
		return err
	}
	return s.state.LLM().Load(dir)
}

// Unload releases the loaded model.
func (s *LlmService) Unload() error { return s.state.LLM().Unload() }

// Generate produces a completion, streaming tokens via the "llm:token" event
// when opts.Stream is true. Reasoning sections (if the model emits them) are
// streamed via the "llm:reasoning" event. Returns the full generated text.
func (s *LlmService) Generate(prompt string, opts llm.GenOptions) (string, error) {
	s.state.Emit("llm:status", map[string]any{"generating": true})
	defer s.state.Emit("llm:status", map[string]any{"generating": false})
	return s.state.LLM().Generate(prompt, opts, func(chunk string) {
		s.state.Emit("llm:token", map[string]any{"text": chunk})
	}, func(reasoning string) {
		s.state.Emit("llm:reasoning", map[string]any{"text": reasoning})
	})
}

// Stop aborts the current generation.
func (s *LlmService) Stop() { s.state.LLM().Stop() }

// LocalOnnxModels returns local model directories that look like LLM ONNX
// exports (model.onnx + config.json).
func (s *LlmService) LocalOnnxModels() ([]string, error) {
	root := s.state.ModelRoot()
	seen := map[string]bool{}
	var out []string
	filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if filepath.Base(p) != "model.onnx" {
			return nil
		}
		dir := filepath.Dir(p)
		if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
			return nil // not an LLM export
		}
		if !seen[dir] {
			seen[dir] = true
			out = append(out, dir)
		}
		return nil
	})
	return out, nil
}

// DefaultModelDir returns the configured LLM model directory.
func (s *LlmService) DefaultModelDir() string { return s.state.cfg.LlmDir }