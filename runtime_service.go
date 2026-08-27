package main

import (
	"fmt"
	"net/http"
	"os"
	"time"

	"cpm_orc/internal/config"
	"cpm_orc/internal/ort"
)

// RuntimeService manages the ONNX Runtime shared library lifecycle.
type RuntimeService struct {
	state *State
}

// NewRuntimeService creates the runtime service.
func NewRuntimeService(s *State) *RuntimeService { return &RuntimeService{state: s} }

// Status returns whether the ONNX Runtime library is present and loadable.
func (s *RuntimeService) Status() ort.RuntimeStatus {
	return ort.CheckRuntime(s.state.cfg.OnnxLibPath)
}

// Version returns the bundled expected ONNX Runtime version.
func (s *RuntimeService) Version() string { return ort.LatestRuntimeVersion }

// LibFileName returns the library file name for the host platform.
func (s *RuntimeService) LibFileName() string { return ort.LibFileName() }

// Config returns the current application configuration.
func (s *RuntimeService) Config() *config.Config { return s.state.cfg }

// SaveConfig persists the configuration and returns errors if any.
func (s *RuntimeService) SaveConfig() error { return s.state.SaveConfig() }

// Download fetches and extracts the ONNX Runtime shared library. Progress is
// reported through the "dl:progress" event.
func (s *RuntimeService) Download(version string) (string, error) {
	if version == "" {
		version = ort.LatestRuntimeVersion
	}
	// Apply the configured proxy before downloading.
	ort.SetProxy(s.state.cfg.Proxy)
	libDir := dirOf(s.state.cfg.OnnxLibPath)
	path, err := ort.DownloadRuntime(libDir, version, func(done, total int64) {
		s.state.Emit("dl:progress", map[string]any{
			"id":    "onnxruntime",
			"file":  "onnxruntime " + version,
			"done":  done,
			"total": total,
		})
	})
	if err != nil {
		return "", err
	}
	s.state.cfg.OnnxLibPath = path
	s.state.SaveConfig()
	s.state.Emit("dl:done", map[string]any{"id": "onnxruntime", "path": path})
	return path, nil
}

// EnsureRuntime initializes the ONNX Runtime environment, downloading the
// library first if it is missing.
func (s *RuntimeService) EnsureRuntime() (string, error) {
	if ort.IsInitialized() {
		return s.state.cfg.OnnxLibPath, nil
	}
	if _, err := os.Stat(s.state.cfg.OnnxLibPath); err != nil {
		if _, err := s.Download(ort.LatestRuntimeVersion); err != nil {
			return "", fmt.Errorf("failed to provision ONNX Runtime: %w", err)
		}
	}
	if err := ort.Init(s.state.cfg.OnnxLibPath); err != nil {
		return "", err
	}
	return s.state.cfg.OnnxLibPath, nil
}

// TestProxy verifies that a proxy can reach the HuggingFace Hub.
func (s *RuntimeService) TestProxy(proxy string) error {
	if proxy == "" {
		proxy = s.state.cfg.Proxy
	}
	client := config.HTTPClient(proxy, 15*time.Second)
	resp, err := client.Get("https://huggingface.co/api/models?limit=1")
	if err != nil {
		return fmt.Errorf("代理不可用: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("代理返回状态码 %d", resp.StatusCode)
	}
	return nil
}

// OpenFolder opens the given directory in the file manager.
func (s *RuntimeService) OpenFolder(path string) error {
	return s.state.OpenFolder(path)
}

// OpenMain shows and focuses the main window.
func (s *RuntimeService) OpenMain() {
	s.state.ShowMainWindow()
}

func dirOf(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' || p[i] == '\\' {
			return p[:i]
		}
	}
	return "."
}