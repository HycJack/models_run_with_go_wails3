package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/wailsapp/wails/v3/pkg/application"

	"cpm_orc/internal/config"
	"cpm_orc/internal/llm"
	"cpm_orc/internal/ort"
	"cpm_orc/internal/paddleocr"
)

// State holds the shared application resources used by the services.
type State struct {
	mu          sync.Mutex
	cfg         *config.Config
	app         *application.App
	ocr         *paddleocr.Engine
	llm         *llm.Engine
	mainWindow  *application.WebviewWindow
}

func newState(cfg *config.Config) *State {
	return &State{
		cfg: cfg,
		ocr: paddleocr.NewEngine(),
		llm: llm.NewEngine(4),
	}
}

// SetApp wires the application instance for event emission.
func (s *State) SetApp(app *application.App) { s.app = app }

// SetMainWindow records the main window so background tasks (tray, shortcuts)
// can show/focus it.
func (s *State) SetMainWindow(w *application.WebviewWindow) { s.mainWindow = w }

// EnsureOrt initializes the ONNX Runtime environment if it is not already
// ready. Session-creating operations (OCR/LLM/ASR load) call this so they
// never fail with "ONNX Runtime is not initialized".
func (s *State) EnsureOrt() error {
	if ort.IsInitialized() {
		return nil
	}
	if _, err := os.Stat(s.cfg.OnnxLibPath); err != nil {
		return fmt.Errorf("ONNX Runtime 未安装，请先在「运行环境」页下载")
	}
	return ort.Init(s.cfg.OnnxLibPath)
}

// ShowMainWindow brings the main window to the front.
func (s *State) ShowMainWindow() {
	if s.mainWindow != nil {
		s.mainWindow.Show()
		s.mainWindow.Focus()
	}
}

// MainWindow returns the main window (may be nil).
func (s *State) MainWindow() *application.WebviewWindow { return s.mainWindow }

// Emit sends an event to the frontend (best effort).
func (s *State) Emit(name string, data any) {
	if s.app == nil {
		return
	}
	s.app.Event.Emit(name, data)
}

// ConfigPath returns the config file location.
func (s *State) ConfigPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cpm_orc", "config.json")
}

// configPath returns the global config file location.
func configPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cpm_orc", "config.json")
}

// Save persists the current configuration.
func (s *State) SaveConfig() error {
	return s.cfg.Save(s.ConfigPath())
}

// ModelRoot returns the model root directory.
func (s *State) ModelRoot() string { return s.cfg.ModelRoot }

// Orc returns the OCR engine.
func (s *State) Orc() *paddleocr.Engine { return s.ocr }

// LLM returns the LLM engine.
func (s *State) LLM() *llm.Engine { return s.llm }

// OpenFolder opens a directory in the platform file manager.
func (s *State) OpenFolder(path string) error {
	if _, err := os.Stat(path); err != nil {
		return err
	}
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", path)
	case "windows":
		cmd = exec.Command("explorer", path)
	default:
		cmd = exec.Command("xdg-open", path)
	}
	return cmd.Start()
}