package config

import (
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// Config holds all user-adjustable paths for the application.
type Config struct {
	// ModelRoot is where downloaded HuggingFace models are stored.
	ModelRoot string `json:"modelRoot"`
	// OnnxLibPath is the location of the ONNX Runtime shared library.
	OnnxLibPath string `json:"onnxLibPath"`
	// OcrDir is where PaddleOCR ONNX models live.
	OcrDir string `json:"ocrDir"`
	// LlmDir is where ONNX LLM (MiniCPM) exports live.
	LlmDir string `json:"llmDir"`
	// Proxy is an optional HTTP(S) proxy URL (e.g. "http://127.0.0.1:7890")
	// used for HuggingFace and GitHub downloads.
	Proxy string `json:"proxy"`
	// CoreML enables the ONNX Runtime CoreML execution provider on macOS to
	// accelerate CNN models (OCR) on the Apple GPU / Neural Engine.
	CoreML bool `json:"coreML"`
	// WhisperBin is the path to the whisper.cpp CLI (Metal), used for speech
	// recognition.
	WhisperBin string `json:"whisperBin"`
	// WhisperModel is the path to a whisper.cpp GGML/GGUF model.
	WhisperModel string `json:"whisperModel"`
	// AsrBackend selects the ASR engine: "sensevoice" (FunASR SenseVoiceSmall,
	// ONNX) or "whisper" (whisper.cpp).
	AsrBackend string `json:"asrBackend"`
	// SenseVoiceDir is the FunASR SenseVoiceSmall ONNX model directory.
	SenseVoiceDir string `json:"senseVoiceDir"`
	// OllamaHost is the Ollama server base URL (OpenAI-compatible chat API).
	OllamaHost string `json:"ollamaHost"`
}

// Default returns the default configuration rooted in the user's home
// directory.
func Default() *Config {
	home, _ := os.UserHomeDir()
	base := filepath.Join(home, ".cpm_orc")
	return &Config{
		ModelRoot:    filepath.Join(base, "models"),
		OnnxLibPath:  filepath.Join(base, "lib", ortLibName()),
		OcrDir:       filepath.Join(base, "models", "paddleocr"),
		LlmDir:       filepath.Join(base, "models", "llm"),
		// CoreML is off by default: measured slower for the small CNN OCR
		// models (Apple CPU is faster than the CoreML EP overhead here).
		CoreML:       false,
		WhisperBin:   filepath.Join(base, "whisper", "whisper-cli"),
		WhisperModel: filepath.Join(base, "models", "whisper", "ggml-base.bin"),
		AsrBackend:   "sensevoice",
		SenseVoiceDir: filepath.Join(base, "models", "sensevoice"),
		OllamaHost:    "http://localhost:11434",
	}
}

// HTTPClient builds an *http.Client that routes through the configured proxy
// when proxy is non-empty, otherwise it falls back to the environment's proxy
// settings (http_proxy / https_proxy). A zero timeout means no timeout.
func HTTPClient(proxy string, timeout time.Duration) *http.Client {
	transport := &http.Transport{Proxy: http.ProxyFromEnvironment}
	if proxy != "" {
		if u, err := url.Parse(proxy); err == nil {
			transport.Proxy = http.ProxyURL(u)
		}
	}
	return &http.Client{Transport: transport, Timeout: timeout}
}

func ortLibName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return "libonnxruntime.dylib"
	default:
		return "libonnxruntime.so"
	}
}

// Load reads the config file, falling back to defaults if it does not exist.
func Load(path string) (*Config, error) {
	c := Default()
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return c, nil
		}
		return nil, err
	}
	if err := json.Unmarshal(data, c); err != nil {
		return nil, err
	}
	return c, nil
}

// Save writes the config to disk.
func (c *Config) Save(path string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// EnsureDirs creates all configured directories.
func (c *Config) EnsureDirs() error {
	for _, d := range []string{c.ModelRoot, filepath.Dir(c.OnnxLibPath), c.OcrDir, c.LlmDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return err
		}
	}
	return nil
}