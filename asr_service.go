package main

import (
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"cpm_orc/internal/hfhub"
	"cpm_orc/internal/sensevoice"
)

// AsrStatus describes the whisper.cpp setup.
type AsrStatus struct {
	BinReady      bool   `json:"binReady"`
	ModelReady    bool   `json:"modelReady"`
	BinPath       string `json:"binPath"`
	ModelPath     string `json:"modelPath"`
	SenseVoiceDir string `json:"senseVoiceDir"`
	SenseVoiceReady bool `json:"senseVoiceReady"`
	Backend       string `json:"backend"`
	Transcribing  bool   `json:"transcribing"`
	Recording     bool   `json:"recording"`
}

// AsrService runs speech recognition. Backend "sensevoice" uses FunASR
// SenseVoiceSmall (ONNX, in-process); "whisper" shells out to whisper.cpp
// (Metal on Apple Silicon).
type AsrService struct {
	state        *State
	sv           *sensevoice.Engine
	mu           sync.Mutex
	transcribing bool
}

// NewAsrService creates the ASR service.
func NewAsrService(s *State) *AsrService {
	return &AsrService{state: s, sv: sensevoice.NewEngine(4)}
}

// Status reports the ASR backend readiness and busy state.
func (s *AsrService) Status() AsrStatus {
	bin := s.state.cfg.WhisperBin
	model := s.state.cfg.WhisperModel
	svDir := s.state.cfg.SenseVoiceDir
	s.mu.Lock()
	busy := s.transcribing
	s.mu.Unlock()
	return AsrStatus{
		BinReady:       fileExists(bin),
		ModelReady:     fileExists(model),
		BinPath:        bin,
		ModelPath:      model,
		SenseVoiceDir:  svDir,
		SenseVoiceReady: fileExists(filepath.Join(svDir, "model.onnx")),
		Backend:        s.state.cfg.AsrBackend,
		Transcribing:   busy,
		Recording:      s.IsRecording(),
	}
}

// SetBackend switches between "sensevoice" and "whisper".
func (s *AsrService) SetBackend(backend string) error {
	if backend != "sensevoice" && backend != "whisper" {
		return fmt.Errorf("unknown backend %q", backend)
	}
	s.state.cfg.AsrBackend = backend
	return s.state.SaveConfig()
}

// SetSenseVoiceDir sets the FunASR model directory and loads it.
func (s *AsrService) SetSenseVoiceDir(dir string) error {
	if err := s.state.EnsureOrt(); err != nil {
		return err
	}
	s.state.cfg.SenseVoiceDir = dir
	if err := s.state.SaveConfig(); err != nil {
		return err
	}
	return s.sv.Load(dir)
}

// SetBin changes the whisper-cli path and saves the config.
func (s *AsrService) SetBin(path string) error {
	s.state.cfg.WhisperBin = path
	return s.state.SaveConfig()
}

// SetModel changes the whisper model path and saves the config.
func (s *AsrService) SetModel(path string) error {
	s.state.cfg.WhisperModel = path
	return s.state.SaveConfig()
}

// DownloadModel fetches the default ggml-base.bin model via the configured
// proxy, reporting progress through the "dl:progress" event.
func (s *AsrService) DownloadModel() error {
	model := s.state.cfg.WhisperModel
	if err := os.MkdirAll(filepath.Dir(model), 0o755); err != nil {
		return err
	}
	if _, err := os.Stat(model); err == nil {
		return nil
	}
	client := hfhub.NewClient(s.state.cfg.Proxy)
	s.state.Emit("dl:start", map[string]any{"id": "whisper", "file": filepath.Base(model)})
	err := client.Download("ggerganov/whisper.cpp", "main", "ggml-base.bin", model,
		func(done, total int64) {
			s.state.Emit("dl:progress", map[string]any{
				"id": "whisper", "file": filepath.Base(model), "done": done, "total": total,
			})
		})
	if err != nil {
		return err
	}
	s.state.Emit("dl:file-done", map[string]any{"id": "whisper", "file": filepath.Base(model)})
	return nil
}

// Transcribe runs whisper.cpp on an audio file and returns the transcript.
// language is an ISO code (en/zh/...) or "auto".
func (s *AsrService) Transcribe(audioPath, language string) (string, error) {
	raw, err := os.ReadFile(audioPath)
	if err != nil {
		return "", err
	}
	return s.transcribeBytes(raw, language)
}

// TranscribeBase64 accepts base64-encoded audio bytes and transcribes them
// with the selected backend.
func (s *AsrService) TranscribeBase64(dataB64, language, filename string) (string, error) {
	raw, err := base64.StdEncoding.DecodeString(dataB64)
	if err != nil {
		return "", fmt.Errorf("解码音频失败: %w", err)
	}
	return s.transcribeBytes(raw, language)
}

func (s *AsrService) transcribeBytes(raw []byte, language string) (string, error) {
	s.mu.Lock()
	s.transcribing = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.transcribing = false
		s.mu.Unlock()
	}()
	s.state.Emit("asr:status", map[string]any{"transcribing": true})
	defer s.state.Emit("asr:status", map[string]any{"transcribing": false})

	if s.state.cfg.AsrBackend == "sensevoice" {
		return s.senseVoiceTranscribe(raw, language)
	}
	// whisper backend: write to a temp file.
	tmp, err := os.CreateTemp("", "asr-*.wav")
	if err != nil {
		return "", err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(raw); err != nil {
		tmp.Close()
		return "", err
	}
	tmp.Close()
	return s.whisperTranscribe(tmpPath, language)
}

func (s *AsrService) senseVoiceTranscribe(raw []byte, language string) (string, error) {
	text, err := s.sv.Transcribe(raw, language, "woitn")
	if err != nil {
		return "", err
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return "", fmt.Errorf("未识别出语音内容（音频可能过短或无人声）")
	}
	s.state.Emit("asr:result", map[string]any{"text": text})
	s.state.ShowMainWindow()
	return text, nil
}

// Transcribe runs whisper.cpp on an audio file and returns the transcript.
// language is an ISO code (en/zh/...) or "auto".
func (s *AsrService) whisperTranscribe(audioPath, language string) (string, error) {
	bin := s.state.cfg.WhisperBin
	model := s.state.cfg.WhisperModel
	if !fileExists(bin) {
		return "", fmt.Errorf("未找到 whisper-cli（%s），请先构建并配置", bin)
	}
	if !fileExists(model) {
		return "", fmt.Errorf("未找到 whisper 模型（%s），请先下载", model)
	}

	args := []string{"-m", model, "-f", audioPath, "--no-timestamps"}
	if language != "" && language != "auto" {
		args = append(args, "-l", language)
	}
	cmd := exec.Command(bin, args...)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("whisper 运行失败: %v", err)
	}
	text := cleanTranscript(string(out))
	if text == "" {
		return "", fmt.Errorf("未识别出语音内容（音频可能过短或无人声）")
	}
	s.state.Emit("asr:result", map[string]any{"text": text})
	s.state.ShowMainWindow()
	return text, nil
}

// cleanTranscript extracts the transcript text from whisper-cli stdout.
func cleanTranscript(out string) string {
	var lines []string
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		lines = append(lines, ln)
	}
	return strings.Join(lines, "\n")
}

// recording state.
var (
	recMu   sync.Mutex
	recCmd  *exec.Cmd
	recPath string
)

// StartRecording begins microphone capture via ffmpeg (AVFoundation on
// macOS). The recording is written to a temp WAV (16k mono).
func (s *AsrService) StartRecording() error {
	recMu.Lock()
	defer recMu.Unlock()
	if recCmd != nil {
		return fmt.Errorf("已在录音中")
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return fmt.Errorf("未找到 ffmpeg，无法录音（macOS 可用 brew install ffmpeg）")
	}
	recPath = filepath.Join(os.TempDir(), "cpm-recording.wav")
	os.Remove(recPath)
	cmd := exec.Command("ffmpeg",
		"-hide_banner", "-loglevel", "error",
		"-f", "avfoundation", "-i", ":0",
		"-ac", "1", "-ar", "16000",
		"-y", recPath)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("启动录音失败: %w", err)
	}
	recCmd = cmd
	s.state.Emit("asr:status", map[string]any{"recording": true})
	return nil
}

// StopAndTranscribe stops the recording and transcribes it with the active
// backend.
func (s *AsrService) StopAndTranscribe(language string) (string, error) {
	recMu.Lock()
	if recCmd == nil {
		recMu.Unlock()
		return "", fmt.Errorf("当前没有在录音")
	}
	cmd := recCmd
	path := recPath
	recCmd = nil
	recMu.Unlock()

	// ffmpeg stops cleanly on SIGINT and flushes the file.
	_ = cmd.Process.Signal(os.Interrupt)
	_ = cmd.Wait()
	s.state.Emit("asr:status", map[string]any{"recording": false})

	raw, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("读取录音失败: %w", err)
	}
	return s.transcribeBytes(raw, language)
}

// IsRecording reports whether a recording is in progress.
func (s *AsrService) IsRecording() bool {
	recMu.Lock()
	defer recMu.Unlock()
	return recCmd != nil
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}