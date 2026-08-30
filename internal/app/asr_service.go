package app

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"cpm_orc/internal/hfhub"
	"cpm_orc/internal/sensevoice"
)

// AsrStatus describes the ASR backend readiness.
type AsrStatus struct {
	BinReady        bool   `json:"binReady"`
	ModelReady      bool   `json:"modelReady"`
	BinPath         string `json:"binPath"`
	ModelPath       string `json:"modelPath"`
	SenseVoiceDir   string `json:"senseVoiceDir"`
	SenseVoiceReady bool   `json:"senseVoiceReady"`
	MossBin         string `json:"mossBin"`
	MossBinReady    bool   `json:"mossBinReady"`
	MossModel       string `json:"mossModel"`
	MossModelReady  bool   `json:"mossModelReady"`
	Backend         string `json:"backend"`
	Transcribing    bool   `json:"transcribing"`
	Recording       bool   `json:"recording"`
}

// AsrService runs speech recognition. Backend "sensevoice" uses FunASR
// SenseVoiceSmall (ONNX, in-process); "whisper" shells out to whisper.cpp
// (Metal on Apple Silicon); "moss" shells out to moss-transcribe (ggml/CPU).
type AsrService struct {
	state        *State
	sv           *sensevoice.Engine
	mu           sync.Mutex
	transcribing bool
}

// NewAsrService creates the ASR service.
func NewAsrService(s *State) *AsrService {
	return &AsrService{
		state: s,
		sv:    sensevoice.NewEngine(4),
	}
}

// Status reports the ASR backend readiness and busy state.
func (s *AsrService) Status() AsrStatus {
	bin := s.state.cfg.WhisperBin
	model := s.state.cfg.WhisperModel
	svDir := s.state.cfg.SenseVoiceDir
	mossBin := s.state.cfg.MossBin
	mossModel := s.state.cfg.MossModel
	s.mu.Lock()
	busy := s.transcribing
	s.mu.Unlock()
	return AsrStatus{
		BinReady:        fileExists(bin),
		ModelReady:      fileExists(model),
		BinPath:         bin,
		ModelPath:       model,
		SenseVoiceDir:   svDir,
		SenseVoiceReady: fileExists(filepath.Join(svDir, "model.onnx")),
		MossBin:         mossBin,
		MossBinReady:    fileExists(mossBin),
		MossModel:       mossModel,
		MossModelReady:  fileExists(mossModel),
		Backend:         s.state.cfg.AsrBackend,
		Transcribing:    busy,
		Recording:       s.IsRecording(),
	}
}

// SetBackend switches between "sensevoice", "whisper", and "moss".
func (s *AsrService) SetBackend(backend string) error {
	if backend != "sensevoice" && backend != "whisper" && backend != "moss" {
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

// SetMossBin changes the moss-transcribe binary path and saves the config.
func (s *AsrService) SetMossBin(path string) error {
	s.state.cfg.MossBin = path
	return s.state.SaveConfig()
}

// SetMossModel changes the moss-transcribe model path and saves the config.
func (s *AsrService) SetMossModel(path string) error {
	s.state.cfg.MossModel = path
	return s.state.SaveConfig()
}

// DownloadMossModel fetches the moss-transcribe GGUF model from HuggingFace.
func (s *AsrService) DownloadMossModel() error {
	model := s.state.cfg.MossModel
	if err := os.MkdirAll(filepath.Dir(model), 0o755); err != nil {
		return err
	}
	if _, err := os.Stat(model); err == nil {
		return nil
	}
	client := hfhub.NewClient(s.state.cfg.Proxy)
	s.state.Emit("dl:start", map[string]any{"id": "moss-transcribe", "file": filepath.Base(model)})
	err := client.Download("mudler/moss-transcribe.cpp-gguf", "main", filepath.Base(model), model,
		func(done, total int64) {
			s.state.Emit("dl:progress", map[string]any{
				"id": "moss-transcribe", "file": filepath.Base(model), "done": done, "total": total,
			})
		})
	if err != nil {
		return err
	}
	s.state.Emit("dl:file-done", map[string]any{"id": "moss-transcribe", "file": filepath.Base(model)})
	return nil
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

// Transcribe runs the active backend on an audio file and returns the transcript.
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

	switch s.state.cfg.AsrBackend {
	case "sensevoice":
		return s.senseVoiceTranscribe(raw, language)
	case "moss":
		return s.mossTranscribe(raw, language)
	default:
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

// mossTranscribe calls the moss-transcribe CLI binary.
func (s *AsrService) mossTranscribe(raw []byte, language string) (string, error) {
	bin := s.state.cfg.MossBin
	model := s.state.cfg.MossModel
	if !fileExists(bin) {
		return "", fmt.Errorf("未找到 moss-transcribe（%s），请先构建并配置", bin)
	}
	if !fileExists(model) {
		return "", fmt.Errorf("未找到 moss-transcribe 模型（%s），请先下载", model)
	}

	// Write audio to temp file
	tmp, err := os.CreateTemp("", "moss-*.wav")
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

	// Run: moss-transcribe transcribe model.gguf audio.wav --format json
	// GGML_METAL_NO_RESIDENCY works around a ggml-metal teardown assert
	// (GGML_ASSERT([rsets->data count] == 0)) that aborts the process on exit
	// even after a successful transcription.
	args := []string{"transcribe", model, tmpPath, "--format", "json"}
	cmd := exec.Command(bin, args...)
	cmd.Env = append(os.Environ(), "GGML_METAL_NO_RESIDENCY=1")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("moss-transcribe 运行失败: %v\n%s", err, lastLines(stderr.String(), 5))
	}

	// Parse JSON output. The CLI emits a bare array of segments; older/other
	// builds may emit an object with "text"/"segments".
	var result mossResult
	if segErr := json.Unmarshal(out, &result.Segments); segErr != nil {
		if objErr := json.Unmarshal(out, &result); objErr != nil {
			// Fallback: treat as plain text
			text := cleanTranscript(string(out))
			if text == "" {
				return "", fmt.Errorf("moss-transcribe 未识别出语音内容")
			}
			s.state.Emit("asr:result", map[string]any{"text": text})
			s.state.ShowMainWindow()
			return text, nil
		}
	}

	// When the CLI returned a bare segment array, build the display text from
	// the speaker-attributed segments.
	text := strings.TrimSpace(result.Text)
	if text == "" && len(result.Segments) > 0 {
		var b strings.Builder
		for _, seg := range result.Segments {
			fmt.Fprintf(&b, "[%s][%s] %s\n",
				formatMossTime(seg.Start), seg.Speaker, strings.TrimSpace(seg.Text))
		}
		text = strings.TrimSpace(b.String())
	}
	if text == "" {
		return "", fmt.Errorf("moss-transcribe 未识别出语音内容（音频可能过短或无人声）")
	}
	s.state.Emit("asr:result", map[string]any{
		"text":     text,
		"segments": result.Segments,
		"backend":  "moss",
	})
	s.state.ShowMainWindow()
	return text, nil
}

// mossResult is the parsed moss-transcribe JSON output.
type mossResult struct {
	Text     string        `json:"text"`
	Segments []mossSegment `json:"segments"`
}

// mossSegment is one speaker-attributed, time-stamped transcript segment.
type mossSegment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
	Text    string  `json:"text"`
}

// formatMossTime renders seconds as MM:SS.s.
func formatMossTime(seconds float64) string {
	m := int(seconds) / 60
	s := seconds - float64(m*60)
	return fmt.Sprintf("%02d:%04.1f", m, s)
}

// lastLines returns at most n trailing non-empty lines of s.
func lastLines(s string, n int) string {
	var lines []string
	for _, ln := range strings.Split(s, "\n") {
		if ln = strings.TrimSpace(ln); ln != "" {
			lines = append(lines, ln)
		}
	}
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

// whisperTranscribe runs whisper.cpp on an audio file.
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
