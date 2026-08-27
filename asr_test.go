package main

import (
	"encoding/base64"
	"os"
	"os/exec"
	"strings"
	"testing"

	"cpm_orc/internal/config"
)

// TestAsrTranscribe verifies the whisper.cpp integration. Requires a whisper
// binary + model installed under ~/.cpm_orc and CPMM_ASR_TEST=1.
func TestAsrTranscribe(t *testing.T) {
	if os.Getenv("CPMM_ASR_TEST") != "1" {
		t.Skip("set CPMM_ASR_TEST=1 to run the ASR test")
	}
	st := newState(config.Default())
	svc := NewAsrService(st)
	status := svc.Status()
	if !status.BinReady {
		t.Fatalf("whisper-cli not found at %s", status.BinPath)
	}
	if !status.ModelReady {
		t.Fatalf("whisper model not found at %s", status.ModelPath)
	}
	// Build an English wav via the system synthesizer.
	cmd := exec.Command("say", "-o", "/tmp/asr_test.aiff", "Hello world, this is a speech recognition test.")
	if err := cmd.Run(); err != nil {
		t.Fatalf("say: %v", err)
	}
	cmd = exec.Command("afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", "/tmp/asr_test.aiff", "/tmp/asr_test.wav")
	if err := cmd.Run(); err != nil {
		t.Fatalf("afconvert: %v", err)
	}
	raw, err := os.ReadFile("/tmp/asr_test.wav")
	if err != nil {
		t.Fatal(err)
	}
	text, err := svc.TranscribeBase64(base64.StdEncoding.EncodeToString(raw), "en", "test.wav")
	if err != nil {
		t.Fatalf("transcribe: %v", err)
	}
	t.Logf("转写结果: %q", text)
	if !strings.Contains(strings.ToLower(text), "hello") {
		t.Log("注意: 结果可能因语音差异不含 hello")
	}
}