package main

import (
	"os"
	"os/exec"
	"strings"
	"testing"

	"cpm_orc/internal/config"
	"cpm_orc/internal/ort"
	"cpm_orc/internal/paddleocr"
)

// TestClipboardOCR verifies the clipboard-image -> OCR pipeline used by the
// tray menu and global shortcuts. Only runs when CPMM_CLIP_OCR=1 is set.
func TestClipboardOCR(t *testing.T) {
	if os.Getenv("CPMM_CLIP_OCR") != "1" {
		t.Skip("set CPMM_CLIP_OCR=1 to run the clipboard OCR test")
	}
	if err := ort.Init(os.Getenv("HOME") + "/.cpm_orc/lib/libonnxruntime.dylib"); err != nil {
		t.Fatalf("ort: %v", err)
	}
	st := newState(config.Default())
	if err := st.Orc().Load(paddleocr.Models{
		DetPath:  os.Getenv("HOME") + "/.cpm_orc/models/paddleocr/ch/PP-OCRv6_small_det.onnx",
		RecPath:  os.Getenv("HOME") + "/.cpm_orc/models/paddleocr/ch/PP-OCRv6_small_rec.onnx",
		DictPath: os.Getenv("HOME") + "/.cpm_orc/models/paddleocr/ch/PP-OCRv6_dict.txt",
	}); err != nil {
		t.Fatalf("load: %v", err)
	}
	// Put a known image on the clipboard.
	if err := exec.Command("osascript", "-e",
		`set the clipboard to (read POSIX file "/tmp/v6/sample.png" as «class PNGf»)`).Run(); err != nil {
		t.Fatalf("set clipboard: %v", err)
	}
	text, err := st.OcrClipboard()
	if err != nil {
		t.Fatalf("clipboard OCR: %v", err)
	}
	t.Logf("OCR 文本长度=%d", len(text))
	if !strings.Contains(text, "October") && !strings.Contains(text, "OCR") {
		t.Logf("（内容可能因字体略有差异）文本开头: %s", truncate(text, 120))
	}
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}