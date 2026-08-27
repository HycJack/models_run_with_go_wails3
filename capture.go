package main

import (
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// saveClipboardImage writes the image currently on the clipboard to a PNG file.
func saveClipboardImage(path string) error {
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("clipboard image OCR is not yet supported on %s", runtime.GOOS)
	}
	script := fmt.Sprintf(
		"set d to (the clipboard as «class PNGf»)\n"+
			"set f to open for access POSIX file %q with write permission\n"+
			"write d to f\n"+
			"close access f", path)
	return exec.Command("osascript", "-e", script).Run()
}

// captureScreenSelection interactively lets the user select a screen region and
// saves it as a PNG.
func captureScreenSelection(path string) error {
	switch runtime.GOOS {
	case "darwin":
		return exec.Command("screencapture", "-i", path).Run()
	case "windows":
		// PowerShell + System.Drawing region capture is intentionally avoided;
		// use the built-in snipping tool instead.
		return fmt.Errorf("screen capture OCR is not yet supported on Windows")
	default:
		return fmt.Errorf("screen capture OCR is not yet supported on %s", runtime.GOOS)
	}
}

// OcrClipboard OCRs the current clipboard image and reports the result.
func (s *State) OcrClipboard() (string, error) {
	tmp := filepath.Join(os.TempDir(), "cpm-clipboard.png")
	os.Remove(tmp)
	if err := saveClipboardImage(tmp); err != nil {
		return "", fmt.Errorf("读取剪贴板图片失败: %w", err)
	}
	return s.recognizeAndReport(tmp, "clipboard")
}

// OcrScreenshot interactively captures a screen region and OCRs it.
func (s *State) OcrScreenshot() (string, error) {
	tmp := filepath.Join(os.TempDir(), "cpm-screenshot.png")
	os.Remove(tmp)
	if err := captureScreenSelection(tmp); err != nil {
		return "", fmt.Errorf("截图失败: %w", err)
	}
	return s.recognizeAndReport(tmp, "screenshot")
}

// recognizeAndReport runs the OCR engine on an image and pushes the result to
// the frontend via events, focusing the main window when done.
func (s *State) recognizeAndReport(path, source string) (string, error) {
	s.Emit("ocr:busy", map[string]any{"source": source})
	res, err := s.Orc().RecogniseFile(path)
	if err != nil {
		s.Emit("ocr:error", map[string]any{"source": source, "error": err.Error()})
		return "", err
	}
	var texts []string
	for _, l := range res.Lines {
		texts = append(texts, l.Text)
	}
	text := strings.Join(texts, "\n")
	// Include the source image so the frontend can preview it and render the
	// reconstructed (boxed) image.
	var imageB64 string
	if data, err := os.ReadFile(path); err == nil {
		imageB64 = base64.StdEncoding.EncodeToString(data)
	}
	s.Emit("ocr:result", map[string]any{
		"source": source,
		"text":   text,
		"lines":  res.Lines,
		"image":  imageB64,
	})
	s.ShowMainWindow()
	return text, nil
}