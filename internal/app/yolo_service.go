package app

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"

	"cpm_orc/internal/hfhub"
	"cpm_orc/internal/yolo"
)

// YoloService exposes YOLOv8 object detection to the frontend.
type YoloService struct {
	state *State
}

// NewYoloService creates the service.
func NewYoloService(s *State) *YoloService {
	return &YoloService{state: s}
}

// Load loads a YOLOv8 ONNX model from the given path.
func (s *YoloService) Load(modelPath string) error {
	if err := s.state.EnsureOrt(); err != nil {
		return err
	}
	return s.state.Yolo().Load(modelPath)
}

// Detect runs object detection on a base64-encoded image.
// Returns bounding boxes with class labels and confidence scores.
func (s *YoloService) Detect(imgB64 string, confThresh float64, iouThresh float64) ([]yolo.Detection, error) {
	if !s.state.Yolo().IsLoaded() {
		return nil, fmt.Errorf("YOLO model not loaded")
	}
	img, err := decodeBase64Image(imgB64)
	if err != nil {
		return nil, err
	}
	return s.state.Yolo().Detect(img, confThresh, iouThresh)
}

// DetectFile runs detection on a local image file.
func (s *YoloService) DetectFile(path string, confThresh float64, iouThresh float64) ([]yolo.Detection, error) {
	if !s.state.Yolo().IsLoaded() {
		return nil, fmt.Errorf("YOLO model not loaded")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	return s.state.Yolo().Detect(img, confThresh, iouThresh)
}

// SetClassNames sets human-readable class labels for detections.
func (s *YoloService) SetClassNames(names []string) {
	s.state.Yolo().SetClassNames(names)
}

// IsLoaded reports whether a YOLO model is loaded.
func (s *YoloService) IsLoaded() bool {
	return s.state.Yolo().IsLoaded()
}

// Close releases the current model.
func (s *YoloService) Close() {
	s.state.Yolo().Close()
}

// ModelDir returns the default YOLO model directory.
func (s *YoloService) ModelDir() string {
	return s.state.cfg.YoloDir
}

// ListLocalModels lists .onnx files in the YOLO model directory.
func (s *YoloService) ListLocalModels() []string {
	dir := s.state.cfg.YoloDir
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var models []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".onnx" {
			models = append(models, filepath.Join(dir, e.Name()))
		}
	}
	return models
}

// DownloadModel downloads a YOLO ONNX model file from a HuggingFace repo into
// the YOLO model directory. hfPath is the path inside the repo (e.g.
// "onnx/model.onnx"); saveAs is the local filename.
func (s *YoloService) DownloadModel(repoID, hfPath, saveAs string) (string, error) {
	if hfPath == "" {
		hfPath = "onnx/model.onnx"
	}
	if saveAs == "" {
		saveAs = "model.onnx"
	}
	if err := os.MkdirAll(s.state.cfg.YoloDir, 0o755); err != nil {
		return "", err
	}
	dest := filepath.Join(s.state.cfg.YoloDir, saveAs)
	client := hfhub.NewClient(s.state.cfg.Proxy)

	s.state.Emit("dl:start", map[string]any{"id": repoID, "file": saveAs})
	err := client.Download(repoID, "main", hfPath, dest, func(done, total int64) {
		s.state.Emit("dl:progress", map[string]any{
			"id": repoID, "file": saveAs, "done": done, "total": total,
		})
	})
	if err != nil {
		return "", err
	}
	// config.json carries id2label, used for human-readable class names.
	cfgDest := filepath.Join(s.state.cfg.YoloDir, strings.TrimSuffix(saveAs, ".onnx")+".config.json")
	_ = client.Download(repoID, "main", "config.json", cfgDest, nil)
	s.state.Emit("dl:file-done", map[string]any{"id": repoID, "file": saveAs})
	return dest, nil
}

// DownloadYOLO26 downloads a YOLO26 ONNX model from the onnx-community mirror.
// scale is one of n, s, m, l, x. The official Ultralytics repo ships only .pt
// weights, so the community ONNX exports are used instead.
func (s *YoloService) DownloadYOLO26(scale string) (string, error) {
	switch scale {
	case "n", "s", "m", "l", "x":
	case "":
		scale = "n"
	default:
		return "", fmt.Errorf("unknown YOLO26 scale %q (want n/s/m/l/x)", scale)
	}
	repo := fmt.Sprintf("onnx-community/yolo26%s-ONNX", scale)
	return s.DownloadModel(repo, "onnx/model.onnx", fmt.Sprintf("yolo26%s.onnx", scale))
}

// PresetModels lists the recommended YOLO26 variants for one-click download.
func (s *YoloService) PresetModels() []map[string]any {
	return []map[string]any{
		{"id": "yolo26n", "scale": "n", "size": "~9MB", "desc": "最快，适合实时检测"},
		{"id": "yolo26s", "scale": "s", "size": "~35MB", "desc": "速度/精度平衡"},
		{"id": "yolo26m", "scale": "m", "size": "~75MB", "desc": "中等精度"},
		{"id": "yolo26l", "scale": "l", "size": "~95MB", "desc": "高精度"},
		{"id": "yolo26x", "scale": "x", "size": "~210MB", "desc": "最高精度"},
	}
}

// decodeBase64Image decodes a raw base64 string (no data-URI prefix) into an
// image.Image. If the string starts with "data:", it strips the prefix.
func decodeBase64Image(b64 string) (image.Image, error) {
	// Strip data-URI prefix if present
	if len(b64) > 22 && b64[:22] == "data:image/" {
		// find comma
		for i := 22; i < len(b64); i++ {
			if b64[i] == ',' {
				b64 = b64[i+1:]
				break
			}
		}
	}

	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		// try URL-safe encoding
		raw, err = base64.RawURLEncoding.DecodeString(b64)
		if err != nil {
			return nil, fmt.Errorf("decode base64: %w", err)
		}
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}
