package yolo

import (
	"encoding/json"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"cpm_orc/internal/onnxmeta"
	"cpm_orc/internal/ort"
)

// Detection holds a single YOLO detection result.
type Detection struct {
	Box   [4]float64 `json:"box"`   // x0, y0, x1, y1 in original image pixels
	Class int        `json:"class"` // class index
	Score float64    `json:"score"` // confidence 0..1
	Label string     `json:"label"` // class name when known
}

// Engine runs a YOLO ONNX model for object detection.
type Engine struct {
	mu      sync.Mutex
	session *ort.Session
	io      *onnxmeta.ModelIO
	loaded  bool
	threads int
	inputW  int
	inputH  int
	names   []string
}

// NewEngine creates a YOLO engine. Call Load before Detect.
func NewEngine(threads int) *Engine {
	return &Engine{threads: threads, inputW: 640, inputH: 640}
}

// Load loads a YOLO ONNX model. When a config.json sits next to the model (or
// in its parent directory, as with the onnx-community layout), its id2label map
// is used for human-readable labels.
func (e *Engine) Load(modelPath string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	io, err := onnxmeta.Parse(modelPath)
	if err != nil {
		return fmt.Errorf("parse model metadata: %w", err)
	}
	if len(io.Inputs) == 0 || len(io.Outputs) < 2 {
		return fmt.Errorf("expected 1 input and 2 outputs (logits, pred_boxes), got %v -> %v", io.Inputs, io.Outputs)
	}

	sess, err := ort.NewSession(modelPath, io.Inputs, io.Outputs, e.threads)
	if err != nil {
		return fmt.Errorf("create session: %w", err)
	}

	if e.session != nil {
		e.session.Destroy()
	}
	e.session = sess
	e.io = io
	e.loaded = true
	if names := loadLabels(modelPath); len(names) > 0 {
		e.names = names
	}
	return nil
}

// Detect runs inference and returns detections filtered by confidence and
// de-duplicated by per-class NMS.
func (e *Engine) Detect(img image.Image, confThresh, iouThresh float64) ([]Detection, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.loaded || e.session == nil {
		return nil, fmt.Errorf("model not loaded")
	}
	if confThresh <= 0 {
		confThresh = 0.25
	}
	if iouThresh <= 0 {
		iouThresh = 0.45
	}

	input, origW, origH, scale, padX, padY := preprocess(img, e.inputW, e.inputH)

	tensor, err := ort.NewTensor([]int64{1, 3, int64(e.inputH), int64(e.inputW)}, input)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer tensor.Destroy()

	outputs := make([]ort.Value, len(e.io.Outputs))
	if err := e.session.Run([]ort.Value{tensor}, outputs); err != nil {
		return nil, fmt.Errorf("run inference: %w", err)
	}

	// Resolve outputs by name so ordering changes do not break decoding.
	var logitsT, boxesT *ort.Tensor[float32]
	for i, name := range e.io.Outputs {
		t, ok := outputs[i].(*ort.Tensor[float32])
		if !ok {
			continue
		}
		defer t.Destroy()
		switch name {
		case "logits":
			logitsT = t
		case "pred_boxes":
			boxesT = t
		}
	}
	if logitsT == nil || boxesT == nil {
		return nil, fmt.Errorf("model outputs %v missing logits/pred_boxes", e.io.Outputs)
	}

	dets := postprocessDualHead(
		logitsT.GetData(), logitsT.GetShape(),
		boxesT.GetData(), boxesT.GetShape(),
		origW, origH, scale, padX, padY, e.inputW, e.inputH,
		e.names,
	)

	filtered := make([]Detection, 0, len(dets))
	for _, d := range dets {
		if d.Score >= confThresh {
			filtered = append(filtered, d)
		}
	}
	return nms(filtered, iouThresh), nil
}

// Close releases the session.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}
	e.loaded = false
}

// IsLoaded reports whether a model is loaded.
func (e *Engine) IsLoaded() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.loaded
}

// SetClassNames overrides the class label list.
func (e *Engine) SetClassNames(names []string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.names = names
}

// loadLabels reads id2label from a sidecar config. It looks for
// "<model>.config.json" next to the model (what DownloadModel writes), then a
// plain config.json beside the model or one level up (the onnx-community layout
// keeps the model under onnx/ and config.json at the repo root).
func loadLabels(modelPath string) []string {
	dir := filepath.Dir(modelPath)
	base := strings.TrimSuffix(filepath.Base(modelPath), ".onnx")
	for _, candidate := range []string{
		filepath.Join(dir, base+".config.json"),
		filepath.Join(dir, "config.json"),
		filepath.Join(filepath.Dir(dir), "config.json"),
	} {
		raw, err := os.ReadFile(candidate)
		if err != nil {
			continue
		}
		var cfg struct {
			ID2Label map[string]string `json:"id2label"`
		}
		if json.Unmarshal(raw, &cfg) != nil || len(cfg.ID2Label) == 0 {
			continue
		}
		type kv struct {
			id   int
			name string
		}
		pairs := make([]kv, 0, len(cfg.ID2Label))
		for k, v := range cfg.ID2Label {
			id, err := strconv.Atoi(k)
			if err != nil {
				continue
			}
			pairs = append(pairs, kv{id, v})
		}
		sort.Slice(pairs, func(i, j int) bool { return pairs[i].id < pairs[j].id })
		names := make([]string, 0, len(pairs))
		for i, p := range pairs {
			if p.id != i {
				return nil // non-contiguous ids; index mapping would be wrong
			}
			names = append(names, p.name)
		}
		return names
	}
	return nil
}
