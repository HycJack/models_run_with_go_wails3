package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"

	"cpm_orc/internal/hfhub"
	"cpm_orc/internal/paddleocr"
)

// OcrStatus describes the current OCR engine configuration.
type OcrStatus struct {
	Loaded   bool   `json:"loaded"`
	DetPath  string `json:"detPath"`
	RecPath  string `json:"recPath"`
	ClsPath  string `json:"clsPath"`
	DictPath string `json:"dictPath"`
	Threads  int    `json:"threads"`
}

// OcrService manages PaddleOCR ONNX models and runs recognition.
type OcrService struct {
	state *State
}

// NewOcrService creates the OCR service.
func NewOcrService(s *State) *OcrService { return &OcrService{state: s} }

// Status returns the current OCR configuration.
func (s *OcrService) Status() OcrStatus {
	eng := s.state.Orc()
	return OcrStatus{
		Loaded: eng.Loaded(),
		Threads: 4,
	}
}

// Load opens the given ONNX model files. oriPath (document orientation
// classifier, optional) corrects rotated scans before recognition.
func (s *OcrService) Load(detPath, recPath, clsPath, oriPath, dictPath string) error {
	if err := s.state.EnsureOrt(); err != nil {
		return err
	}
	eng := s.state.Orc()
	return eng.Load(paddleocr.Models{
		DetPath:  detPath,
		RecPath:  recPath,
		ClsPath:  clsPath,
		OriPath:  oriPath,
		DictPath: dictPath,
	})
}

// SetThreads configures the ONNX thread count.
func (s *OcrService) SetThreads(n int) {
	s.state.Orc().SetThreads(n)
}

// Recognise runs OCR on an image file.
func (s *OcrService) Recognise(path string) (*paddleocr.Result, error) {
	return s.state.Orc().RecogniseFile(path)
}

// RecogniseBase64 runs OCR on base64-encoded image data.
func (s *OcrService) RecogniseBase64(data string) (*paddleocr.Result, error) {
	return s.state.Orc().RecogniseBase64(data)
}

// OcrClipboard OCRs the image currently on the system clipboard.
func (s *OcrService) OcrClipboard() (string, error) {
	return s.state.OcrClipboard()
}

// ScreenshotOcr interactively captures a screen region and OCRs it.
func (s *OcrService) ScreenshotOcr() (string, error) {
	return s.state.OcrScreenshot()
}

// DefaultModelDir returns where default OCR models are stored.
func (s *OcrService) DefaultModelDir() string {
	return filepath.Join(s.state.cfg.OcrDir, "ch")
}

// v6Repos are the official PP-OCRv6 ONNX model repos on HuggingFace.
const (
	v6RecRepo = "PaddlePaddle/PP-OCRv6_small_rec_onnx"
	oriRepo   = "monkt/paddleocr-onnx"
	oriFile   = "preprocessing/doc-orientation/PP-LCNet_x1_0_doc_ori.onnx"
)

// v6Tiers maps the three PP-OCRv6 model tiers to their HF repos and file
// prefix (also used for recognition, one per tier).
var v6Tiers = map[string]struct {
	detRepo string
	recRepo string
	prefix  string
}{
	"tiny": {
		detRepo: "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
		recRepo: "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
		prefix:  "PP-OCRv6_tiny",
	},
	"small": {
		detRepo: "PaddlePaddle/PP-OCRv6_small_det_onnx",
		recRepo: "PaddlePaddle/PP-OCRv6_small_rec_onnx",
		prefix:  "PP-OCRv6_small",
	},
	"medium": {
		detRepo: "PaddlePaddle/PP-OCRv6_medium_det_onnx",
		recRepo: "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
		prefix:  "PP-OCRv6_medium",
	},
}

// OCRTierNames returns the available PP-OCRv6 tiers.
func (s *OcrService) OCRTierNames() []string {
	return []string{"tiny", "small", "medium"}
}

func tierInfo(tier string) (detRepo, recRepo, prefix string, ok bool) {
	if tier == "" {
		tier = "small"
	}
	t, ok := v6Tiers[tier]
	return t.detRepo, t.recRepo, t.prefix, ok
}

// InstallDefaults downloads a set of PaddleOCR ONNX models. lang only selects
// a sub-folder name; tier is "tiny"/"small"/"medium" (PP-OCRv6 model tiers).
func (s *OcrService) InstallDefaults(lang, tier string) error {
	if lang == "" {
		lang = "ch"
	}
	detRepo, recRepo, prefix, ok := tierInfo(tier)
	if !ok {
		return fmt.Errorf("未知的模型档位 %q（可选 tiny/small/medium）", tier)
	}
	client := hfhub.NewClient(s.state.cfg.Proxy)
	base := filepath.Join(s.state.cfg.OcrDir, lang)
	if err := os.MkdirAll(base, 0o755); err != nil {
		return err
	}
	det := filepath.Join(base, prefix+"_det.onnx")
	rec := filepath.Join(base, prefix+"_rec.onnx")
	ori := filepath.Join(base, "PP-LCNet_x1_0_doc_ori.onnx")
	dict := filepath.Join(base, "PP-OCRv6_dict.txt")

	download := func(repo, remote, target string) error {
		if _, err := os.Stat(target); err == nil {
			return nil
		}
		s.state.Emit("dl:start", map[string]any{"id": "paddleocr-" + lang, "file": filepath.Base(target)})
		err := client.Download(repo, "main", remote, target, func(done, total int64) {
			s.state.Emit("dl:progress", map[string]any{
				"id":    "paddleocr-" + lang,
				"file":  filepath.Base(target),
				"done":  done,
				"total": total,
			})
		})
		if err != nil {
			return fmt.Errorf("download %s: %w", remote, err)
		}
		s.state.Emit("dl:file-done", map[string]any{"id": "paddleocr-" + lang, "file": filepath.Base(target)})
		return nil
	}

	if err := download(detRepo, "inference.onnx", det); err != nil {
		return err
	}
	if err := download(recRepo, "inference.onnx", rec); err != nil {
		return err
	}
	if err := download(oriRepo, oriFile, ori); err != nil {
		return err
	}
	// Generate the character dictionary from the recognition pipeline config.
	if _, err := os.Stat(dict); err != nil {
		yml := filepath.Join(base, "_rec.yml")
		if err := download(v6RecRepo, "inference.yml", yml); err != nil {
			return err
		}
		defer os.Remove(yml)
		var cfg struct {
			PostProcess struct {
				CharacterDict []string `yaml:"character_dict"`
			} `yaml:"PostProcess"`
		}
		data, err := os.ReadFile(yml)
		if err != nil {
			return err
		}
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return err
		}
		if len(cfg.PostProcess.CharacterDict) == 0 {
			return fmt.Errorf("no character_dict found in inference.yml")
		}
		var sb strings.Builder
		for _, c := range cfg.PostProcess.CharacterDict {
			sb.WriteString(c)
			sb.WriteString("\n")
		}
		if err := os.WriteFile(dict, []byte(sb.String()), 0o644); err != nil {
			return err
		}
		s.state.Emit("dl:file-done", map[string]any{"id": "paddleocr-" + lang, "file": "PP-OCRv6_dict.txt"})
	}
	s.state.Emit("dl:done", map[string]any{"id": "paddleocr-" + lang, "path": base})
	return nil
}

// DefaultPaths returns the expected default OCR model paths for a language and
// tier.
func (s *OcrService) DefaultPaths(lang, tier string) (det, rec, cls, ori, dict string) {
	if lang == "" {
		lang = "ch"
	}
	_, _, prefix, ok := tierInfo(tier)
	if !ok {
		prefix = "PP-OCRv6_small"
	}
	base := filepath.Join(s.state.cfg.OcrDir, lang)
	return filepath.Join(base, prefix+"_det.onnx"),
		filepath.Join(base, prefix+"_rec.onnx"),
		"",
		filepath.Join(base, "PP-LCNet_x1_0_doc_ori.onnx"),
		filepath.Join(base, "PP-OCRv6_dict.txt")
}

// GuessPathsFromDir attempts to locate det/rec/cls/ori/dict files inside a
// directory by filename heuristics.
func (s *OcrService) GuessPathsFromDir(dir string) (det, rec, cls, ori, dict string, err error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", "", "", "", "", err
	}
	detFound := false
	recFound := false
	for _, e := range entries {
		name := strings.ToLower(e.Name())
		switch {
		case strings.Contains(name, "det") && !strings.Contains(name, "ori"):
			det = filepath.Join(dir, e.Name())
			detFound = true
		case strings.Contains(name, "rec"):
			rec = filepath.Join(dir, e.Name())
			recFound = true
		case strings.Contains(name, "cls"):
			cls = filepath.Join(dir, e.Name())
		case strings.Contains(name, "doc_ori") || strings.Contains(name, "docori"):
			ori = filepath.Join(dir, e.Name())
		case strings.Contains(name, "dict") || strings.HasSuffix(name, ".txt"):
			dict = filepath.Join(dir, e.Name())
		}
	}
	// PP-OCRv6 repos ship both models as inference.onnx; disambiguate by size
	// or by the corresponding inference.yml.
	if !detFound || !recFound {
		for _, e := range entries {
			if strings.EqualFold(e.Name(), "inference.onnx") {
				p := filepath.Join(dir, e.Name())
				info, err := e.Info()
				if err != nil {
					continue
				}
				if !detFound && info.Size() < 20_000_000 {
					det = p
					detFound = true
				} else if !recFound {
					rec = p
					recFound = true
				}
			}
		}
	}
	if dict == "" {
		for _, e := range entries {
			if strings.Contains(strings.ToLower(e.Name()), "dict") {
				dict = filepath.Join(dir, e.Name())
				break
			}
		}
	}
	if !detFound || !recFound {
		return "", "", "", "", "", fmt.Errorf("could not locate detection/recognition ONNX files in %s", dir)
	}
	return det, rec, cls, ori, dict, nil
}