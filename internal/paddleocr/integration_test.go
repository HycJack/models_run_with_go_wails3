package paddleocr

import (
	"image"
	"image/color"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golang/freetype"
	"github.com/golang/freetype/truetype"
	"gopkg.in/yaml.v3"
	"cpm_orc/internal/hfhub"
	"cpm_orc/internal/ort"
)

// TestRecogniseEndToEnd verifies the full OCR pipeline against real models.
// It only runs when CPMM_OCR_TEST=1 is set (downloads ~15MB of models plus a
// generated test image) and CPMM_ORT_LIB points at a local onnxruntime lib.
// installedModels returns the paths of the pre-installed OCR models, or empty
// when not present (test will download fresh copies instead).
func installedModels(t *testing.T) (det, rec, dict string, ok bool) {
	home, _ := os.UserHomeDir()
	base := filepath.Join(home, ".cpm_orc/models/paddleocr/ch")
	det = filepath.Join(base, "PP-OCRv6_small_det.onnx")
	rec = filepath.Join(base, "PP-OCRv6_small_rec.onnx")
	dict = filepath.Join(base, "PP-OCRv6_dict.txt")
	if _, e1 := os.Stat(det); e1 == nil {
		if _, e2 := os.Stat(rec); e2 == nil {
			if _, e3 := os.Stat(dict); e3 == nil {
				return det, rec, dict, true
			}
		}
	}
	return "", "", "", false
}

func TestRecogniseEndToEnd(t *testing.T) {
	if os.Getenv("CPMM_OCR_TEST") != "1" {
		t.Skip("set CPMM_OCR_TEST=1 to run the OCR integration test")
	}
	if err := ort.Init(os.Getenv("CPMM_ORT_LIB")); err != nil {
		t.Fatalf("ort init: %v", err)
	}
	var det, rec, dict string
	if d, r, dc, ok := installedModels(t); ok {
		det, rec, dict = d, r, dc
		t.Log("使用已安装的模型")
	} else {
		dir := t.TempDir()
		client := hfhub.NewClient("")
		det = filepath.Join(dir, "det.onnx")
		rec = filepath.Join(dir, "rec.onnx")
		dict = filepath.Join(dir, "dict.txt")
		if _, err := os.Stat(det); err != nil {
			if err := client.Download("PaddlePaddle/PP-OCRv6_small_det_onnx", "main", "inference.onnx", det, nil); err != nil {
				t.Fatalf("download det: %v", err)
			}
		}
		if _, err := os.Stat(rec); err != nil {
			if err := client.Download("PaddlePaddle/PP-OCRv6_small_rec_onnx", "main", "inference.onnx", rec, nil); err != nil {
				t.Fatalf("download rec: %v", err)
			}
		}
		if _, err := os.Stat(dict); err != nil {
			yml := filepath.Join(dir, "rec.yml")
			if err := client.Download("PaddlePaddle/PP-OCRv6_small_rec_onnx", "main", "inference.yml", yml, nil); err != nil {
				t.Fatalf("download yml: %v", err)
			}
			defer os.Remove(yml)
			var cfg struct {
				PostProcess struct {
					CharacterDict []string `yaml:"character_dict"`
				} `yaml:"PostProcess"`
			}
			data, err := os.ReadFile(yml)
			if err != nil {
				t.Fatal(err)
			}
			if err := yaml.Unmarshal(data, &cfg); err != nil {
				t.Fatal(err)
			}
			var sb strings.Builder
			for _, c := range cfg.PostProcess.CharacterDict {
				sb.WriteString(c + "\n")
			}
			if err := os.WriteFile(dict, []byte(sb.String()), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	img := makeTestImage(t, 900, 320)
	eng := NewEngine()
	if err := eng.Load(Models{
		DetPath:  det,
		RecPath:  rec,
		DictPath: dict,
	}); err != nil {
		t.Fatalf("load: %v", err)
	}
	defer eng.Close()
	res, err := eng.Recognise(img)
	if err != nil {
		t.Fatalf("recognise: %v", err)
	}
	t.Logf("detected %d lines in %.0fms", len(res.Lines), res.Elapsed)
	if len(res.Lines) == 0 {
		t.Fatal("expected at least one text line")
	}
	for _, l := range res.Lines {
		t.Logf("  [%.2f] %q", l.Confidence, l.Text)
	}
}

// makeTestImage renders a few lines of English text onto a white image using
// a system TTF font.
func makeTestImage(t *testing.T, w, h int) *Image {
	t.Helper()
	fontBytes, err := os.ReadFile("/System/Library/Fonts/Supplemental/Arial.ttf")
	if err != nil {
		t.Skipf("no system font available: %v", err)
	}
	f, err := truetype.Parse(fontBytes)
	if err != nil {
		t.Fatal(err)
	}
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			rgba.SetRGBA(x, y, color.RGBA{250, 250, 252, 255})
		}
	}
	ctx := freetype.NewContext()
	ctx.SetDPI(120)
	ctx.SetFont(f)
	ctx.SetClip(rgba.Bounds())
	ctx.SetDst(rgba)
	ctx.SetSrc(image.NewUniform(color.RGBA{20, 20, 30, 255}))
	y := 70
	for _, ln := range []string{"Hello OCR 12345", "OpenAI GPT models", "PP-OCRv4 ONNX Runtime"} {
		ctx.SetFontSize(44)
		_, err := ctx.DrawString(ln, freetype.Pt(40, y))
		if err != nil {
			t.Fatal(err)
		}
		y += 90
	}
	return FromImage(rgba)
}
// TestDocOrientation verifies the document orientation classifier corrects a
// 90°-rotated image so the same lines are detected as in the upright image.
func TestDocOrientation(t *testing.T) {
	if os.Getenv("CPMM_OCR_TEST") != "1" {
		t.Skip("set CPMM_OCR_TEST=1 to run the OCR integration test")
	}
	if err := ort.Init(os.Getenv("CPMM_ORT_LIB")); err != nil {
		t.Fatalf("ort init: %v", err)
	}
	home, _ := os.UserHomeDir()
	eng := NewEngine()
	if err := eng.Load(Models{
		DetPath:  home + "/.cpm_orc/models/paddleocr/ch/PP-OCRv6_small_det.onnx",
		RecPath:  home + "/.cpm_orc/models/paddleocr/ch/PP-OCRv6_small_rec.onnx",
		OriPath:  home + "/.cpm_orc/models/paddleocr/ch/PP-LCNet_x1_0_doc_ori.onnx",
		DictPath: home + "/.cpm_orc/models/paddleocr/ch/PP-OCRv6_dict.txt",
	}); err != nil {
		t.Fatalf("load: %v", err)
	}
	defer eng.Close()

	upright := makeTestImage(t, 900, 320)
	rotated := upright.Rotate90CW()
	res, err := eng.Recognise(rotated)
	if err != nil {
		t.Fatalf("recognise rotated: %v", err)
	}
	t.Logf("rotated image: rotation=%d, lines=%d", res.Rotation, len(res.Lines))
	if res.Rotation != 270 {
		t.Logf("注意: 期望矫正旋转 270（把 90°CW 转回正），实际 %d", res.Rotation)
	}
	if len(res.Lines) == 0 {
		t.Fatal("矫正后仍应识别出文本行")
	}
	for _, l := range res.Lines {
		t.Logf("  [%.2f] %q", l.Confidence, l.Text)
	}
}
