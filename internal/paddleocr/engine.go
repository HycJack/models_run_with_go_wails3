package paddleocr

import (
	"encoding/base64"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"time"

	"cpm_orc/internal/onnxmeta"
	"cpm_orc/internal/ort"
)

// debugOCR enables stderr diagnostics for recognition.
var debugOCR = os.Getenv("OCR_DEBUG") != ""

// EnableCoreML registers the CoreML execution provider for OCR sessions on
// macOS when set.
var EnableCoreML = true

// Models groups the PaddleOCR ONNX model paths plus the dictionary.
type Models struct {
	DetPath  string `json:"detPath"`
	RecPath  string `json:"recPath"`
	ClsPath  string `json:"clsPath"` // optional text-line orientation (180°)
	OriPath  string `json:"oriPath"` // optional document orientation (0/90/180/270°)
	DictPath string `json:"dictPath"`
}

// Line is a recognized text line.
type Line struct {
	Text       string        `json:"text"`
	Confidence float32       `json:"confidence"`
	Box        [4][2]float64 `json:"box"` // box in the (possibly corrected) image pixels
}

// Result is the complete OCR output for one image.
type Result struct {
	Lines   []Line  `json:"lines"`
	Elapsed float64 `json:"elapsedMs"`
	// Rotation is the clockwise degrees applied to the input to correct its
	// document orientation (0/90/180/270). Boxes are in the corrected space.
	Rotation int `json:"rotation"`
}

// Engine runs PaddleOCR ONNX models through ONNX Runtime.
type Engine struct {
	mu    sync.Mutex
	det   *ort.Session
	rec   *ort.Session
	cls   *ort.Session
	ori   *ort.Session
	dict  *Dict
	detIO *onnxmeta.ModelIO
	recIO *onnxmeta.ModelIO
	clsIO *onnxmeta.ModelIO
	oriIO *onnxmeta.ModelIO

	detParams DetParams
	recParams RecParams
	clsParams ClsParams
	oriParams OriParams

	// BGR marks that the ONNX models expect BGR channel order (PaddleOCR's
	// OpenCV-based preprocessing). Defaults to RGB (false).
	bgr     bool
	loaded  bool
	threads int
}

// NewEngine returns an empty engine.
func NewEngine() *Engine {
	return &Engine{
		detParams: DefaultDetParams(),
		recParams: DefaultRecParams(),
		clsParams: DefaultClsParams(),
		oriParams: DefaultOriParams(),
		threads:   4,
	}
}

// SetBGR selects the input channel order used by the loaded models.
func (e *Engine) SetBGR(bgr bool) { e.bgr = bgr }

// BGR reports the configured channel order.
func (e *Engine) BGR() bool { return e.bgr }

// SetDetParams overrides the detection post-processing parameters.
func (e *Engine) SetDetParams(p DetParams) { e.detParams = p }

// SetRecParams overrides the recognition pre-processing parameters.
func (e *Engine) SetRecParams(p RecParams) { e.recParams = p }

// Loaded reports whether the engine has loaded models.
func (e *Engine) Loaded() bool { return e.loaded }

// SetThreads sets the per-session thread count (default 4).
func (e *Engine) SetThreads(n int) { e.threads = n }

// Load opens the given models. Det and rec are required; cls and dict are
// optional.
func (e *Engine) Load(m Models) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	detIO, err := onnxmeta.Parse(m.DetPath)
	if err != nil {
		return fmt.Errorf("det model: %w", err)
	}
	recIO, err := onnxmeta.Parse(m.RecPath)
	if err != nil {
		return fmt.Errorf("rec model: %w", err)
	}
	// Auto-configure the recognition input height from the ONNX graph so both
	// PP-OCRv3 (32px) and PP-OCRv4 (48px) rec models work.
	if sh := recIO.InputShape(recIO.Inputs[0]); len(sh) >= 3 && sh[2] > 0 {
		e.recParams.ImgH = int(sh[2])
	}
	det, err := ort.NewSession(m.DetPath, detIO.Inputs, detIO.Outputs, e.threads, EnableCoreML)
	if err != nil {
		return fmt.Errorf("load det session: %w", err)
	}
	rec, err := ort.NewSession(m.RecPath, recIO.Inputs, recIO.Outputs, e.threads, EnableCoreML)
	if err != nil {
		det.Destroy()
		return fmt.Errorf("load rec session: %w", err)
	}
	var dict *Dict
	if m.DictPath != "" {
		dict, err = LoadDict(m.DictPath)
		if err != nil {
			det.Destroy()
			rec.Destroy()
			return fmt.Errorf("load dict: %w", err)
		}
	}
	var cls *ort.Session
	var clsIO *onnxmeta.ModelIO
	if m.ClsPath != "" {
		clsIO, err = onnxmeta.Parse(m.ClsPath)
		if err != nil {
			det.Destroy()
			rec.Destroy()
			return fmt.Errorf("cls model: %w", err)
		}
		cls, err = ort.NewSession(m.ClsPath, clsIO.Inputs, clsIO.Outputs, e.threads, EnableCoreML)
		if err != nil {
			det.Destroy()
			rec.Destroy()
			return fmt.Errorf("load cls session: %w", err)
		}
	}
	var ori *ort.Session
	var oriIO *onnxmeta.ModelIO
	if m.OriPath != "" {
		oriIO, err = onnxmeta.Parse(m.OriPath)
		if err != nil {
			det.Destroy()
			rec.Destroy()
			if cls != nil {
				cls.Destroy()
			}
			return fmt.Errorf("doc orientation model: %w", err)
		}
		ori, err = ort.NewSession(m.OriPath, oriIO.Inputs, oriIO.Outputs, e.threads, EnableCoreML)
		if err != nil {
			det.Destroy()
			rec.Destroy()
			if cls != nil {
				cls.Destroy()
			}
			return fmt.Errorf("load doc orientation session: %w", err)
		}
	}

	e.destroyLocked()
	e.det = det
	e.rec = rec
	e.cls = cls
	e.ori = ori
	e.dict = dict
	e.detIO = detIO
	e.recIO = recIO
	e.clsIO = clsIO
	e.oriIO = oriIO
	e.loaded = true
	return nil
}

func (e *Engine) destroyLocked() {
	for _, s := range []*ort.Session{e.det, e.rec, e.cls, e.ori} {
		if s != nil {
			s.Destroy()
		}
	}
	e.det, e.rec, e.cls, e.ori = nil, nil, nil, nil
}

// Close releases all sessions.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.destroyLocked()
	e.loaded = false
	return nil
}

// RecogniseFile runs OCR on an image file.
func (e *Engine) RecogniseFile(path string) (*Result, error) {
	img, err := LoadImage(path)
	if err != nil {
		return nil, err
	}
	return e.Recognise(img)
}

// RecogniseBase64 runs OCR on base64-encoded image data.
func (e *Engine) RecogniseBase64(data string) (*Result, error) {
	raw, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, err
	}
	tmp, err := os.CreateTemp("", "ocr-*.png")
	if err != nil {
		return nil, err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(raw); err != nil {
		tmp.Close()
		return nil, err
	}
	tmp.Close()
	return e.RecogniseFile(tmpPath)
}

// Recognise runs the full detection + classification + recognition pipeline.
func (e *Engine) Recognise(img *Image) (*Result, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.loaded || e.det == nil || e.rec == nil {
		return nil, fmt.Errorf("OCR engine is not loaded")
	}
	if e.dict == nil {
		return nil, fmt.Errorf("recognition dictionary is not configured")
	}
	start := nowMs()

	// 0. Optional document orientation correction (0/90/180/270).
	res := &Result{}
	if e.ori != nil {
		ori, err := e.runOri(img)
		if err == nil && ori.Score > 0.5 {
			correct := ori.correctCW()
			res.Rotation = correct
			if correct != 0 {
				img = img.RotateCW(correct)
			}
		}
	}

	// 1. Text detection.
	sx, sy, prob, tw, th, err := e.runDet(img)
	if err != nil {
		return nil, fmt.Errorf("detection: %w", err)
	}
	detRes := DetectPostprocess(prob, tw, th, e.detParams)

	for _, box := range detRes {
		// Scale box back to original image coordinates.
		for i := 0; i < 4; i++ {
			box.Box[i][0] *= sx
			box.Box[i][1] *= sy
		}
		crop := img.CropBox(box.Box, cropW(box), cropH(box))
		if crop == nil {
			continue
		}
		// 2. Optional angle classification.
		if e.cls != nil {
			cls, err := e.runCls(crop)
			if err == nil && cls.Rotated {
				crop = crop.Rotate180()
			}
		}
		// 3. Recognition.
		text, conf, err := e.runRec(crop)
		if err != nil {
			if debugOCR {
				fmt.Fprintf(os.Stderr, "ocr: rec error: %v\n", err)
			}
			continue
		}
		if text == "" {
			if debugOCR {
				fmt.Fprintf(os.Stderr, "ocr: empty text for box %v\n", box.Box)
			}
			continue
		}
		res.Lines = append(res.Lines, Line{
			Text:       text,
			Confidence: conf,
			Box:        box.Box,
		})
	}
	sortLines(res.Lines)
	res.Elapsed = nowMs() - start
	return res, nil
}

// runDet preprocesses and runs the detection model, returning the per-axis
// scale factors (original/resized) plus the raw probability map dimensions.
func (e *Engine) runDet(img *Image) (sx, sy float64, prob []float32, ow, oh int, err error) {
	side := e.detParams.LimitSideLen
	w, h := img.W, img.H
	ratio := float64(side) / float64(max(w, h))
	if ratio > 1 {
		ratio = 1
	}
	tw := int(math.Round(float64(w) * ratio))
	th := int(math.Round(float64(h) * ratio))
	tw = max(tw, 1)
	th = max(th, 1)
	// The detection network has a stride of 32; dimensions must be multiples
	// of 32 or the internal broadcasting fails.
	tw = ((tw + 31) / 32) * 32
	th = ((th + 31) / 32) * 32
	sx = float64(w) / float64(tw)
	sy = float64(h) / float64(th)
	resized := img.Resize(tw, th)
	inp := resized.ToFloatCHW(1.0/255.0, e.detParams.Mean, e.detParams.Std)
	if e.bgr {
		inp = resized.ToFloatCHWBGR(1.0/255.0, e.detParams.Mean, e.detParams.Std)
	}

	inputTensor, err := ort.NewTensor([]int64{1, 3, int64(th), int64(tw)}, inp)
	if err != nil {
		return 0, 0, nil, 0, 0, err
	}
	defer inputTensor.Destroy()

	// Auto-allocate the output so ONNX Runtime sets the correct shape.
	outputs := []ort.Value{nil}
	if err := e.det.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return 0, 0, nil, 0, 0, err
	}
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return 0, 0, nil, 0, 0, fmt.Errorf("unexpected det output type")
	}
	defer outTensor.Destroy()
	shape := outTensor.GetShape()
	ow = 1
	oh = 1
	if len(shape) >= 4 {
		ow = int(shape[3])
		oh = int(shape[2])
	}
	return sx, sy, outTensor.GetData(), ow, oh, nil
}

// runRec resizes a crop and runs the recognition model.
func (e *Engine) runRec(crop *Image) (string, float32, error) {
	imgH := e.recParams.ImgH
	maxW := e.recParams.MaxWidth
	ratio := float64(crop.W) / float64(max(crop.H, 1))
	targetW := int(math.Ceil(float64(imgH) * ratio))
	if targetW > maxW {
		targetW = maxW
	}
	targetW = max(targetW, 4)
	resized := crop.Resize(targetW, imgH)
	inp := resized.ToFloatCHW(1.0/255.0, e.recParams.Mean, e.recParams.Std)
	if e.bgr {
		inp = resized.ToFloatCHWBGR(1.0/255.0, e.recParams.Mean, e.recParams.Std)
	}

	inputTensor, err := ort.NewTensor([]int64{1, 3, int64(imgH), int64(targetW)}, inp)
	if err != nil {
		return "", 0, err
	}
	defer inputTensor.Destroy()

	outputs := []ort.Value{nil}
	if err := e.rec.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return "", 0, err
	}
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return "", 0, fmt.Errorf("unexpected rec output type")
	}
	defer outTensor.Destroy()
	shape := outTensor.GetShape()
	// Expected [1, T, classes].
	T := 1
	classes := e.dict.Classes()
	if len(shape) >= 3 {
		T = int(shape[1])
		classes = int(shape[2])
	}
	text, conf := e.dict.DecodeCTC(outTensor.GetData(), T, classes)
	return text, conf, nil
}

// runCls runs the angle classifier on a crop.
func (e *Engine) runCls(crop *Image) (ClsResult, error) {
	iw, ih := e.clsParams.ImgW, e.clsParams.ImgH
	resized := crop.Resize(iw, ih)
	inp := resized.ToFloatCHW(1.0/255.0, e.clsParams.Mean, e.clsParams.Std)

	inputTensor, err := ort.NewTensor([]int64{1, 3, int64(ih), int64(iw)}, inp)
	if err != nil {
		return ClsResult{}, err
	}
	defer inputTensor.Destroy()
	outputs := []ort.Value{nil}
	if err := e.cls.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return ClsResult{}, err
	}
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return ClsResult{}, fmt.Errorf("unexpected cls output type")
	}
	defer outTensor.Destroy()
	return ClsPostprocess(outTensor.GetData()), nil
}

// runOri runs the document orientation classifier on the full image.
func (e *Engine) runOri(img *Image) (OriResult, error) {
	iw, ih := e.oriParams.ImgW, e.oriParams.ImgH
	resized := img.Resize(iw, ih)
	inp := resized.ToFloatCHW(1.0/255.0, e.oriParams.Mean, e.oriParams.Std)

	inputTensor, err := ort.NewTensor([]int64{1, 3, int64(ih), int64(iw)}, inp)
	if err != nil {
		return OriResult{}, err
	}
	defer inputTensor.Destroy()
	outputs := []ort.Value{nil}
	if err := e.ori.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return OriResult{}, err
	}
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return OriResult{}, fmt.Errorf("unexpected doc orientation output type")
	}
	defer outTensor.Destroy()
	return OriPostprocess(outTensor.GetData()), nil
}

func cropW(box DetectResult) int {
	w := math.Hypot(box.Box[1][0]-box.Box[0][0], box.Box[1][1]-box.Box[0][1])
	return max(int(w), 4)
}

func cropH(box DetectResult) int {
	h := math.Hypot(box.Box[3][0]-box.Box[0][0], box.Box[3][1]-box.Box[0][1])
	return max(int(h), 4)
}

func nowMs() float64 {
	return float64(time.Now().UnixNano()) / 1e6
}

// sortLines orders text boxes top-to-bottom then left-to-right.
func sortLines(lines []Line) {
	sort.SliceStable(lines, func(i, j int) bool {
		if math.Abs(lines[i].Box[0][1]-lines[j].Box[0][1]) > 8 {
			return lines[i].Box[0][1] < lines[j].Box[0][1]
		}
		return lines[i].Box[0][0] < lines[j].Box[0][0]
	})
}