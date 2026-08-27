package sensevoice

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"cpm_orc/internal/onnxmeta"
	"cpm_orc/internal/ort"
)

// Engine runs FunASR SenseVoiceSmall (ONNX) on-device.
type Engine struct {
	mu       sync.Mutex
	dir      string
	session  *ort.Session
	sp       *SentencePiece
	means    []float64
	vars     []float64
	opts     FbankOptions
	loaded   bool
	threads  int
}

// NewEngine creates an empty SenseVoice engine.
func NewEngine(threads int) *Engine {
	return &Engine{
		opts:    DefaultFbankOptions(),
		threads: threads,
	}
}

// Loaded reports whether the engine is ready.
func (e *Engine) Loaded() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.loaded
}

// Load initializes the engine from a model directory containing model.onnx,
// am.mvn and the SentencePiece .bpe.model.
func (e *Engine) Load(dir string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	onnxPath := filepath.Join(dir, "model.onnx")
	cmvnPath := filepath.Join(dir, "am.mvn")
	spPath := ""
	for _, name := range []string{
		"chn_jpn_yue_eng_ko_spectok.bpe.model",
		"tokenizer.model",
	} {
		if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
			spPath = filepath.Join(dir, name)
			break
		}
	}
	if _, err := os.Stat(onnxPath); err != nil {
		return fmt.Errorf("model.onnx not found in %s", dir)
	}
	if _, err := os.Stat(cmvnPath); err != nil {
		return fmt.Errorf("am.mvn not found in %s", dir)
	}
	if spPath == "" {
		return fmt.Errorf("sentencepiece .model not found in %s", dir)
	}

	io, err := onnxmeta.Parse(onnxPath)
	if err != nil {
		return err
	}
	sess, err := ort.NewSession(onnxPath, io.Inputs, io.Outputs, e.threads)
	if err != nil {
		return fmt.Errorf("create session: %w", err)
	}
	sp, err := LoadSentencePiece(spPath)
	if err != nil {
		sess.Destroy()
		return err
	}
	cmvn, err := os.ReadFile(cmvnPath)
	if err != nil {
		sess.Destroy()
		return err
	}
	means, vars, err := ParseCMVN(cmvn)
	if err != nil {
		sess.Destroy()
		return err
	}

	if e.session != nil {
		e.session.Destroy()
	}
	e.session = sess
	e.sp = sp
	e.means = means
	e.vars = vars
	e.dir = dir
	e.loaded = true
	return nil
}

// Close releases the session.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}
	e.loaded = false
	return nil
}

// Transcribe runs SenseVoice on WAV audio bytes. language uses the FunASR id
// ("auto"=0, "zh"=3, "en"=4, ...); textnorm "woitn"=15 (default).
func (e *Engine) Transcribe(audio []byte, language, textnorm string) (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.loaded || e.session == nil {
		return "", fmt.Errorf("SenseVoice engine is not loaded")
	}

	samples, err := ParseWav(audio)
	if err != nil {
		return "", fmt.Errorf("decode wav: %w", err)
	}
	if len(samples) < 400 {
		return "", fmt.Errorf("audio too short")
	}
	// Frontend: fbank -> LFR(7,6) -> CMVN.
	wave := make([]float64, len(samples))
	for i, s := range samples {
		wave[i] = float64(s) * 32768.0
	}
	feats := Fbank(wave, e.opts)
	if len(feats) == 0 {
		return "", fmt.Errorf("empty fbank")
	}
	feats = LFR(feats, e.opts.LFRM, e.opts.LFRN)
	feats = CMVN(feats, e.means, e.vars)

	T := len(feats)
	dim := 560
	flat := make([]float32, 0, T*dim)
	for _, row := range feats {
		flat = append(flat, row...)
	}

	langID := map[string]int32{"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12}[language]
	tnID := map[string]int32{"woitn": 15, "withitn": 14}[textnorm]

	speech, err := ort.NewTensor([]int64{1, int64(T), int64(dim)}, flat)
	if err != nil {
		return "", err
	}
	defer speech.Destroy()
	lengths, err := ort.NewTensor([]int64{1}, []int32{int32(T)})
	if err != nil {
		return "", err
	}
	defer lengths.Destroy()
	langT, err := ort.NewTensor([]int64{1}, []int32{langID})
	if err != nil {
		return "", err
	}
	defer langT.Destroy()
	tnT, err := ort.NewTensor([]int64{1}, []int32{tnID})
	if err != nil {
		return "", err
	}
	defer tnT.Destroy()

	// Auto-allocate the two outputs.
	outputs := []ort.Value{nil, nil}
	if err := e.session.Run([]ort.Value{speech, lengths, langT, tnT}, outputs); err != nil {
		return "", fmt.Errorf("inference: %w", err)
	}
	logitsT, ok1 := outputs[0].(*ort.Tensor[float32])
	lensT, ok2 := outputs[1].(*ort.Tensor[int32])
	if !ok1 || !ok2 {
		return "", fmt.Errorf("unexpected output types")
	}
	defer logitsT.Destroy()
	defer lensT.Destroy()

	shape := logitsT.GetShape()
	L := 1
	vocab := 25055
	if len(shape) >= 3 {
		L = int(shape[1])
		vocab = int(shape[2])
	}
	encLen := int(lensT.GetData()[0])
	if encLen > L {
		encLen = L
	}
	logits := logitsT.GetData()

	// Greedy CTC decode: argmax -> collapse -> remove blank(0).
	var ids []int
	prev := -1
	for t := 0; t < encLen; t++ {
		base := t * vocab
		best := 0
		bestV := logits[base]
		for c := 1; c < vocab; c++ {
			if logits[base+c] > bestV {
				bestV = logits[base+c]
				best = c
			}
		}
		if best == 0 {
			prev = 0
			continue
		}
		if best != prev {
			ids = append(ids, best)
			prev = best
		}
	}
	return e.sp.Decode(ids), nil
}