package llm

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"cpm_orc/internal/onnxmeta"
	"cpm_orc/internal/ort"
)

// GenOptions controls generation.
type GenOptions struct {
	MaxNewTokens      int     `json:"maxNewTokens"`
	Temperature       float64 `json:"temperature"`
	TopK              int     `json:"topK"`
	TopP              float64 `json:"topP"`
	RepetitionPenalty float64 `json:"repetitionPenalty"`
	SystemPrompt      string  `json:"systemPrompt"`
	UseChatTemplate   bool    `json:"useChatTemplate"`
	Seed              int64   `json:"seed"`
	Stream            bool    `json:"stream"`
}

func DefaultGenOptions() GenOptions {
	return GenOptions{
		MaxNewTokens:      512,
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		RepetitionPenalty: 1.0,
		Seed:              42,
		Stream:            true,
	}
}

// Engine runs decoder-only causal LMs (Qwen2/MiniCPM family) exported to ONNX
// with a per-layer KV cache (past_key_values.N.key/value), using ONNX
// Runtime. This is the format produced by HuggingFace Optimum and
// transformers.js exports.
type Engine struct {
	mu        sync.Mutex
	dir       string
	session   *ort.Session
	io        *onnxmeta.ModelIO
	config    *ModelConfig
	tokenizer *Tokenizer
	layers    int
	threads   int

	stopMu     sync.Mutex
	stopFlag   bool
	generating bool
}

// NewEngine creates an empty engine.
func NewEngine(threads int) *Engine {
	return &Engine{threads: threads}
}

// Loaded reports whether a model is loaded.
func (e *Engine) Loaded() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.session != nil
}

// Status returns basic info about the loaded model.
func (e *Engine) Status() (dir, modelType string, err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.session == nil {
		return "", "", fmt.Errorf("no model loaded")
	}
	return e.dir, e.config.ModelType, nil
}

// Load loads an ONNX export (model.onnx + config.json + tokenizer.json).
// The model must expose a per-layer KV cache.
func (e *Engine) Load(dir string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	onnxPath, err := findOnnxModel(dir)
	if err != nil {
		return err
	}
	cfg, err := LoadConfig(dir)
	if err != nil {
		return err
	}
	tk, err := LoadTokenizer(dir)
	if err != nil {
		return err
	}
	io, err := onnxmeta.Parse(onnxPath)
	if err != nil {
		return err
	}
	layers, err := countKVLayers(io)
	if err != nil {
		return err
	}
	sess, err := ort.NewSession(onnxPath, io.Inputs, io.Outputs, e.threads)
	if err != nil {
		return fmt.Errorf("create ONNX session: %w", err)
	}

	if e.session != nil {
		e.session.Destroy()
	}
	e.session = sess
	e.io = io
	e.config = cfg
	e.tokenizer = tk
	e.dir = dir
	e.layers = layers
	return nil
}

// Unload releases the loaded model.
func (e *Engine) Unload() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}
	return nil
}

// findOnnxModel locates a single .onnx model in a directory.
func findOnnxModel(dir string) (string, error) {
	preferred := []string{"model.onnx", "decoder_model_merged.onnx"}
	for _, p := range preferred {
		cand := filepath.Join(dir, p)
		if _, err := os.Stat(cand); err == nil {
			return cand, nil
		}
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "*.onnx"))
	if len(matches) == 0 {
		matches, _ = filepath.Glob(filepath.Join(dir, "*", "*.onnx"))
	}
	if len(matches) == 0 {
		return "", fmt.Errorf("no .onnx model found in %s", dir)
	}
	sort.Strings(matches)
	if len(matches) > 1 {
		return "", fmt.Errorf("multiple .onnx files found in %s; expected a single model.onnx", dir)
	}
	return matches[0], nil
}

// countKVLayers determines the number of decoder layers from the KV cache
// input names and validates the cache format.
func countKVLayers(io *onnxmeta.ModelIO) (int, error) {
	maxLayer := -1
	for _, in := range io.Inputs {
		if in == "past_key_values" {
			return 0, fmt.Errorf("model uses a merged past_key_values cache; this engine requires the per-layer cache format (export with Optimum using transformers)")
		}
		if strings.HasPrefix(in, "past_key_values.") {
			idx, err := parseKVPrefix(in)
			if err != nil {
				return 0, err
			}
			if idx > maxLayer {
				maxLayer = idx
			}
		}
	}
	if maxLayer < 0 {
		return 0, fmt.Errorf("model does not expose a past_key_values.N.key/value KV cache")
	}
	return maxLayer + 1, nil
}

// parseKVPrefix extracts the layer index from a name like
// "past_key_values.3.key" or "past_key_values.3.value".
func parseKVPrefix(name string) (int, error) {
	rest := strings.TrimPrefix(name, "past_key_values.")
	rest = strings.TrimSuffix(rest, ".key")
	rest = strings.TrimSuffix(rest, ".value")
	rest = strings.TrimSuffix(rest, ".K")
	rest = strings.TrimSuffix(rest, ".V")
	idx, err := strconv.Atoi(rest)
	if err != nil {
		return 0, fmt.Errorf("cannot parse KV cache input name %q", name)
	}
	return idx, nil
}

// Generate produces a completion for the prompt. When onToken is non-nil it is
// called with each incremental decoded chunk for streaming. For reasoning
// models (MiniCPM thinking/response style), the reasoning section is reported
// through onReasoning and only the final response is returned.
func (e *Engine) Generate(prompt string, opts GenOptions, onToken func(string), onReasoning ...func(string)) (string, error) {
	e.mu.Lock()
	if e.session == nil {
		e.mu.Unlock()
		return "", fmt.Errorf("no model loaded")
	}
	sess := e.session
	cfg := e.config
	tk := e.tokenizer
	io := e.io
	layers := e.layers
	e.mu.Unlock()

	e.stopMu.Lock()
	e.stopFlag = false
	e.generating = true
	e.stopMu.Unlock()
	defer func() {
		e.stopMu.Lock()
		e.generating = false
		e.stopFlag = false
		e.stopMu.Unlock()
	}()

	text := prompt
	if opts.UseChatTemplate {
		text = BuildChatTemplate(opts.SystemPrompt, prompt, cfg.ModelType)
	} else if needsBOS(cfg.ModelType) {
		// Llama-family decoders expect the sequence to start with <s>.
		text = "<s>" + text
	}
	inputIDs := tk.Encode(text)
	if len(inputIDs) == 0 {
		return "", fmt.Errorf("prompt tokenized to empty input")
	}

	vocab := cfg.VocabSize
	if vocab <= 0 {
		vocab = cfg.HiddenSize * 16
	}
	sampler := Sampler{
		Temperature:       opts.Temperature,
		TopK:              opts.TopK,
		TopP:              opts.TopP,
		RepetitionPenalty: opts.RepetitionPenalty,
	}
	if opts.Seed == 0 {
		opts.Seed = 42
	}
	rng := rand.New(rand.NewSource(opts.Seed))
	eos := cfg.EOSTokens()

	maxNew := opts.MaxNewTokens
	if maxNew <= 0 {
		maxNew = 512
	}
	generated := make([]int, 0, maxNew)
	var displayed string
	reasoningCb := func(string) {}
	if len(onReasoning) > 0 && onReasoning[0] != nil {
		reasoningCb = onReasoning[0]
	}

	// Reasoning models emit a marker token (MiniCPM: " response"; Qwen3:
	// "<|startofresponse|>" / "<|endofthinking|>") marking the start of the
	// answer. Detect it from the tokenizer so reasoning is streamed separately.
	reasoningModel := false
	responseMarker := "</think>"
	for _, m := range []string{"</think>", "<|endofthinking|>", "<|startofresponse|>"} {
		if _, ok := tk.TokenToID(m); ok {
			reasoningModel = true
			responseMarker = m
			break
		}
	}
	inResponse := !reasoningModel

	emit := func(ids []int) string {
		decoded := tk.Decode(ids)
		if inResponse {
			delta := strings.TrimPrefix(decoded, displayed)
			if len(delta) > 0 && onToken != nil {
				onToken(delta)
			}
			displayed = decoded
			return decoded
		}
		// Reasoning model, still before the response marker.
		if idx := strings.Index(decoded, responseMarker); idx >= 0 {
			inResponse = true
			reasoning := strings.TrimSpace(decoded[:idx])
			if reasoning != "" {
				reasoningCb(reasoning)
			}
			response := strings.TrimLeft(decoded[idx+len(responseMarker):], " \n")
			if response != "" && onToken != nil {
				onToken(response)
			}
			displayed = decoded
			return decoded
		}
		delta := strings.TrimPrefix(decoded, displayed)
		if len(delta) > 0 {
			reasoningCb(delta)
		}
		displayed = decoded
		return decoded
	}

	kvShape := func(seq int) []int64 {
		return []int64{1, int64(cfg.NumKeyValueHeads), int64(seq), int64(cfg.HeadDim)}
	}

	// Prefill: empty KV caches for every layer.
	pastKeys := make([]*ort.Tensor[float32], layers)
	pastVals := make([]*ort.Tensor[float32], layers)
	for i := 0; i < layers; i++ {
		k, err := ort.NewEmptyTensor[float32](kvShape(0))
		if err != nil {
			return "", err
		}
		v, err := ort.NewEmptyTensor[float32](kvShape(0))
		if err != nil {
			k.Destroy()
			return "", err
		}
		pastKeys[i] = k
		pastVals[i] = v
	}
	cleanupPast := func() {
		for i := 0; i < layers; i++ {
			if pastKeys[i] != nil {
				pastKeys[i].Destroy()
			}
			if pastVals[i] != nil {
				pastVals[i].Destroy()
			}
		}
	}
	defer cleanupPast()

	// Build the ordered input/output name lists (excluding KV handled below).
	var inputNames []string
	var kvInKeyIdx, kvInValIdx []int
	for _, n := range io.Inputs {
		switch {
		case n == "input_ids", n == "attention_mask", n == "position_ids", n == "cache_position":
			inputNames = append(inputNames, n)
		case strings.HasPrefix(n, "past_key_values."):
			idx, _ := parseKVPrefix(n)
			if strings.HasSuffix(n, ".key") || strings.HasSuffix(n, ".K") {
				for len(kvInKeyIdx) <= idx {
					kvInKeyIdx = append(kvInKeyIdx, -1)
				}
				kvInKeyIdx[idx] = len(inputNames)
				inputNames = append(inputNames, n)
			} else {
				for len(kvInValIdx) <= idx {
					kvInValIdx = append(kvInValIdx, -1)
				}
				kvInValIdx[idx] = len(inputNames)
				inputNames = append(inputNames, n)
			}
		}
	}

	// Position of each named input in the session's expected order.
	posOf := func(name string) int {
		for i, n := range io.Inputs {
			if n == name {
				return i
			}
		}
		return -1
	}

	// We build inputs by iterating the session's input order, mapping to our
	// tensor list.
	idsPos := posOf("input_ids")
	maskPos := posOf("attention_mask")
	if idsPos < 0 {
		return "", fmt.Errorf("model has no input_ids input")
	}

	lastID := 0
	for step := 0; step <= maxNew; step++ {
		if e.shouldStop() {
			break
		}
		curLen := len(inputIDs) + len(generated)
		var idsTensor *ort.Tensor[int64]
		var err error
		if step == 0 {
			ids := make([]int64, len(inputIDs))
			for i, v := range inputIDs {
				ids[i] = int64(v)
			}
			idsTensor, err = ort.NewTensor([]int64{1, int64(len(inputIDs))}, ids)
		} else {
			idsTensor, err = ort.NewTensor([]int64{1, 1}, []int64{int64(lastID)})
		}
		if err != nil {
			return "", err
		}
		attnTensor, err := makeOnesTensor(curLen)
		if err != nil {
			idsTensor.Destroy()
			return "", err
		}
		posTensor := (*ort.Tensor[int64])(nil)
		if posOf("position_ids") >= 0 || posOf("cache_position") >= 0 {
			if step == 0 {
				posTensor, err = makePositionTensor(curLen)
			} else {
				posTensor, err = ort.NewTensor([]int64{1, 1}, []int64{int64(curLen - 1)})
			}
			if err != nil {
				idsTensor.Destroy()
				attnTensor.Destroy()
				return "", err
			}
		}

		// Assemble inputs in session order.
		inputs := make([]ort.Value, len(io.Inputs))
		inputs[idsPos] = idsTensor
		if maskPos >= 0 {
			inputs[maskPos] = attnTensor
		}
		if posTensor != nil {
			if p := posOf("position_ids"); p >= 0 {
				inputs[p] = posTensor
			}
			if c := posOf("cache_position"); c >= 0 {
				inputs[c] = posTensor
			}
		}
		for l := 0; l < layers; l++ {
			if l < len(kvInKeyIdx) && kvInKeyIdx[l] >= 0 {
				inputs[kvInKeyIdx[l]] = pastKeys[l]
			}
			if l < len(kvInValIdx) && kvInValIdx[l] >= 0 {
				inputs[kvInValIdx[l]] = pastVals[l]
			}
		}

		outputs := make([]ort.Value, len(io.Outputs))
		err = sess.Run(inputs, outputs)
		if err != nil {
			idsTensor.Destroy()
			attnTensor.Destroy()
			if posTensor != nil {
				posTensor.Destroy()
			}
			return "", fmt.Errorf("inference step %d: %w", step, err)
		}

		// logits is the output whose name is "logits".
		logitsT := outputs[logitsIdx(io)]
		lT, ok := logitsT.(*ort.Tensor[float32])
		if !ok {
			return "", fmt.Errorf("unexpected logits output type")
		}
		logits := lT.GetData()
		shape := lT.GetShape()
		logitsSeq := 1
		if len(shape) >= 2 {
			logitsSeq = int(shape[1])
		}
		lastRow := make([]float32, vocab)
		copy(lastRow, logits[(logitsSeq-1)*vocab:logitsSeq*vocab])

		nextID := sampler.Sample(lastRow, generated, rng)
		lastID = nextID
		generated = append(generated, nextID)

		// Grab new present tensors and advance the cache.
		for l := 0; l < layers; l++ {
			ki, vi := presentIdx(io, l)
			newK, okK := outputs[ki].(*ort.Tensor[float32])
			newV, okV := outputs[vi].(*ort.Tensor[float32])
			if !okK || !okV {
				return "", fmt.Errorf("unexpected present tensor type")
			}
			if old := pastKeys[l]; old != nil {
				old.Destroy()
			}
			if old := pastVals[l]; old != nil {
				old.Destroy()
			}
			pastKeys[l] = newK
			pastVals[l] = newV
		}

		idsTensor.Destroy()
		attnTensor.Destroy()
		if posTensor != nil {
			posTensor.Destroy()
		}
		lT.Destroy()

		if e.shouldStop() {
			break
		}
		if len(generated) >= maxNew {
			break
		}
		if eos[nextID] {
			break
		}
		chunk := tk.Decode([]int{nextID})
		if strings.Contains(chunk, "<|im_end|>") || strings.Contains(chunk, "<|endoftext|>") {
			break
		}
		emit(generated)
	}
	emit(generated)
	// For reasoning models return only the final answer section.
	if reasoningModel {
		if idx := strings.Index(displayed, responseMarker); idx >= 0 {
			return strings.TrimSpace(displayed[idx+len(responseMarker):]), nil
		}
		return strings.TrimSpace(displayed), nil
	}
	return displayed, nil
}

// logitsIdx returns the index of the logits output.
func logitsIdx(io *onnxmeta.ModelIO) int {
	for i, n := range io.Outputs {
		if n == "logits" {
			return i
		}
	}
	return 0
}

// presentIdx returns the output indices for a layer's key and value.
func presentIdx(io *onnxmeta.ModelIO, layer int) (int, int) {
	ki, vi := -1, -1
	for i, n := range io.Outputs {
		if strings.HasPrefix(n, "present.") {
			rest := strings.TrimPrefix(n, "present.")
			rest = strings.TrimSuffix(rest, ".key")
			rest = strings.TrimSuffix(rest, ".value")
			rest = strings.TrimSuffix(rest, ".K")
			rest = strings.TrimSuffix(rest, ".V")
			if idx, err := strconv.Atoi(rest); err == nil && idx == layer {
				if strings.Contains(n, "key") || strings.HasSuffix(n, ".K") {
					ki = i
				} else {
					vi = i
				}
			}
		}
	}
	return ki, vi
}

// Stop signals an ongoing generation to stop at the next step boundary.
func (e *Engine) Stop() {
	e.stopMu.Lock()
	e.stopFlag = true
	e.stopMu.Unlock()
}

// IsGenerating reports whether a generation is running.
func (e *Engine) IsGenerating() bool {
	e.stopMu.Lock()
	defer e.stopMu.Unlock()
	return e.generating
}

func (e *Engine) shouldStop() bool {
	e.stopMu.Lock()
	defer e.stopMu.Unlock()
	return e.stopFlag
}

func makeOnesTensor(n int) (*ort.Tensor[int64], error) {
	ones := make([]int64, n)
	for i := range ones {
		ones[i] = 1
	}
	return ort.NewTensor([]int64{1, int64(n)}, ones)
}

func makePositionTensor(n int) (*ort.Tensor[int64], error) {
	pos := make([]int64, n)
	for i := range pos {
		pos[i] = int64(i)
	}
	return ort.NewTensor([]int64{1, int64(n)}, pos)
}

// needsBOS reports whether the model family expects an explicit <s> sequence
// start token.
func needsBOS(modelType string) bool {
	switch modelType {
	case "llama", "mistral":
		return true
	}
	return false
}

// BuildChatTemplate wraps a user prompt in the ChatML template used by
// MiniCPM and Qwen chat models.
func BuildChatTemplate(system, user, modelType string) string {
	if system == "" {
		system = "You are a helpful assistant."
	}
	var sb strings.Builder
	if needsBOS(modelType) {
		sb.WriteString("<s>")
	}
	sb.WriteString("<|im_start|>system\n")
	sb.WriteString(system)
	sb.WriteString("<|im_end|>\n<|im_start|>user\n")
	sb.WriteString(user)
	sb.WriteString("<|im_end|>\n<|im_start|>assistant\n")
	return sb.String()
}