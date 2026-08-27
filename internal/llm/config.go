package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// IntOrSlice unmarshals a JSON field that may be either a single int or an
// array of ints (e.g. eos_token_id in some configs).
type IntOrSlice []int

func (i *IntOrSlice) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] == '[' {
		return json.Unmarshal(b, (*[]int)(i))
	}
	var v int
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	*i = []int{v}
	return nil
}

// First returns the first id or -1 when empty.
func (i IntOrSlice) First() int {
	if len(i) == 0 {
		return -1
	}
	return i[0]
}

// ModelConfig is the subset of config.json the engine needs.
type ModelConfig struct {
	ModelType             string    `json:"model_type"`
	HiddenSize            int       `json:"hidden_size"`
	NumHiddenLayers       int       `json:"num_hidden_layers"`
	NumAttentionHeads     int       `json:"num_attention_heads"`
	NumKeyValueHeads      int       `json:"num_key_value_heads"`
	HeadDim               int       `json:"head_dim"`
	VocabSize             int       `json:"vocab_size"`
	MaxPositionEmbeddings int       `json:"max_position_embeddings"`
	RopeTheta             float64   `json:"rope_theta"`
	EOSTokenID            IntOrSlice `json:"eos_token_id"`
	BOSTokenID            IntOrSlice `json:"bos_token_id"`
	PADTokenID            IntOrSlice `json:"pad_token_id"`
}

// LoadConfig reads and validates config.json from a model directory.
func LoadConfig(dir string) (*ModelConfig, error) {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c ModelConfig
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, err
	}
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 {
		return nil, fmt.Errorf("config.json missing required dimensions (hidden_size, num_hidden_layers, num_attention_heads)")
	}
	if c.NumKeyValueHeads <= 0 {
		c.NumKeyValueHeads = c.NumAttentionHeads
	}
	if c.HeadDim <= 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 10000
	}
	if c.MaxPositionEmbeddings <= 0 {
		c.MaxPositionEmbeddings = 32768
	}
	switch c.ModelType {
	case "qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "gpt2", "qwen", "minicpm", "baichuan", "stablelm":
	default:
		return nil, fmt.Errorf("model_type %q is not supported by the ONNX engine (expected qwen2/llama/mistral family)", c.ModelType)
	}
	return &c, nil
}

// KVShape returns the shape of the merged past_key_values tensor.
// The cache is laid out as [2, layers, kv_heads, seq, head_dim] where the
// first axis indexes [key, value].
func (c *ModelConfig) KVShape(seq int) []int64 {
	return []int64{2, int64(c.NumHiddenLayers), int64(c.NumKeyValueHeads), int64(seq), int64(c.HeadDim)}
}

// EOSTokens returns a set of end-of-sequence token ids.
func (c *ModelConfig) EOSTokens() map[int]bool {
	m := map[int]bool{}
	for _, id := range c.EOSTokenID {
		if id >= 0 {
			m[id] = true
		}
	}
	// Qwen-style chat models stop on the special sequence terminator.
	m[151645] = true // <|im_end|> for Qwen2 151K vocab
	return m
}