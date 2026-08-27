package llm

import (
	"fmt"
	"os"
	"path/filepath"
)

// Tokenizer wraps a HuggingFace tokenizer loaded from tokenizer.json.
type Tokenizer struct {
	tk *qwenTokenizer
}

// LoadTokenizer loads tokenizer.json from the model directory.
func LoadTokenizer(dir string) (*Tokenizer, error) {
	paths := []string{
		filepath.Join(dir, "tokenizer.json"),
		filepath.Join(dir, "tokenizer", "tokenizer.json"),
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			tk, err := loadQwenTokenizer(p)
			if err != nil {
				return nil, fmt.Errorf("load tokenizer: %w", err)
			}
			return &Tokenizer{tk: tk}, nil
		}
	}
	return nil, fmt.Errorf("tokenizer.json not found in %s", dir)
}

// Encode converts text into token ids without adding special tokens.
func (t *Tokenizer) Encode(text string) []int {
	return t.tk.Encode(text)
}

// Decode converts token ids back into text, skipping special tokens.
func (t *Tokenizer) Decode(ids []int) string {
	return t.tk.Decode(ids, true)
}

// TokenToID looks up a special token (e.g. "<|im_end|>").
func (t *Tokenizer) TokenToID(token string) (int, bool) {
	return t.tk.TokenToID(token)
}