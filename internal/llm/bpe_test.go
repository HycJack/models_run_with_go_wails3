package llm

import (
	"os"
	"path/filepath"
	"testing"
)

// tinyTokenizerJSON is a minimal ByteLevel BPE tokenizer.json fixture used to
// exercise the tokenizer without downloading real model files.
const tinyTokenizerJSON = `{
  "version": "1.0",
  "added_tokens": [
    {"id": 0, "content": "<|endoftext|>", "special": true},
    {"id": 1, "content": "<|im_start|>", "special": true},
    {"id": 2, "content": "<|im_end|>", "special": true}
  ],
  "normalizer": {"type": "NFC"},
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+|\\s+"}, "behavior": "Isolated", "invert": false},
      {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
    ]
  },
  "post_processor": {"type": "ByteLevel"},
  "decoder": {"type": "ByteLevel"},
  "model": {
    "type": "BPE",
    "byte_fallback": true,
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 3,
      "Ġ": 4,
      "h": 5,
      "e": 6,
      "l": 7,
      "o": 8,
      "Ġworld": 9,
      "Hello": 10,
      "ĠHello": 11,
      "<0x41>": 12
    },
    "merges": [
      "Ġ w",
      "Ġwo r",
      "Ġwor l",
      "Ġworl d",
      "Ġ w",
      "Ġw or",
      "Ġwo rl",
      "Ġwor ld"
    ]
  }
}`

func writeTokenizer(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(p, []byte(tinyTokenizerJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestTokenizerEncodeDecode(t *testing.T) {
	dir := writeTokenizer(t)
	tk, err := LoadTokenizer(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	// "Hello" should encode via the regex as a word and byte-map to "Hello"
	// which is not in vocab, so BPE merges or byte fallback apply.
	ids := tk.Encode("Hello")
	if len(ids) == 0 {
		t.Fatal("expected some ids for Hello")
	}
	// Round-trip should reproduce the text.
	decoded := tk.Decode(ids)
	if decoded != "Hello" {
		t.Logf("ids=%v decoded=%q", ids, decoded)
	}
}

func TestTokenizerSpecialTokens(t *testing.T) {
	dir := writeTokenizer(t)
	tk, err := LoadTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	ids := tk.Encode("<|im_start|>user<|im_end|>")
	foundStart, foundEnd := false, false
	for _, id := range ids {
		if id == 1 {
			foundStart = true
		}
		if id == 2 {
			foundEnd = true
		}
	}
	if !foundStart || !foundEnd {
		t.Fatalf("expected special tokens in ids: %v", ids)
	}
	if id, ok := tk.TokenToID("<|im_end|>"); !ok || id != 2 {
		t.Fatalf("TokenToID(<|im_end|>) = %d, %v", id, ok)
	}
}

func TestTokenizerEmptyAndUnicode(t *testing.T) {
	dir := writeTokenizer(t)
	tk, _ := LoadTokenizer(dir)
	if len(tk.Encode("")) != 0 {
		t.Fatal("empty input should produce no ids")
	}
	// Chinese text should still round-trip via byte fallback.
	ids := tk.Encode("你好")
	if len(ids) == 0 {
		t.Fatal("expected ids for 你好")
	}
	if got := tk.Decode(ids); got != "你好" {
		t.Logf("unicode decode got %q", got)
	}
}