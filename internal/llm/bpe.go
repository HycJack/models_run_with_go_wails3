package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/dlclark/regexp2"
	"golang.org/x/text/unicode/norm"
)

// byteLevelAlphabet maps each byte to a printable unicode character, matching
// the HuggingFace tokenizers ByteLevel alphabet.
func byteLevelAlphabet() ([256]string, map[rune]byte) {
	var alpha [256]string
	inverse := make(map[rune]byte, 256)
	// Characters from '!' to '~', '¡' to '¬', '®' to 'ÿ'.
	bs := []int{}
	for b := '!'; b <= '~'; b++ {
		bs = append(bs, int(b))
	}
	for b := '¡'; b <= '¬'; b++ {
		bs = append(bs, int(b))
	}
	for b := '®'; b <= 'ÿ'; b++ {
		bs = append(bs, int(b))
	}
	cs := append([]int{}, bs...)
	n := 0
	for b := 0; b < 256; b++ {
		if !containsInt(bs, b) {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	for i := range bs {
		alpha[bs[i]] = string(rune(cs[i]))
		inverse[rune(cs[i])] = byte(bs[i])
	}
	return alpha, inverse
}

func containsInt(s []int, v int) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

type addedToken struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type tokenizerModel struct {
	Vocab             map[string]int   `json:"vocab"`
	Merges            json.RawMessage  `json:"merges"`
	ByteFallback      bool             `json:"byte_fallback"`
	UnkToken          string           `json:"unk_token"`
}

// qwenTokenizer is a pure-Go HuggingFace BPE tokenizer sufficient for
// Qwen2/MiniCPM (ByteLevel + byte fallback) tokenizer.json files.
type qwenTokenizer struct {
	vocab        map[string]int
	idToToken    map[int]string
	addedIDs     map[string]int
	merges       map[string]int
	added        []addedToken
	alpha        [256]string
	inverse      map[rune]byte
	splitRegex   *regexp2.Regexp
	unkID        int
	specialSet   map[string]int
}

// loadQwenTokenizer parses a tokenizer.json file.
func loadQwenTokenizer(path string) (*qwenTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		AddedTokens []addedToken `json:"added_tokens"`
		PreTokenizer struct {
			Pretokenizers []struct {
				Type    string `json:"type"`
				Pattern *struct {
					Regex string `json:"Regex"`
				} `json:"pattern"`
			} `json:"pretokenizers"`
		} `json:"pre_tokenizer"`
		Model tokenizerModel `json:"model"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	if doc.Model.Vocab == nil {
		return nil, fmt.Errorf("tokenizer.json has no BPE vocab")
	}
	regexStr := "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
	for _, pt := range doc.PreTokenizer.Pretokenizers {
		if pt.Type == "Split" && pt.Pattern != nil && pt.Pattern.Regex != "" {
			regexStr = pt.Pattern.Regex
		}
	}
	re, err := regexp2.Compile(regexStr, regexp2.None)
	if err != nil {
		return nil, fmt.Errorf("compile pretokenizer regex: %w", err)
	}
	alpha, inverse := byteLevelAlphabet()

	// Merges may be encoded either as "a b" strings or ["a","b"] arrays.
	merges := make(map[string]int)
	rawMerges := doc.Model.Merges
	if len(rawMerges) > 0 {
		switch rawMerges[0] {
		case '[':
			var arr [][]string
			if err := json.Unmarshal(rawMerges, &arr); err == nil {
				for i, m := range arr {
					merges[strings.Join(m, " ")] = i
				}
			}
		default:
			var strs []string
			if err := json.Unmarshal(rawMerges, &strs); err == nil {
				for i, m := range strs {
					merges[m] = i
				}
			}
		}
	}

	idToToken := make(map[int]string, len(doc.Model.Vocab))
	for tok, id := range doc.Model.Vocab {
		idToToken[id] = tok
	}

	specialSet := make(map[string]int)
	unkID := -1
	if u := doc.Model.UnkToken; u != "" {
		if id, ok := doc.Model.Vocab[u]; ok {
			unkID = id
		}
	}
	// Sort added tokens by length descending for longest-match.
	added := append([]addedToken{}, doc.AddedTokens...)
	sort.Slice(added, func(i, j int) bool { return len(added[i].Content) > len(added[j].Content) })
	// Added tokens are authoritative for their id (some tokenizers have
	// duplicate ids in the vocab map) and are always lookable.
	addedIDs := make(map[string]int, len(added))
	for _, at := range added {
		idToToken[at.ID] = at.Content
		addedIDs[at.Content] = at.ID
		if at.Special {
			specialSet[at.Content] = at.ID
		}
	}

	return &qwenTokenizer{
		vocab:      doc.Model.Vocab,
		idToToken:  idToToken,
		merges:     merges,
		added:      added,
		addedIDs:   addedIDs,
		alpha:      alpha,
		inverse:    inverse,
		splitRegex: re,
		unkID:      unkID,
		specialSet: specialSet,
	}, nil
}

// Encode converts text into token ids.
func (t *qwenTokenizer) Encode(text string) []int {
	text = norm.NFC.String(text)
	var ids []int
	// Split on added tokens while preserving the plain-text runs.
	for len(text) > 0 {
		matched := false
		for _, at := range t.added {
			if strings.HasPrefix(text, at.Content) {
				ids = append(ids, at.ID)
				text = text[len(at.Content):]
				matched = true
				break
			}
		}
		if matched {
			continue
		}
		// Take a chunk up to the next added token occurrence.
		next := len(text)
		for _, at := range t.added {
			if j := strings.Index(text, at.Content); j >= 0 && j < next {
				next = j
			}
		}
		chunk := text[:next]
		text = text[next:]
		ids = append(ids, t.encodeChunk(chunk)...)
	}
	return ids
}

// encodeChunk runs the ByteLevel BPE on a single text run.
func (t *qwenTokenizer) encodeChunk(text string) []int {
	if text == "" {
		return nil
	}
	var ids []int
	// Pre-tokenize with the Split regex (matches are the tokens).
	m, _ := t.splitRegex.FindStringMatch(text)
	for m != nil {
		group := m.GroupByNumber(0)
		tok := group.String()
		ids = append(ids, t.encodeWord(tok)...)
		m, _ = t.splitRegex.FindNextMatch(m)
	}
	return ids
}

// encodeWord byte-maps a regex token and applies BPE merges.
func (t *qwenTokenizer) encodeWord(word string) []int {
	// Byte-map the utf-8 bytes.
	var mapped strings.Builder
	for _, b := range []byte(word) {
		mapped.WriteString(t.alpha[b])
	}
	ms := mapped.String()
	if id, ok := t.vocab[ms]; ok {
		return []int{id}
	}
	merged := t.bpe(ms)
	out := make([]int, 0, len(merged))
	for _, tk := range merged {
		if id, ok := t.vocab[tk]; ok {
			out = append(out, id)
			continue
		}
		// Byte fallback: encode each byte as its <0xXX> token.
		for _, r := range tk {
			b, ok := t.inverse[r]
			if !ok {
				continue
			}
			hexTok := fmt.Sprintf("<0x%02X>", b)
			if id, ok := t.vocab[hexTok]; ok {
				out = append(out, id)
			} else if t.unkID >= 0 {
				out = append(out, t.unkID)
			}
		}
	}
	if len(out) == 0 && t.unkID >= 0 {
		return []int{t.unkID}
	}
	return out
}

// bpe applies the merge algorithm to a byte-mapped string and returns the
// resulting merged token strings.
func (t *qwenTokenizer) bpe(token string) []string {
	if token == "" {
		return nil
	}
	runes := []rune(token)
	word := make([]string, len(runes))
	for i, r := range runes {
		word[i] = string(r)
	}
	mergeOne := func(w []string) ([]string, bool) {
		bestRank := 1 << 30
		bestI := -1
		for i := 0; i < len(w)-1; i++ {
			key := w[i] + " " + w[i+1]
			if r, ok := t.merges[key]; ok && r < bestRank {
				bestRank = r
				bestI = i
			}
		}
		if bestI < 0 {
			return w, false
		}
		merged := w[bestI] + w[bestI+1]
		next := make([]string, 0, len(w)-1)
		next = append(next, w[:bestI]...)
		next = append(next, merged)
		next = append(next, w[bestI+2:]...)
		return next, true
	}
	for {
		var changed bool
		word, changed = mergeOne(word)
		if !changed {
			break
		}
	}
	return word
}

// Decode converts token ids back to text, skipping special tokens.
func (t *qwenTokenizer) Decode(ids []int, skipSpecial bool) string {
	var sb strings.Builder
	for _, id := range ids {
		tok := t.idToToken[id]
		if tok == "" {
			continue
		}
		if skipSpecial {
			if _, isSpecial := t.specialSet[tok]; isSpecial {
				continue
			}
		}
		for _, r := range tok {
			if b, ok := t.inverse[r]; ok {
				sb.WriteByte(b)
			}
		}
	}
	// The byte stream may split multi-byte runes across ids; decode in one go.
	return strings.ToValidUTF8(sb.String(), "\uFFFD")
}

// TokenToID looks up a special token id.
func (t *qwenTokenizer) TokenToID(token string) (int, bool) {
	if id, ok := t.specialSet[token]; ok {
		return id, true
	}
	if id, ok := t.addedIDs[token]; ok {
		return id, true
	}
	id, ok := t.vocab[token]
	return id, ok
}