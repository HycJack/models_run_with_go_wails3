package sensevoice

import (
	"fmt"
	"os"
	"strings"
)

// SentencePiece is a minimal decoder for SentencePiece BPE .model files:
// it reads the pieces table and maps token ids back to text.
type SentencePiece struct {
	pieces []string
	types  []int32
}

// LoadSentencePiece parses a SentencePiece .model protobuf.
func LoadSentencePiece(path string) (*SentencePiece, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	sp := &SentencePiece{}
	i := 0
	for i < len(data) {
		field, wt, next, err := readTag(data, i)
		if err != nil {
			return nil, err
		}
		i = next
		switch wt {
		case 0:
			_, i, err = readVarint(data, i)
		case 1:
			i += 8
		case 2:
			l, ni, err := readVarint(data, i)
			if err != nil {
				return nil, err
			}
			if field == 1 { // repeated SentencePiece
				if err := sp.parsePiece(data[ni : ni+int(l)]); err != nil {
					return nil, err
				}
			}
			i = ni + int(l)
		case 5:
			i += 4
		default:
			return nil, fmt.Errorf("unsupported wire type %d", wt)
		}
		if err != nil {
			return nil, err
		}
	}
	if len(sp.pieces) == 0 {
		return nil, fmt.Errorf("no pieces in sentencepiece model")
	}
	return sp, nil
}

func (sp *SentencePiece) parsePiece(data []byte) error {
	var piece string
	var typ int32
	i := 0
	for i < len(data) {
		field, wt, next, err := readTag(data, i)
		if err != nil {
			return err
		}
		i = next
		switch wt {
		case 0:
			v, ni, err := readVarint(data, i)
			if err != nil {
				return err
			}
			if field == 3 {
				typ = int32(v)
			}
			i = ni
		case 1:
			i += 8
		case 2:
			l, ni, err := readVarint(data, i)
			if err != nil {
				return err
			}
			if field == 1 {
				piece = string(data[ni : ni+int(l)])
			}
			i = ni + int(l)
		case 5:
			i += 4
		default:
			return fmt.Errorf("unsupported wire type %d", wt)
		}
	}
	sp.pieces = append(sp.pieces, piece)
	sp.types = append(sp.types, typ)
	return nil
}

// Decode converts a sequence of token ids into text, skipping control/unknown
// tokens and SenseVoice special tags (<|zh|>, <|NEUTRAL|>, ...) and turning the
// SentencePiece space marker (▁) into spaces.
func (sp *SentencePiece) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(sp.pieces) {
			continue
		}
		t := sp.types[id]
		if t == 2 || t == 3 || t == 6 { // UNKNOWN/CONTROL/UNUSED
			continue
		}
		p := sp.pieces[id]
		if p == "<unk>" || p == "<s>" || p == "</s>" || p == "<blank>" {
			continue
		}
		// Strip SenseVoice task/emotion/event tags.
		if strings.HasPrefix(p, "<|") && strings.HasSuffix(p, "|>") {
			continue
		}
		sb.WriteString(p)
	}
	return strings.ReplaceAll(sb.String(), "▁", " ")
}

func readTag(b []byte, i int) (field int, wt int, next int, err error) {
	if i >= len(b) {
		return 0, 0, i, fmt.Errorf("unexpected end")
	}
	v, ni, err := readVarint(b, i)
	if err != nil {
		return 0, 0, i, err
	}
	return int(v >> 3), int(v & 7), ni, nil
}

func readVarint(b []byte, i int) (uint64, int, error) {
	var result uint64
	var shift uint
	for {
		if i >= len(b) {
			return 0, i, fmt.Errorf("unexpected end")
		}
		c := b[i]
		i++
		result |= uint64(c&0x7f) << shift
		if c&0x80 == 0 {
			return result, i, nil
		}
		shift += 7
		if shift >= 64 {
			return 0, i, fmt.Errorf("varint overflow")
		}
	}
}