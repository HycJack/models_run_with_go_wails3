package paddleocr

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// RecParams configures recognition model inference.
type RecParams struct {
	ImgH      int // fixed input height (32 for mobile models, 48 for server)
	MaxWidth  int // maximum input width before it is stretched
	Mean      [3]float32
	Std       [3]float32
	CharSpace float32
}

func DefaultRecParams() RecParams {
	return RecParams{
		ImgH:     32,
		MaxWidth: 320,
		Mean:     [3]float32{0.5, 0.5, 0.5},
		Std:      [3]float32{0.5, 0.5, 0.5},
	}
}

// Dict holds the character set used by the recognition model. Index 0 is
// reserved for the CTC blank token.
type Dict struct {
	chars []string
	byID  map[string]int
}

// LoadDict reads a PaddleOCR dictionary file (one character per line).
func LoadDict(path string) (*Dict, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	d := &Dict{byID: map[string]int{}}
	d.chars = append(d.chars, " ") // index 0 = blank
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSuffix(sc.Text(), "\n")
		line = strings.TrimSuffix(line, "\r")
		if line == "" {
			continue
		}
		d.chars = append(d.chars, line)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	for i, c := range d.chars {
		d.byID[c] = i
	}
	return d, nil
}

// Char returns the character for a class index (blank excluded).
func (d *Dict) Char(i int) string {
	if i < 0 || i >= len(d.chars) {
		return ""
	}
	return d.chars[i]
}

// Classes returns the number of classes including blank.
func (d *Dict) Classes() int { return len(d.chars) }

// DecodeCTC collapses repeated labels and removes blanks from the argmax
// sequence. Returns the recognized text and its confidence.
func (d *Dict) DecodeCTC(logits []float32, T, classes int) (string, float32) {
	var sb strings.Builder
	prev := -1
	confSum := float32(0)
	count := 0
	for t := 0; t < T; t++ {
		best := 0
		bestVal := logits[t*classes]
		for c := 1; c < classes; c++ {
			v := logits[t*classes+c]
			if v > bestVal {
				bestVal = v
				best = c
			}
		}
		if best == 0 {
			prev = 0
			continue
		}
		if best != prev {
			sb.WriteString(d.Char(best))
			prev = best
			count++
			confSum += bestVal
		}
	}
	conf := float32(0)
	if count > 0 {
		conf = confSum / float32(count)
	}
	return sb.String(), conf
}

// ClassCount reads the number of classes from the model config or the dict.
func (d *Dict) ClassCount() int { return d.Classes() }

// DecodeTextOnly is a helper used by the engine.
func (d *Dict) DecodeTextOnly(logits []float32, T, classes int) string {
	s, _ := d.DecodeCTC(logits, T, classes)
	return s
}

// String returns a diagnostic description.
func (d *Dict) String() string {
	return fmt.Sprintf("Dict(%d chars)", d.Classes())
}