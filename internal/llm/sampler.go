package llm

import (
	"math"
	"math/rand"
	"sort"
)

// Sampler implements greedy, temperature, top-k and top-p sampling.
type Sampler struct {
	Temperature       float64
	TopK              int
	TopP              float64
	RepetitionPenalty float64
}

func DefaultSampler() Sampler {
	return Sampler{Temperature: 0.7, TopK: 40, TopP: 0.9, RepetitionPenalty: 1.0}
}

// Sample chooses the next token index from the raw logits (last step).
// seen is the list of token ids already generated (for repetition penalty).
func (s *Sampler) Sample(logits []float32, seen []int, rng *rand.Rand) int {
	n := len(logits)
	if n == 0 {
		return 0
	}
	negInf := float32(math.Inf(-1))
	// Repetition penalty.
	if s.RepetitionPenalty > 0 && s.RepetitionPenalty != 1.0 {
		seenSet := make(map[int]bool, len(seen))
		for _, id := range seen {
			seenSet[id] = true
		}
		for i := 0; i < n; i++ {
			if seenSet[i] {
				if logits[i] > 0 {
					logits[i] /= float32(s.RepetitionPenalty)
				} else {
					logits[i] *= float32(s.RepetitionPenalty)
				}
			}
		}
	}
	// Greedy.
	if s.Temperature <= 0 {
		best := 0
		for i := 1; i < n; i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}
	// Apply temperature.
	for i := range logits {
		logits[i] /= float32(s.Temperature)
	}

	// Top-k filtering.
	if s.TopK > 0 && s.TopK < n {
		idxs := make([]int, n)
		for i := range idxs {
			idxs[i] = i
		}
		sort.Slice(idxs, func(a, b int) bool { return logits[idxs[a]] > logits[idxs[b]] })
		cutoff := logits[idxs[s.TopK-1]]
		for i := 0; i < n; i++ {
			if logits[i] < cutoff {
				logits[i] = negInf
			}
		}
	}

	// Softmax.
	maxL := float64(math.Inf(-1))
	for _, v := range logits {
		if float64(v) > maxL {
			maxL = float64(v)
		}
	}
	sum := float64(0)
	probs := make([]float64, n)
	for i, v := range logits {
		if v == negInf {
			probs[i] = 0
			continue
		}
		probs[i] = math.Exp(float64(v) - maxL)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	// Top-p (nucleus) filtering.
	if s.TopP > 0 && s.TopP < 1.0 {
		idxSort := make([]int, n)
		for i := range idxSort {
			idxSort[i] = i
		}
		sort.Slice(idxSort, func(a, b int) bool { return probs[idxSort[a]] > probs[idxSort[b]] })
		acc := 0.0
		cutoff := 1.0
		for _, i := range idxSort {
			acc += probs[i]
			if acc > s.TopP {
				cutoff = probs[i]
				break
			}
		}
		for i := range probs {
			if probs[i] < cutoff {
				probs[i] = 0
			}
		}
		// Renormalize.
		s2 := 0.0
		for _, v := range probs {
			s2 += v
		}
		if s2 > 0 {
			for i := range probs {
				probs[i] /= s2
			}
		}
	}

	// Sample.
	r := rng.Float64()
	acc := 0.0
	for i, p := range probs {
		acc += p
		if r <= acc {
			return i
		}
	}
	// Fallback: highest probability.
	best := 0
	for i := 1; i < n; i++ {
		if probs[i] > probs[best] {
			best = i
		}
	}
	return best
}