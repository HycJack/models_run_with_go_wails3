package sensevoice

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// FbankOptions mirrors the SenseVoice frontend config (config.yaml
// frontend_conf) + kaldi defaults.
type FbankOptions struct {
	SampleRate    int     // 16000
	FrameLengthMs float64 // 25
	FrameShiftMs  float64 // 10
	NumMels       int     // 80
	LowFreq       float64 // 20
	HighFreq      float64 // 0 -> nyquist
	LFRM          int     // 7
	LFRN          int     // 6
}

func DefaultFbankOptions() FbankOptions {
	return FbankOptions{
		SampleRate:    16000,
		FrameLengthMs: 25,
		FrameShiftMs:  10,
		NumMels:       80,
		LowFreq:       20,
		HighFreq:      0,
		LFRM:          7,
		LFRN:          6,
	}
}

// Fbank computes the log mel filterbank features for a waveform (already
// scaled by 32768), returning [frames, numMels].
func Fbank(waveform []float64, opts FbankOptions) [][]float32 {
	sr := opts.SampleRate
	frameLen := int(math.Round(opts.FrameLengthMs / 1000 * float64(sr)))
	frameShift := int(math.Round(opts.FrameShiftMs / 1000 * float64(sr)))
	n := len(waveform)

	numFrames := 0
	if n >= frameLen {
		numFrames = (n-frameLen)/frameShift + 1
	}
	if numFrames <= 0 {
		return nil
	}

	fftSize := nextPow2(frameLen)

	// Hamming window.
	hamming := make([]float64, frameLen)
	for i := 0; i < frameLen; i++ {
		hamming[i] = 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(frameLen-1))
	}

	// Precompute mel filterbank weights.
	fb := melFilterbank(fftSize, sr, opts)

	feats := make([][]float32, numFrames)
	re := make([]float64, fftSize)
	im := make([]float64, fftSize)
	for f := 0; f < numFrames; f++ {
		// Windowed frame.
		for i := 0; i < fftSize; i++ {
			if i < frameLen {
				re[i] = waveform[f*frameShift+i] * hamming[i]
			} else {
				re[i] = 0
			}
			im[i] = 0
		}
		fftRadix2(re, im, false)
		// Power spectrum (first half + nyquist).
		half := fftSize/2 + 1
		power := make([]float64, half)
		for k := 0; k < half; k++ {
			power[k] = re[k]*re[k] + im[k]*im[k]
		}
		// Mel bins with log floor 1.0 (kaldi FbankComputer).
		row := make([]float32, opts.NumMels)
		for b := 0; b < opts.NumMels; b++ {
			v := fb[b].sum(power)
			if v < 1.0 {
				v = 1.0
			}
			row[b] = float32(math.Log(v))
		}
		feats[f] = row
	}
	return feats
}

// LFR stacks consecutive frames (m=7, n=6) into 560-dim vectors and pads the
// last partial frame with the final frame.
func LFR(feats [][]float32, m, n int) [][]float32 {
	T := len(feats)
	if T == 0 {
		return nil
	}
	if m <= 1 {
		return feats
	}
	dim := len(feats[0])
	Tlfr := (T + n - 1) / n // ceil(T/n)

	// Left padding: tile first frame (m-1)/2 times.
	left := (m - 1) / 2
	padded := make([][]float32, 0, T+left)
	for i := 0; i < left; i++ {
		padded = append(padded, feats[0])
	}
	padded = append(padded, feats...)
	Tp := T + left

	out := make([][]float32, 0, Tlfr)
	for i := 0; i < Tlfr; i++ {
		start := i * n
		if m <= Tp-start {
			row := make([]float32, 0, m*dim)
			for j := start; j < start+m; j++ {
				row = append(row, padded[j]...)
			}
			out = append(out, row)
		} else {
			// Last partial frame: pad with the final padded frame.
			row := make([]float32, 0, m*dim)
			for j := start; j < Tp; j++ {
				row = append(row, padded[j]...)
			}
			for len(row) < m*dim {
				row = append(row, padded[Tp-1]...)
			}
			out = append(out, row)
		}
	}
	return out
}

// CMVN applies the mean-shift/rescale from am.mvn.
func CMVN(feats [][]float32, means, vars []float64) [][]float32 {
	out := make([][]float32, len(feats))
	for i, row := range feats {
		nr := make([]float32, len(row))
		for j, v := range row {
			// (v + means[j]) * vars[j] — means are the <AddShift> values
			// (i.e. -mean), vars the <Rescale> (i.e. 1/std).
			nr[j] = float32((float64(v) + means[j]) * vars[j])
		}
		out[i] = nr
	}
	return out
}

// ParseCMVN reads the Kaldi am.mvn file (AddShift/Rescale) into means/vars.
func ParseCMVN(data []byte) (means, vars []float64, err error) {
	lines := strings.Split(string(data), "\n")
	parseRow := func(which string) []float64 {
		for i := 0; i < len(lines); i++ {
			f := strings.Fields(lines[i])
			if len(f) == 0 {
				continue
			}
			if f[0] == which {
				// Value line follows: <LearnRateCoef> 1 <Begin> ... <End>
				for j := i + 1; j < len(lines); j++ {
					vf := strings.Fields(lines[j])
					if len(vf) > 3 && vf[0] == "<LearnRateCoef>" {
						vals := vf[3 : len(vf)-1]
						out := make([]float64, 0, len(vals))
						for _, s := range vals {
							if v, e := strconv.ParseFloat(s, 64); e == nil {
								out = append(out, v)
							}
						}
						return out
					}
				}
			}
		}
		return nil
	}
	means = parseRow("<AddShift>")
	vars = parseRow("<Rescale>")
	if len(means) == 0 || len(vars) == 0 {
		return nil, nil, fmt.Errorf("could not parse am.mvn (AddShift/Rescale)")
	}
	return means, vars, nil
}

// melFilterbank builds triangular mel filters over the power spectrum.
type melBin struct {
	indices []int
	weights []float64
}

func (b *melBin) sum(power []float64) float64 {
	s := 0.0
	for i := range b.indices {
		if b.indices[i] < len(power) {
			s += b.weights[i] * power[b.indices[i]]
		}
	}
	return s
}

func melFilterbank(fftSize, sr int, opts FbankOptions) []melBin {
	numBins := opts.NumMels
	highFreq := opts.HighFreq
	if highFreq <= 0 {
		highFreq = float64(sr) / 2
	}
	melLow := freqToMel(opts.LowFreq)
	melHigh := freqToMel(highFreq)

	// numBins+2 mel points.
	melPoints := make([]float64, numBins+2)
	for i := 0; i < numBins+2; i++ {
		melPoints[i] = melLow + (melHigh-melLow)*float64(i)/float64(numBins+1)
	}
	freqPoints := make([]float64, numBins+2)
	binPoints := make([]int, numBins+2)
	for i := 0; i < numBins+2; i++ {
		freqPoints[i] = melToFreq(melPoints[i])
		binPoints[i] = int(float64(fftSize+1) * freqPoints[i] / float64(sr))
	}

	bins := make([]melBin, numBins)
	for b := 0; b < numBins; b++ {
		left := binPoints[b]
		center := binPoints[b+1]
		right := binPoints[b+2]
		if center <= left {
			center = left + 1
		}
		if right <= center {
			right = center + 1
		}
		var mb melBin
		for i := left; i < center; i++ {
			w := float64(i-left) / float64(center-left)
			mb.indices = append(mb.indices, i)
			mb.weights = append(mb.weights, w)
		}
		for i := center; i <= right; i++ {
			w := float64(right-i) / float64(right-center)
			mb.indices = append(mb.indices, i)
			mb.weights = append(mb.weights, w)
		}
		bins[b] = mb
	}
	return bins
}

func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}