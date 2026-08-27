package sensevoice

import (
	"bytes"
	"encoding/binary"
	"fmt"
)

// ParseWav decodes a 16-bit PCM WAV (mono/stereo) into float samples in [-1,1].
func ParseWav(data []byte) ([]float32, error) {
	if len(data) < 44 || !bytes.Equal(data[:4], []byte("RIFF")) || !bytes.Equal(data[8:12], []byte("WAVE")) {
		return nil, fmt.Errorf("not a RIFF/WAVE file")
	}
	var sampleRate, channels uint32
	var pcm []byte
	i := 12
	for i+8 <= len(data) {
		chunkID := string(data[i : i+4])
		sz := binary.LittleEndian.Uint32(data[i+4 : i+8])
		body := i + 8
		if body+int(sz) > len(data) {
			break
		}
		switch chunkID {
		case "fmt ":
			if sz >= 16 {
				audioFormat := binary.LittleEndian.Uint16(data[body : body+2])
				if audioFormat != 1 && audioFormat != 0xFFFE {
					return nil, fmt.Errorf("unsupported audio format %d (only PCM)", audioFormat)
				}
				channels = uint32(binary.LittleEndian.Uint16(data[body+2 : body+4]))
				sampleRate = binary.LittleEndian.Uint32(data[body+4 : body+8])
			}
		case "data":
			pcm = data[body : body+int(sz)]
		}
		i = body + int(sz)
		if sz%2 == 1 {
			i++
		}
	}
	if len(pcm) == 0 {
		return nil, fmt.Errorf("no data chunk in WAV")
	}
	if sampleRate == 0 || channels == 0 {
		return nil, fmt.Errorf("invalid WAV header")
	}
	out := make([]float32, 0, len(pcm)/2/int(channels))
	for s := 0; s+1 < len(pcm); s += 2 {
		v := int16(binary.LittleEndian.Uint16(pcm[s : s+2]))
		out = append(out, float32(v)/32768.0)
	}
	return out, nil
}

// ResampleTo16k resamples a mono float signal to 16000 Hz using linear
// interpolation.
func ResampleTo16k(samples []float32, fromRate int) []float32 {
	if fromRate == 16000 {
		return samples
	}
	if fromRate <= 0 || len(samples) == 0 {
		return samples
	}
	n := int(float64(len(samples)) * 16000.0 / float64(fromRate))
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		pos := float64(i) * float64(fromRate) / 16000.0
		lo := int(pos)
		hi := lo + 1
		if hi >= len(samples) {
			hi = len(samples) - 1
		}
		frac := pos - float64(lo)
		out[i] = float32(float64(samples[lo])*(1-frac) + float64(samples[hi])*frac)
	}
	return out
}