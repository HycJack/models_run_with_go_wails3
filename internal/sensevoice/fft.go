package sensevoice

import "math"

// fftRadix2 performs an in-place radix-2 complex FFT. real/imag hold the
// samples (length must be a power of two). inverse selects IFFT.
func fftRadix2(real, imag []float64, inverse bool) {
	n := len(real)
	// Bit reversal permutation.
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			real[i], real[j] = real[j], real[i]
			imag[i], imag[j] = imag[j], imag[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		ang := 2 * math.Pi / float64(length)
		if inverse {
			ang = -ang
		}
		wr, wi := math.Cos(ang), math.Sin(ang)
		for i := 0; i < n; i += length {
			curR, curI := 1.0, 0.0
			for j := 0; j < length/2; j++ {
				uR := real[i+j]
				uI := imag[i+j]
				vR := real[i+j+length/2]*curR - imag[i+j+length/2]*curI
				vI := real[i+j+length/2]*curI + imag[i+j+length/2]*curR
				real[i+j] = uR + vR
				imag[i+j] = uI + vI
				real[i+j+length/2] = uR - vR
				imag[i+j+length/2] = uI - vI
				curR, curI = curR*wr-curI*wi, curR*wi+curI*wr
			}
		}
	}
	if inverse {
		for i := 0; i < n; i++ {
			real[i] /= float64(n)
			imag[i] /= float64(n)
		}
	}
}

// melToFreq converts mel to Hz using the standard (Slaney) mel scale.
func melToFreq(mel float64) float64 {
	return 700.0 * (math.Exp(mel/1127.0) - 1.0)
}

// freqToMel converts Hz to mel using the standard (Slaney) mel scale.
func freqToMel(freq float64) float64 {
	return 1127.0 * math.Log(1.0+freq/700.0)
}