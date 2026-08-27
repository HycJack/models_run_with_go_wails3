package paddleocr

import (
	"math"
	"testing"
)

// TestDetectPostprocessSynthetic ensures a solid rectangle in the probability
// map yields a detected box.
func TestDetectPostprocessSynthetic(t *testing.T) {
	const W, H = 640, 640
	prob := make([]float32, W*H)
	// A filled rectangle 100x30 near the center.
	for y := 300; y < 330; y++ {
		for x := 250; x < 350; x++ {
			prob[y*W+x] = 0.9
		}
	}
	params := DefaultDetParams()
	params.BoxThresh = 0.5
	params.MinBoxArea = 50
	res := DetectPostprocess(prob, W, H, params)
	if len(res) == 0 {
		t.Fatalf("expected at least one box, got 0")
	}
	t.Logf("boxes: %d", len(res))
	for _, r := range res {
		t.Logf("  box=%v score=%.2f w=%.1f h=%.1f angle=%.3f",
			r.Box, r.Score, r.RectangleW, r.RectangleH, r.Angle)
		if math.IsNaN(r.RectangleW) || math.IsNaN(r.RectangleH) {
			t.Fatalf("NaN in rectangle dims")
		}
	}
}

// TestDetectPostprocessLowThresh checks that noisy maps don't crash.
func TestDetectPostprocessNoise(t *testing.T) {
	const W, H = 128, 128
	prob := make([]float32, W*H)
	for i := range prob {
		prob[i] = 0.5
	}
	params := DefaultDetParams()
	params.BoxThresh = 0.8
	res := DetectPostprocess(prob, W, H, params)
	t.Logf("noise boxes: %d", len(res))
}