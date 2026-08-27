package paddleocr

import (
	"testing"
)

func TestDetectPostprocessTrace(t *testing.T) {
	const W, H = 640, 640
	prob := make([]float32, W*H)
	for y := 300; y < 330; y++ {
		for x := 250; x < 350; x++ {
			prob[y*W+x] = 0.9
		}
	}
	binary := make([]bool, W*H)
	for i, v := range prob {
		binary[i] = float64(v) > 0.3
	}
	labels := connectedComponents(binary, W, H)
	t.Logf("components: %d", len(labels))
	for _, cc := range labels {
		t.Logf("  component size: %d", len(cc))
		pts := make([]point, 0, len(cc))
		for _, idx := range cc {
			px := idx % W
			py := idx / W
			pts = append(pts, point{float64(px), float64(py)})
		}
		cx, cy, w, h, angle := minAreaRectOfPoints(pts)
		t.Logf("  minAreaRect: cx=%.1f cy=%.1f w=%.1f h=%.1f angle=%.3f", cx, cy, w, h, angle)
		poly := boxPolygon(cx, cy, w, h, angle)
		t.Logf("  polygon: %v", poly)
		perimeter := 2 * (w + h)
		offset := w * h * 1.5 / perimeter
		poly2 := unclip(poly, offset)
		t.Logf("  unclipped: %v offset=%.2f", poly2, offset)
		ncx, ncy, nw, nh, nangle := minAreaRect(poly2)
		t.Logf("  final: cx=%.1f cy=%.1f w=%.1f h=%.1f angle=%.3f", ncx, ncy, nw, nh, nangle)
		box := boxPolygon(ncx, ncy, nw, nh, nangle)
		t.Logf("  box: %v", box)
	}
}