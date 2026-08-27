package paddleocr

import (
	"math"
	"testing"
)

func TestMinAreaRectAxisAligned(t *testing.T) {
	pts := []point{{250, 300}, {350, 300}, {350, 330}, {250, 330}}
	cx, cy, w, h, angle := minAreaRectOfPoints(pts)
	t.Logf("rect: cx=%.1f cy=%.1f w=%.1f h=%.1f angle=%.3f", cx, cy, w, h, angle)
	if math.Abs(w-100) > 1 || math.Abs(h-30) > 1 {
		t.Fatalf("expected 100x30, got %.1fx%.1f", w, h)
	}
	// Angle should be 0 or 90 degrees (within a small tolerance).
	deg := angle * 180 / math.Pi
	if math.Abs(deg) > 2 && math.Abs(math.Abs(deg)-90) > 2 {
		t.Fatalf("expected axis-aligned angle, got %.1f deg", deg)
	}
}

func TestConvexHull(t *testing.T) {
	pts := []point{{250, 300}, {350, 300}, {350, 330}, {250, 330}, {300, 315}}
	hull := convexHull(pts)
	t.Logf("hull: %v", hull)
	if len(hull) != 4 {
		t.Fatalf("expected 4 hull points, got %d", len(hull))
	}
}