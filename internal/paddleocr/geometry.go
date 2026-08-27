package paddleocr

import (
	"math"
	"sort"
)

type point struct {
	x, y float64
}

// convexHull returns the convex hull of points using the monotone chain
// algorithm. The result is in counter-clockwise order.
func convexHull(pts []point) []point {
	if len(pts) < 3 {
		return pts
	}
	sort.Slice(pts, func(i, j int) bool {
		if pts[i].x != pts[j].x {
			return pts[i].x < pts[j].x
		}
		return pts[i].y < pts[j].y
	})
	var lower, upper []point
	cross := func(o, a, b point) float64 {
		return (a.x-o.x)*(b.y-o.y) - (a.y-o.y)*(b.x-o.x)
	}
	for _, p := range pts {
		for len(lower) >= 2 && cross(lower[len(lower)-2], lower[len(lower)-1], p) <= 0 {
			lower = lower[:len(lower)-1]
		}
		lower = append(lower, p)
	}
	for i := len(pts) - 1; i >= 0; i-- {
		p := pts[i]
		for len(upper) >= 2 && cross(upper[len(upper)-2], upper[len(upper)-1], p) <= 0 {
			upper = upper[:len(upper)-1]
		}
		upper = append(upper, p)
	}
	hull := lower[:len(lower)-1]
	hull = append(hull, upper[:len(upper)-1]...)
	return hull
}

// minAreaRect computes the minimum-area bounding rectangle of four points.
// It returns center (cx, cy), width, height and rotation angle in radians.
func minAreaRect(pts [4][2]float64) (cx, cy, w, h, angle float64) {
	ps := make([]point, len(pts))
	for i, p := range pts {
		ps[i] = point{p[0], p[1]}
	}
	return minAreaRectOfPoints(ps)
}

// minAreaRectOfPoints computes the minimum-area bounding rectangle of an
// arbitrary point set via convex hull + rotating calipers.
func minAreaRectOfPoints(ps []point) (cx, cy, w, h, angle float64) {
	hull := convexHull(ps)
	if len(hull) < 3 {
		if len(hull) == 1 {
			return hull[0].x, hull[0].y, 1, 1, 0
		}
		// Two points: degenerate rectangle.
		d := math.Hypot(hull[1].x-hull[0].x, hull[1].y-hull[0].y)
		mx := (hull[0].x + hull[1].x) / 2
		my := (hull[0].y + hull[1].y) / 2
		return mx, my, d, 1, 0
	}
	n := len(hull)
	bestArea := math.Inf(1)
	for i := 0; i < n; i++ {
		a := hull[i]
		b := hull[(i+1)%n]
		edgeX := b.x - a.x
		edgeY := b.y - a.y
		len2 := edgeX*edgeX + edgeY*edgeY
		if len2 < 1e-12 {
			continue
		}
		// Unit edge and normal.
		ux, uy := edgeX/math.Sqrt(len2), edgeY/math.Sqrt(len2)
		nx, ny := -uy, ux
		minProjU, maxProjU := math.Inf(1), math.Inf(-1)
		minProjN, maxProjN := math.Inf(1), math.Inf(-1)
		for _, p := range hull {
			pu := (p.x-a.x)*ux + (p.y-a.y)*uy
			pn := (p.x-a.x)*nx + (p.y-a.y)*ny
			minProjU = math.Min(minProjU, pu)
			maxProjU = math.Max(maxProjU, pu)
			minProjN = math.Min(minProjN, pn)
			maxProjN = math.Max(maxProjN, pn)
		}
		width := maxProjU - minProjU
		height := maxProjN - minProjN
		area := width * height
		if area < bestArea {
			bestArea = area
			w = width
			h = height
			// Center is at (minProjU+width/2) along u and (minProjN+height/2) along n.
			cu := minProjU + width/2
			cn := minProjN + height/2
			cx = a.x + cu*ux + cn*nx
			cy = a.y + cu*uy + cn*ny
			angle = math.Atan2(uy, ux)
		}
	}
	// Ensure the width is the longer (text) axis, rotating the angle by 90°
	// when the rectangle is taller than wide so that boxPolygon/CropBox treat
	// the width axis as the reading direction.
	if w < h {
		w, h = h, w
		angle += math.Pi / 2
	}
	return cx, cy, w, h, angle
}

// boxPolygon returns the 4 corner points of a rotated rectangle in
// clockwise order starting from the top-left relative to the text axis.
func boxPolygon(cx, cy, w, h, angle float64) [4][2]float64 {
	cos := math.Cos(angle)
	sin := math.Sin(angle)
	hw, hh := w/2, h/2
	// Corners in (text-axis, perpendicular) box space.
	corners := [4][2]float64{
		{-hw, -hh}, // top-left
		{hw, -hh},  // top-right
		{hw, hh},   // bottom-right
		{-hw, hh},  // bottom-left
	}
	var out [4][2]float64
	for i, c := range corners {
		// Rotate so the text axis follows the box angle in image space.
		x := cx + c[0]*cos - c[1]*sin
		y := cy + c[0]*sin + c[1]*cos
		out[i] = [2]float64{x, y}
	}
	return out
}

// unclip expands a polygon outward by distance d along the direction from the
// polygon centroid to each vertex.
func unclip(pts [4][2]float64, d float64) [4][2]float64 {
	var cxx, cyy float64
	for _, p := range pts {
		cxx += p[0]
		cyy += p[1]
	}
	cxx /= float64(len(pts))
	cyy /= float64(len(pts))
	out := pts
	for i := range pts {
		dx := pts[i][0] - cxx
		dy := pts[i][1] - cyy
		l := math.Hypot(dx, dy)
		if l < 1e-9 {
			continue
		}
		out[i][0] = pts[i][0] + dx/l*d
		out[i][1] = pts[i][1] + dy/l*d
	}
	return out
}