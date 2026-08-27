package paddleocr

import (
	"math"
)

// DetParams configures detection model inference.
type DetParams struct {
	LimitSideLen int     // resize so the longest side equals this (default 960)
	Mean         [3]float32
	Std          [3]float32
	Thresh       float64 // binarization threshold on the probability map
	BoxThresh    float64 // minimum mean probability of a detected box
	UnclipRatio  float64 // expansion factor for detected polygons
	MaxSideLen   int     // upper bound used during post-processing
	MinBoxArea   int     // minimum pixel area for a valid box
}

func DefaultDetParams() DetParams {
	return DetParams{
		LimitSideLen: 960,
		Mean:         [3]float32{0.485, 0.456, 0.406},
		Std:          [3]float32{0.229, 0.224, 0.225},
		Thresh:       0.2,   // PP-OCRv6 det_db_thresh
		BoxThresh:    0.45,  // PP-OCRv6 box_thresh
		UnclipRatio:  1.4,   // PP-OCRv6 unclip_ratio
		MaxSideLen:   3000,
		MinBoxArea:   3,
	}
}

// DetectResult is a single detected text box in the original image
// coordinate space. Points are ordered clockwise from the top-left.
type DetectResult struct {
	Box        [4][2]float64 `json:"box"`
	Score      float64       `json:"score"`
	RectangleX float64       `json:"x"`
	RectangleY float64       `json:"y"`
	RectangleW float64       `json:"w"`
	RectangleH float64       `json:"h"`
	Angle      float64       `json:"angle"` // radians, text direction
}

// Detect runs the detection model on a pre-normalized CHW float tensor of
// the resized image and returns text boxes in resized-image coordinates.
// prob is the model output (1,1,H,W) already passed through sigmoid.
func DetectPostprocess(prob []float32, mapW, mapH int, params DetParams) []DetectResult {
	binary := make([]bool, mapW*mapH)
	for i, v := range prob {
		binary[i] = float64(v) > params.Thresh
	}
	labels := connectedComponents(binary, mapW, mapH)
	results := make([]DetectResult, 0)
	for _, cc := range labels {
		if len(cc) < params.MinBoxArea {
			continue
		}
		pts := make([]point, 0, len(cc))
		scoreSum := 0.0
		for _, idx := range cc {
			px := idx % mapW
			py := idx / mapW
			pts = append(pts, point{float64(px), float64(py)})
			scoreSum += float64(prob[idx])
		}
		cx, cy, w, h, angle := minAreaRectOfPoints(pts)
		area := w * h
		if area < float64(params.MinBoxArea) {
			continue
		}
		meanScore := scoreSum / float64(len(cc))
		if meanScore < params.BoxThresh {
			continue
		}
		// Unclip: expand the polygon by area*ratio/perimeter.
		perimeter := 2 * (w + h)
		if perimeter < 1e-9 {
			continue
		}
		offset := area * params.UnclipRatio / perimeter
		poly := boxPolygon(cx, cy, w, h, angle)
		poly = unclip(poly, offset)
		ncx, ncy, nw, nh, nangle := minAreaRect(poly)
		if nHArea(nw, nh) < float64(params.MinBoxArea) {
			continue
		}
		box := boxPolygon(ncx, ncy, nw, nh, nangle)
		results = append(results, DetectResult{
			Box:        box,
			Score:      meanScore,
			RectangleX: ncx - nw/2,
			RectangleY: ncy - nh/2,
			RectangleW: nw,
			RectangleH: nh,
			Angle:      nangle,
		})
	}
	return results
}

func nHArea(w, h float64) float64 {
	if math.IsNaN(w) || math.IsNaN(h) {
		return 0
	}
	return w * h
}

// connectedComponents labels foreground pixels and returns a list of
// pixel-index sets.
func connectedComponents(binary []bool, w, h int) [][]int {
	visited := make([]byte, w*h)
	dirs := [][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}}
	var out [][]int
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			if !binary[idx] || visited[idx] == 1 {
				continue
			}
			// BFS
			var comp []int
			queue := []int{idx}
			visited[idx] = 1
			for len(queue) > 0 {
				cur := queue[0]
				queue = queue[1:]
				comp = append(comp, cur)
				cx := cur % w
				cy := cur / w
				for _, d := range dirs {
					nx := cx + d[0]
					ny := cy + d[1]
					if nx < 0 || ny < 0 || nx >= w || ny >= h {
						continue
					}
					ni := ny*w + nx
					if binary[ni] && visited[ni] == 0 {
						visited[ni] = 1
						queue = append(queue, ni)
					}
				}
			}
			if len(comp) > 0 {
				out = append(out, comp)
			}
		}
	}
	return out
}