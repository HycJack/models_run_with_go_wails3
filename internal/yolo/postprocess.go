package yolo

import "math"

// postprocessDualHead decodes the transformers.js-style YOLO26 ONNX export,
// which emits two tensors instead of one:
//
//	logits     [1, N, numClasses] - raw (pre-sigmoid) class scores
//	pred_boxes [1, N, 4]          - cx, cy, w, h normalized to 0..1 of the
//	                                letterboxed input square
//
// Boxes are mapped back to original image pixel coordinates using the
// letterbox geometry.
func postprocessDualHead(
	logits []float32, logitsShape []int64,
	boxes []float32, boxesShape []int64,
	origW, origH int, scale float64, padX, padY, inputW, inputH int,
	names []string,
) []Detection {
	if len(logitsShape) != 3 || len(boxesShape) != 3 {
		return nil
	}
	numDets := int(logitsShape[1])
	numClasses := int(logitsShape[2])
	if numDets == 0 || numClasses == 0 || int(boxesShape[1]) != numDets {
		return nil
	}

	dets := make([]Detection, 0, numDets)
	for i := 0; i < numDets; i++ {
		// Best class by raw logit, then convert that one logit to a probability.
		base := i * numClasses
		bestClass := 0
		bestLogit := logits[base]
		for c := 1; c < numClasses; c++ {
			if v := logits[base+c]; v > bestLogit {
				bestLogit = v
				bestClass = c
			}
		}
		score := sigmoid(float64(bestLogit))
		if score < 0.001 {
			continue
		}

		// Normalized cx, cy, w, h -> letterbox pixels.
		b := i * 4
		cx := float64(boxes[b+0]) * float64(inputW)
		cy := float64(boxes[b+1]) * float64(inputH)
		w := float64(boxes[b+2]) * float64(inputW)
		h := float64(boxes[b+3]) * float64(inputH)

		x0 := (cx - w/2 - float64(padX)) / scale
		y0 := (cy - h/2 - float64(padY)) / scale
		x1 := (cx + w/2 - float64(padX)) / scale
		y1 := (cy + h/2 - float64(padY)) / scale

		x0 = clamp(x0, 0, float64(origW))
		y0 = clamp(y0, 0, float64(origH))
		x1 = clamp(x1, 0, float64(origW))
		y1 = clamp(y1, 0, float64(origH))
		if x1 <= x0 || y1 <= y0 {
			continue
		}

		label := ""
		if bestClass < len(names) {
			label = names[bestClass]
		}
		dets = append(dets, Detection{
			Box:   [4]float64{x0, y0, x1, y1},
			Class: bestClass,
			Score: score,
			Label: label,
		})
	}
	return dets
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// nms applies non-maximum suppression to remove overlapping detections.
// Suppression is per-class so different objects that overlap are kept.
func nms(dets []Detection, iouThresh float64) []Detection {
	if len(dets) <= 1 {
		return dets
	}

	// Sort by score descending (insertion sort; N is at most a few hundred).
	for i := 1; i < len(dets); i++ {
		key := dets[i]
		j := i - 1
		for j >= 0 && dets[j].Score < key.Score {
			dets[j+1] = dets[j]
			j--
		}
		dets[j+1] = key
	}

	suppressed := make([]bool, len(dets))
	var result []Detection
	for i := range dets {
		if suppressed[i] {
			continue
		}
		result = append(result, dets[i])
		for j := i + 1; j < len(dets); j++ {
			if suppressed[j] || dets[j].Class != dets[i].Class {
				continue
			}
			if iou(dets[i].Box, dets[j].Box) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return result
}

// iou computes intersection-over-union of two bounding boxes.
func iou(a, b [4]float64) float64 {
	x0 := math.Max(a[0], b[0])
	y0 := math.Max(a[1], b[1])
	x1 := math.Min(a[2], b[2])
	y1 := math.Min(a[3], b[3])

	inter := math.Max(0, x1-x0) * math.Max(0, y1-y0)
	areaA := (a[2] - a[0]) * (a[3] - a[1])
	areaB := (b[2] - b[0]) * (b[3] - b[1])
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}
