package paddleocr

// OriParams configures the document orientation classifier (PP-LCNet doc_ori).
type OriParams struct {
	ImgW int
	ImgH int
	Mean [3]float32
	Std  [3]float32
}

func DefaultOriParams() OriParams {
	return OriParams{
		ImgW: 224,
		ImgH: 224,
		Mean: [3]float32{0.485, 0.456, 0.406},
		Std:  [3]float32{0.229, 0.224, 0.225},
	}
}

// OriResult is the document orientation prediction.
type OriResult struct {
	Label int     `json:"label"` // 0/1/2/3 -> 0/90/180/270 degrees clockwise
	Angle int     `json:"angle"` // degrees the document content is rotated clockwise
	Score float32 `json:"score"`
}

// OriPostprocess interprets the model output (probabilities over 4 classes).
func OriPostprocess(out []float32) OriResult {
	if len(out) < 4 {
		return OriResult{}
	}
	best := 0
	for i := 1; i < 4; i++ {
		if out[i] > out[best] {
			best = i
		}
	}
	return OriResult{Label: best, Angle: best * 90, Score: out[best]}
}

// correctCW returns the clockwise rotation (0/90/180/270) to apply to the
// input image so the document becomes upright.
func (r OriResult) correctCW() int {
	return (360 - r.Label*90) % 360
}