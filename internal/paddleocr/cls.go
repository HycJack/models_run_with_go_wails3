package paddleocr

// ClsParams configures the text direction classifier.
type ClsParams struct {
	ImgW int
	ImgH int
	Mean [3]float32
	Std  [3]float32
}

func DefaultClsParams() ClsParams {
	return ClsParams{
		ImgW: 192,
		ImgH: 48,
		Mean: [3]float32{0.5, 0.5, 0.5},
		Std:  [3]float32{0.5, 0.5, 0.5},
	}
}

// ClsResult is the classifier output. Rotated reports whether the image
// should be rotated 180 degrees to correct its orientation.
type ClsResult struct {
	Rotated bool    `json:"rotated"`
	Label   int     `json:"label"`
	Score   float32 `json:"score"`
}

// ClsPostprocess interprets the model output (probabilities over 2 classes).
func ClsPostprocess(out []float32) ClsResult {
	if len(out) < 2 {
		return ClsResult{}
	}
	label := 0
	if out[1] > out[0] {
		label = 1
	}
	return ClsResult{
		Rotated: label == 1,
		Label:   label,
		Score:   out[label],
	}
}