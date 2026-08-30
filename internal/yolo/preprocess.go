package yolo

import (
	"image"
	"math"
)

// preprocess letterbox-resizes an image to targetW x targetH, normalizes to
// [0,1], and rearranges from HWC to CHW float32 layout. Returns the padded
// image data plus geometry to map detections back to original coordinates.
func preprocess(img image.Image, targetW, targetH int) ([]float32, int, int, float64, int, int) {
	bounds := img.Bounds()
	origW := bounds.Dx()
	origH := bounds.Dy()

	// Compute scale to fit within target
	scaleX := float64(targetW) / float64(origW)
	scaleY := float64(targetH) / float64(origH)
	scale := math.Min(scaleX, scaleY)

	newW := float64(origW) * scale
	newH := float64(origH) * scale

	// Center padding (as int)
	padX := int((float64(targetW) - newW) / 2)
	padY := int((float64(targetH) - newH) / 2)

	// Create letterboxed image
	data := make([]float32, 3*targetH*targetW)

	// Fill with 0.5 (gray) for letterbox padding
	for i := range data {
		data[i] = 0.5
	}

	// Sample original image into the letterbox region
	for y := 0; y < targetH; y++ {
		for x := 0; x < targetW; x++ {
			// Map back to original coordinates
			srcX := int(float64(x-padX) / scale)
			srcY := int(float64(y-padY) / scale)

			if srcX < 0 || srcX >= origW || srcY < 0 || srcY >= origH {
				continue
			}

			r, g, b, _ := img.At(srcX+bounds.Min.X, srcY+bounds.Min.Y).RGBA()
			// Normalize to [0, 1]
			idx := y*targetW + x
			data[0*targetH*targetW+idx] = float32(r) / 65535.0
			data[1*targetH*targetW+idx] = float32(g) / 65535.0
			data[2*targetH*targetW+idx] = float32(b) / 65535.0
		}
	}

	return data, origW, origH, scale, padX, padY
}
