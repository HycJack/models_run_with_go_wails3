package paddleocr

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	_ "image/gif"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/image/draw"
)

// Image wraps an RGBA image with its dimensions.
type Image struct {
	RGBA *image.RGBA
	W    int
	H    int
}

// LoadImage reads a PNG/JPEG/GIF/BMP image from disk.
func LoadImage(path string) (*Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode image %s: %w", filepath.Base(path), err)
	}
	return FromImage(img), nil
}

// FromImage converts any image.Image into an RGBA-backed Image.
func FromImage(src image.Image) *Image {
	b := src.Bounds()
	rgba := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(rgba, rgba.Bounds(), src, b.Min, draw.Src)
	return &Image{RGBA: rgba, W: b.Dx(), H: b.Dy()}
}

// Resize scales the image to the given size using bilinear interpolation.
func (im *Image) Resize(w, h int) *Image {
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.CatmullRom.Scale(dst, dst.Bounds(), im.RGBA, im.RGBA.Bounds(), draw.Over, nil)
	return &Image{RGBA: dst, W: w, H: h}
}

// Rotate180 rotates the image 180 degrees.
func (im *Image) Rotate180() *Image {
	dst := image.NewRGBA(image.Rect(0, 0, im.W, im.H))
	for y := 0; y < im.H; y++ {
		for x := 0; x < im.W; x++ {
			dst.SetRGBA(im.W-1-x, im.H-1-y, im.RGBA.RGBAAt(x, y))
		}
	}
	return &Image{RGBA: dst, W: im.W, H: im.H}
}

// Rotate90CW rotates the image 90 degrees clockwise.
func (im *Image) Rotate90CW() *Image {
	dst := image.NewRGBA(image.Rect(0, 0, im.H, im.W))
	for y := 0; y < im.H; y++ {
		for x := 0; x < im.W; x++ {
			dst.SetRGBA(im.H-1-y, x, im.RGBA.RGBAAt(x, y))
		}
	}
	return &Image{RGBA: dst, W: im.H, H: im.W}
}

// Rotate90CCW rotates the image 90 degrees counter-clockwise.
func (im *Image) Rotate90CCW() *Image {
	dst := image.NewRGBA(image.Rect(0, 0, im.H, im.W))
	for y := 0; y < im.H; y++ {
		for x := 0; x < im.W; x++ {
			dst.SetRGBA(y, im.W-1-x, im.RGBA.RGBAAt(x, y))
		}
	}
	return &Image{RGBA: dst, W: im.H, H: im.W}
}

// RotateCW rotates the image by deg clockwise (0/90/180/270).
func (im *Image) RotateCW(deg int) *Image {
	switch ((deg % 360) + 360) % 360 {
	case 90:
		return im.Rotate90CW()
	case 180:
		return im.Rotate180()
	case 270:
		return im.Rotate90CCW()
	default:
		return im
	}
}

// ToFloatCHW converts the image into a normalized float32 CHW tensor in
// [0,1] range (RGB order).
func (im *Image) ToFloatCHW(scale float32, mean, std [3]float32) []float32 {
	return im.toFloatCHW(scale, mean, std, false)
}

// ToFloatCHWBGR converts the image into a normalized float32 CHW tensor with
// the red/blue channels swapped (BGR order), matching PaddleOCR's OpenCV-based
// preprocessing.
func (im *Image) ToFloatCHWBGR(scale float32, mean, std [3]float32) []float32 {
	return im.toFloatCHW(scale, mean, std, true)
}

func (im *Image) toFloatCHW(scale float32, mean, std [3]float32, bgr bool) []float32 {
	ch := 3
	out := make([]float32, ch*im.W*im.H)
	idx := 0
	for c := 0; c < ch; c++ {
		for y := 0; y < im.H; y++ {
			for x := 0; x < im.W; x++ {
				px := im.RGBA.RGBAAt(x, y)
				var v float32
				switch c {
				case 0:
					if bgr {
						v = float32(px.B)
					} else {
						v = float32(px.R)
					}
				case 1:
					v = float32(px.G)
				case 2:
					if bgr {
						v = float32(px.R)
					} else {
						v = float32(px.B)
					}
				}
				v = v*scale - mean[c]
				v /= std[c]
				out[idx] = v
				idx++
			}
		}
	}
	return out
}

// CropBox extracts a text region from the image given four corner points
// (clockwise from top-left). It samples the source with bilinear
// interpolation at the minimum-area bounding rectangle of the points.
func (im *Image) CropBox(pts [4][2]float64, targetW, targetH int) *Image {
	// Fit a minimum-area rotated rectangle around the four points.
	cx, cy, w, h, angle := minAreaRect(pts)

	// Rotation angle of the text itself (radians); text runs along the
	// rectangle's width axis.
	rot := -angle
	// Sample the destination grid in source coordinates.
	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	cos := math.Cos(rot)
	sin := math.Sin(rot)
	// We map destination (0..targetW-1, 0..targetH-1) over the text width
	// 'w' and height 'h'.
	scaleX := w / float64(targetW)
	scaleY := h / float64(targetH)
	halfW := w / 2
	halfH := h / 2
	for dy := 0; dy < targetH; dy++ {
		// boxY is the perpendicular offset (rows of the crop).
		boxY := (float64(dy)+0.5)*scaleY - halfH
		for dx := 0; dx < targetW; dx++ {
			// boxX is the text-axis offset (columns of the crop).
			boxX := (float64(dx)+0.5)*scaleX - halfW
			// Rotate the box coordinates by the angle to get image coords.
			sx := cx + boxX*cos - boxY*sin
			sy := cy + boxX*sin + boxY*cos
			r, g, b := im.Sample(sx, sy)
			dst.SetRGBA(dx, dy, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}
	return &Image{RGBA: dst, W: targetW, H: targetH}
}

// Sample bilinearly samples the source image at (x, y).
func (im *Image) Sample(x, y float64) (uint8, uint8, uint8) {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1
	fx := x - float64(x0)
	fy := y - float64(y0)
	get := func(px, py int) color.RGBA {
		if px < 0 {
			px = 0
		}
		if py < 0 {
			py = 0
		}
		if px >= im.W {
			px = im.W - 1
		}
		if py >= im.H {
			py = im.H - 1
		}
		return im.RGBA.RGBAAt(px, py)
	}
	c00 := get(x0, y0)
	c10 := get(x1, y0)
	c01 := get(x0, y1)
	c11 := get(x1, y1)
	lerp := func(a, b float64) uint8 {
		return uint8(a*(1-fx) + b*fx)
	}
	lerpY := func(a, b uint8) uint8 {
		return uint8(float64(a)*(1-fy) + float64(b)*fy)
	}
	r := lerpY(lerp(float64(c00.R), float64(c10.R)), lerp(float64(c01.R), float64(c11.R)))
	g := lerpY(lerp(float64(c00.G), float64(c10.G)), lerp(float64(c01.G), float64(c11.G)))
	b := lerpY(lerp(float64(c00.B), float64(c10.B)), lerp(float64(c01.B), float64(c11.B)))
	return r, g, b
}

// EncodePNG encodes the image as PNG bytes.
func (im *Image) EncodePNG() ([]byte, error) {
	var buf strings.Builder
	if err := png.Encode(&byteWriter{&buf}, im.RGBA); err != nil {
		return nil, err
	}
	return []byte(buf.String()), nil
}

// SavePNG writes the image to disk as PNG.
func (im *Image) SavePNG(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, im.RGBA)
}

// SaveJPEG writes the image to disk as JPEG.
func (im *Image) SaveJPEG(path string, quality int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return jpeg.Encode(f, im.RGBA, &jpeg.Options{Quality: quality})
}

type byteWriter struct{ sb *strings.Builder }

func (w *byteWriter) Write(p []byte) (int, error) {
	return w.sb.Write(p)
}