package ort

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"cpm_orc/internal/config"
)

var (
	proxyMu   sync.RWMutex
	proxyURL  string
)

// SetProxy sets the proxy URL used by runtime downloads (e.g.
// "http://127.0.0.1:7890"). Empty clears it, falling back to environment
// proxy variables.
func SetProxy(proxy string) {
	proxyMu.Lock()
	proxyURL = proxy
	proxyMu.Unlock()
}

func getProxy() string {
	proxyMu.RLock()
	defer proxyMu.RUnlock()
	return proxyURL
}

// RuntimeAsset describes the ONNX Runtime release asset for the host platform.
type RuntimeAsset struct {
	Version string // e.g. "1.22.0"
	URL     string
	Archive string // .tgz or .zip
}

// LatestRuntimeVersion is the ONNX Runtime release this app expects.
// It must satisfy the API version requested by the onnxruntime_go binding.
const LatestRuntimeVersion = "1.29.0"

// AssetForPlatform returns the release asset for the current OS/arch.
func AssetForPlatform(version string) (*RuntimeAsset, error) {
	if version == "" {
		version = LatestRuntimeVersion
	}
	base := fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/", version)
	switch runtime.GOOS {
	case "darwin":
		arch := "universal2"
		if runtime.GOARCH == "arm64" {
			arch = "arm64"
		}
		name := fmt.Sprintf("onnxruntime-osx-%s-%s.tgz", arch, version)
		return &RuntimeAsset{Version: version, URL: base + name, Archive: ".tgz"}, nil
	case "linux":
		arch := "x64"
		if runtime.GOARCH == "arm64" {
			arch = "aarch64"
		}
		name := fmt.Sprintf("onnxruntime-linux-%s-%s.tgz", arch, version)
		return &RuntimeAsset{Version: version, URL: base + name, Archive: ".tgz"}, nil
	case "windows":
		arch := "x64"
		if runtime.GOARCH == "arm64" {
			arch = "arm64"
		}
		name := fmt.Sprintf("onnxruntime-win-%s-%s.zip", arch, version)
		return &RuntimeAsset{Version: version, URL: base + name, Archive: ".zip"}, nil
	default:
		return nil, fmt.Errorf("unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

// DownloadRuntime downloads and extracts the ONNX Runtime shared library into
// dstDir. Returns the path of the extracted library.
func DownloadRuntime(dstDir string, version string, onProgress func(done, total int64)) (string, error) {
	asset, err := AssetForPlatform(version)
	if err != nil {
		return "", err
	}
	tmp, err := os.CreateTemp("", "onnxruntime-dl-*"+asset.Archive)
	if err != nil {
		return "", err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	req, err := http.NewRequest(http.MethodGet, asset.URL, nil)
	if err != nil {
		return "", err
	}
	client := config.HTTPClient(getProxy(), 0)
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download failed (%d): %s", resp.StatusCode, asset.URL)
	}
	total := resp.ContentLength
	written := int64(0)
	_, err = io.Copy(tmp, io.TeeReader(resp.Body, &writeCounter{fn: func(n int64) {
		written += n
		if onProgress != nil {
			onProgress(written, total)
		}
	}}))
	if err != nil {
		tmp.Close()
		return "", err
	}
	tmp.Close()

	extractDir := filepath.Join(dstDir, "extract")
	os.MkdirAll(extractDir, 0o755)
	if asset.Archive == ".zip" {
		err = extractZip(tmpPath, extractDir)
	} else {
		err = extractTarGz(tmpPath, extractDir)
	}
	if err != nil {
		return "", err
	}

	// Find the shared library inside the extraction.
	lib, err := findSharedLibrary(extractDir)
	if err != nil {
		return "", err
	}
	dest := filepath.Join(dstDir, LibFileName())
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return "", err
	}
	if err := os.Remove(dest); err != nil && !os.IsNotExist(err) {
		return "", err
	}
	if err := os.Rename(lib, dest); err != nil {
		return "", err
	}
	os.RemoveAll(extractDir)
	return dest, nil
}

func extractTarGz(src, dst string) error {
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		// Only regular files are extracted. Symlinks and other special
		// entries are skipped; the shared library is copied separately.
		if hdr.Typeflag != tar.TypeReg && hdr.Typeflag != tar.TypeRegA {
			continue
		}
		target := filepath.Join(dst, filepath.Clean(hdr.Name))
		if hdr.FileInfo().IsDir() {
			os.MkdirAll(target, 0o755)
			continue
		}
		os.MkdirAll(filepath.Dir(target), 0o755)
		out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, hdr.FileInfo().Mode())
		if err != nil {
			return err
		}
		if _, err := io.Copy(out, tr); err != nil {
			out.Close()
			return err
		}
		out.Close()
	}
	return nil
}

func extractZip(src, dst string) error {
	zr, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer zr.Close()
	for _, f := range zr.File {
		target := filepath.Join(dst, filepath.Clean(f.Name))
		if f.FileInfo().IsDir() {
			os.MkdirAll(target, 0o755)
			continue
		}
		os.MkdirAll(filepath.Dir(target), 0o755)
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, f.Mode())
		if err != nil {
			rc.Close()
			return err
		}
		if _, err := io.Copy(out, rc); err != nil {
			out.Close()
			rc.Close()
			return err
		}
		out.Close()
		rc.Close()
	}
	return nil
}

func findSharedLibrary(dir string) (string, error) {
	type cand struct {
		path string
		size int64
	}
	var candidates []cand
	_ = filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || info.Size() == 0 {
			return nil
		}
		// Skip debug symbol bundles (dSYM).
		if strings.Contains(strings.ToLower(p), ".dsym") {
			return nil
		}
		name := strings.ToLower(info.Name())
		switch {
		case strings.Contains(name, "onnxruntime") && strings.HasSuffix(name, ".dylib"):
			candidates = append(candidates, cand{p, info.Size()})
		case strings.Contains(name, "onnxruntime") && strings.HasSuffix(name, ".so"):
			candidates = append(candidates, cand{p, info.Size()})
		case strings.Contains(name, "onnxruntime") && strings.HasSuffix(name, ".dll"):
			candidates = append(candidates, cand{p, info.Size()})
		}
		return nil
	})
	if len(candidates) == 0 {
		return "", fmt.Errorf("onnxruntime shared library not found in archive")
	}
	// Prefer a file whose name is exactly the expected library name.
	base := LibFileName()
	for _, c := range candidates {
		if filepath.Base(c.path) == base {
			return c.path, nil
		}
	}
	// Otherwise prefer the largest (versioned) file.
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].size > candidates[j].size })
	return candidates[0].path, nil
}

type writeCounter struct{ fn func(int64) }

func (w *writeCounter) Write(b []byte) (int, error) {
	w.fn(int64(len(b)))
	return len(b), nil
}

// RuntimeStatus describes whether a runtime library is present locally.
type RuntimeStatus struct {
	Present bool   `json:"present"`
	Path    string `json:"path"`
	Version string `json:"version"`
	Size    int64  `json:"size"`
}

// CheckRuntime returns the status of the ONNX Runtime library at path.
func CheckRuntime(path string) RuntimeStatus {
	st := RuntimeStatus{Path: path, Version: LatestRuntimeVersion}
	if info, err := os.Stat(path); err == nil {
		st.Present = true
		st.Size = info.Size()
	}
	return st
}