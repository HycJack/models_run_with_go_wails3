package hfhub

import (
	"os"
	"path/filepath"
)

// fileWriter writes to a .part file so an interrupted download does not
// corrupt the final file. Closing the writer renames it into place.
type fileWriter struct {
	f     *os.File
	final string
	part  string
	done  bool
}

func newFileWriter(dst string) (*fileWriter, error) {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return nil, err
	}
	part := dst + ".part"
	f, err := os.OpenFile(part, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, err
	}
	return &fileWriter{f: f, final: dst, part: part}, nil
}

func (w *fileWriter) Write(b []byte) (int, error) {
	return w.f.Write(b)
}

func (w *fileWriter) Close() error {
	if w.done {
		return nil
	}
	w.done = true
	if err := w.f.Close(); err != nil {
		return err
	}
	if err := os.Rename(w.part, w.final); err != nil {
		os.Remove(w.part)
		return err
	}
	return nil
}