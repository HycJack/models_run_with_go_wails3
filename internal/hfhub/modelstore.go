package hfhub

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// LocalModel describes a model directory found under the model root.
type LocalModel struct {
	ID       string `json:"id"`
	Path     string `json:"path"`
	Size     int64  `json:"size"`
	Modified string `json:"modified"`
	Kind     string `json:"kind"` // llm, onnx, paddle, hf, other
	Files    []string `json:"files"`
}

// ScanLocal walks root and returns model directories. A directory is
// considered a model when it directly contains a marker file (config.json,
// tokenizer.json, *.onnx or *.pdmodel). Model IDs are relative to root, so
// both "repoID" and "paddleocr/ch" style layouts are reported.
func ScanLocal(root string) ([]LocalModel, error) {
	var out []LocalModel
	entries, err := os.ReadDir(root)
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil
		}
		return nil, err
	}
	for _, e := range entries {
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		dir := filepath.Join(root, e.Name())
		collectModelDirs(root, dir, &out)
	}
	sort.Slice(out, func(i, j int) bool {
		return strings.ToLower(out[i].ID) < strings.ToLower(out[j].ID)
	})
	return out, nil
}

// collectModelDirs recursively finds model directories and appends them to
// out, skipping parent directories that are themselves models.
func collectModelDirs(root, dir string, out *[]LocalModel) {
	subs, _ := os.ReadDir(dir)
	isModel := false
	for _, s := range subs {
		if s.IsDir() || strings.HasPrefix(s.Name(), ".") {
			continue
		}
		if isModelMarker(s.Name()) {
			isModel = true
			break
		}
	}
	if isModel {
		lm := LocalModel{
			ID:   filepath.ToSlash(strings.TrimPrefix(dir, root+string(filepath.Separator))),
			Path: dir,
		}
		if info, err := os.Stat(dir); err == nil {
			lm.Modified = info.ModTime().Format(time.RFC3339)
		}
		lm.Size, lm.Kind, lm.Files = analyzeDir(dir)
		*out = append(*out, lm)
		return
	}
	// Recurse one level into subdirectories.
	for _, s := range subs {
		if !s.IsDir() || strings.HasPrefix(s.Name(), ".") {
			continue
		}
		collectModelDirs(root, filepath.Join(dir, s.Name()), out)
	}
}

func isModelMarker(name string) bool {
	switch name {
	case "config.json", "tokenizer.json":
		return true
	}
	switch strings.ToLower(filepath.Ext(name)) {
	case ".onnx", ".pdmodel", ".pdiparams":
		return true
	}
	return false
}

func analyzeDir(dir string) (int64, string, []string) {
	var total int64
	var files []string
	hasConfig := false
	hasTokenizer := false
	hasOnnx := false
	hasPaddle := false
	// Walk up to 2 levels deep.
	filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(dir, p)
		if strings.Count(rel, string(filepath.Separator)) > 1 {
			return nil
		}
		name := d.Name()
		ext := strings.ToLower(filepath.Ext(name))
		switch name {
		case "config.json":
			hasConfig = true
		case "tokenizer.json":
			hasTokenizer = true
		}
		switch ext {
		case ".onnx":
			hasOnnx = true
		case ".pdmodel", ".pdiparams":
			hasPaddle = true
		}
		if ext == ".onnx" || ext == ".json" || ext == ".txt" || ext == ".pdmodel" || ext == ".pdiparams" {
			files = append(files, rel)
		}
		if info, err := d.Info(); err == nil {
			total += info.Size()
		}
		return nil
	})
	kind := "other"
	switch {
	case hasConfig && hasOnnx && hasTokenizer:
		kind = "llm"
	case hasOnnx:
		kind = "onnx"
	case hasPaddle:
		kind = "paddle"
	case hasConfig:
		kind = "hf"
	}
	return total, kind, files
}

// DirSize recursively computes the total size of a directory.
func DirSize(dir string) int64 {
	var total int64
	filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if info, err := d.Info(); err == nil {
			total += info.Size()
		}
		return nil
	})
	return total
}