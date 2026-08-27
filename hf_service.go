package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"cpm_orc/internal/hfhub"
)

// HFHubService exposes HuggingFace Hub and local model management to the UI.
type HFHubService struct {
	state *State
}

// NewHFHubService creates the model manager service.
func NewHFHubService(s *State) *HFHubService {
	return &HFHubService{state: s}
}

// client returns a Hub client wired with the current proxy setting.
func (s *HFHubService) client() *hfhub.Client {
	return hfhub.NewClient(s.state.cfg.Proxy)
}

// SearchModel finds models on the Hub.
func (s *HFHubService) SearchModel(query string, limit int) ([]hfhub.Model, error) {
	return s.client().Search(query, limit, "")
}

// LocalModels lists the models stored under the model root.
func (s *HFHubService) LocalModels() ([]hfhub.LocalModel, error) {
	return hfhub.ScanLocal(s.state.ModelRoot())
}

// ModelFiles lists the file tree of a remote repository.
func (s *HFHubService) ModelFiles(repo, revision, subPath string) ([]hfhub.FileInfo, error) {
	if revision == "" {
		revision = "main"
	}
	return s.client().Files(repo, revision, subPath, true)
}

// ModelInfo fetches metadata for a repository.
func (s *HFHubService) ModelInfo(repo string) (*hfhub.Model, error) {
	return s.client().ModelInfo(repo)
}

// DownloadModel downloads a set of files from a repository into the local
// model root (ModelRoot/<repoID>/...). When files is empty every file in the
// repository is downloaded. Progress events are emitted as "dl:progress".
func (s *HFHubService) DownloadModel(repoID string, files []string, revision string) (string, error) {
	if revision == "" {
		revision = "main"
	}
	if len(files) == 0 {
		info, err := s.client().Files(repoID, revision, "", true)
		if err != nil {
			return "", err
		}
		for _, f := range info {
			if f.Type == "file" {
				files = append(files, f.Path)
			}
		}
		if len(files) == 0 {
			return "", fmt.Errorf("no files found in %s", repoID)
		}
	}
	dest := filepath.Join(s.state.ModelRoot(), repoID)
	for _, f := range files {
		target := filepath.Join(dest, filepath.FromSlash(f))
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return "", err
		}
		s.state.Emit("dl:start", map[string]any{"id": repoID, "file": f})
		err := s.client().Download(repoID, revision, f, target, func(done, total int64) {
			s.state.Emit("dl:progress", map[string]any{
				"id":    repoID,
				"file":  f,
				"done":  done,
				"total": total,
			})
		})
		if err != nil {
			return "", err
		}
		s.state.Emit("dl:file-done", map[string]any{"id": repoID, "file": f})
	}
	s.state.Emit("dl:done", map[string]any{"id": repoID, "path": dest})
	return dest, nil
}

// DeleteModel removes a local model directory.
func (s *HFHubService) DeleteModel(id string) error {
	dir := filepath.Join(s.state.ModelRoot(), filepath.FromSlash(id))
	if !strings.HasPrefix(dir, s.state.ModelRoot()) {
		return fmt.Errorf("invalid model path")
	}
	if _, err := os.Stat(dir); err != nil {
		return err
	}
	return os.RemoveAll(dir)
}

// ModelRoot returns the local model root.
func (s *HFHubService) ModelRoot() string { return s.state.ModelRoot() }

// SetModelRoot changes the local model root and saves the config.
func (s *HFHubService) SetModelRoot(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	s.state.cfg.ModelRoot = dir
	return s.state.SaveConfig()
}

// OpenModel opens a local model directory in the file manager.
func (s *HFHubService) OpenModel(id string) error {
	return s.state.OpenFolder(filepath.Join(s.state.ModelRoot(), id))
}