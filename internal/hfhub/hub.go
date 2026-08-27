package hfhub

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"

	"cpm_orc/internal/config"
)

const HubURL = "https://huggingface.co"

// Client talks to the HuggingFace Hub API.
type Client struct {
	HTTP *http.Client
}

// NewClient creates a Hub client. When proxy is non-empty it is used for all
// requests; otherwise the environment's proxy settings apply. No client-level
// timeout is set so large model downloads are not interrupted.
func NewClient(proxy string) *Client {
	return &Client{
		HTTP: config.HTTPClient(proxy, 0),
	}
}

// Model is a single model listed by the Hub API.
type Model struct {
	ID          string   `json:"id"`
	Downloads   int64    `json:"downloads"`
	Likes       int64    `json:"likes"`
	PipelineTag string   `json:"pipeline_tag"`
	Tags        []string `json:"tags"`
	LastMod     string   `json:"lastModified"`
	Siblings    []struct {
		RFilename string `json:"rfilename"`
	} `json:"siblings"`
}

// Search queries the Hub model API.
func (c *Client) Search(query string, limit int, filter string) ([]Model, error) {
	q := url.Values{}
	q.Set("search", query)
	if limit <= 0 {
		limit = 20
	}
	q.Set("limit", strconv.Itoa(limit))
	q.Set("sort", "downloads")
	q.Set("direction", "-1")
	if filter != "" {
		q.Set("filter", filter)
	}
	u := fmt.Sprintf("%s/api/models?%s", HubURL, q.Encode())
	req, err := http.NewRequest(http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("huggingface search failed (%d): %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var out []Model
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// FileInfo is a single entry in a repository tree.
type FileInfo struct {
	Path    string `json:"path"`
	Type    string `json:"type"`
	Size    *int64 `json:"size,omitempty"`
	LFS     *struct {
		Size int64 `json:"size"`
	} `json:"lfs,omitempty"`
}

// Files lists the contents of a repository path.
func (c *Client) Files(repo string, revision, subPath string, recursive bool) ([]FileInfo, error) {
	u := fmt.Sprintf("%s/api/models/%s/tree/%s", HubURL, repo, strings.TrimPrefix(revision+"/"+subPath, "/"))
	u = strings.TrimSuffix(u, "/")
	if recursive {
		sep := "?"
		if strings.Contains(u, "?") {
			sep = "&"
		}
		u += sep + "recursive=true"
	}
	req, err := http.NewRequest(http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil, nil
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("list files failed (%d): %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var out []FileInfo
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// ModelInfo fetches the full metadata for a model.
func (c *Client) ModelInfo(repo string) (*Model, error) {
	u := fmt.Sprintf("%s/api/models/%s", HubURL, repo)
	req, err := http.NewRequest(http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("model info failed (%d)", resp.StatusCode)
	}
	var m Model
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}
	return &m, nil
}

// Download fetches a single resolved file from the hub, writing to dst.
// The onProgress callback receives bytes written so far and the total size.
func (c *Client) Download(repo, revision, filePath, dst string, onProgress func(done, total int64)) error {
	u := fmt.Sprintf("%s/%s/resolve/%s/%s", HubURL, repo, revision, filePath)
	req, err := http.NewRequest(http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "*/*")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("download %s failed (%d): %s", filePath, resp.StatusCode, strings.TrimSpace(string(b)))
	}
	total := resp.ContentLength
	if total <= 0 {
		// Fall back to the x-linked-size header used by HF LFS.
		if v := resp.Header.Get("X-Linked-Size"); v != "" {
			total, _ = strconv.ParseInt(v, 10, 64)
		}
	}
	written := int64(0)
	tee := io.TeeReader(resp.Body, &progressWriter{fn: func(n int64) {
		written += n
		if onProgress != nil {
			onProgress(written, total)
		}
	}})
	out, err := newFileWriter(dst)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, tee)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	return nil
}

// ResolveURL returns the canonical download URL for a file.
func ResolveURL(repo, revision, filePath string) string {
	return fmt.Sprintf("%s/%s/resolve/%s/%s", HubURL, repo, revision, filePath)
}

// JoinRepoPath cleans and joins sub-paths for a repo.
func JoinRepoPath(elems ...string) string {
	return path.Clean("/" + path.Join(elems...))
}

type progressWriter struct{ fn func(int64) }

func (p *progressWriter) Write(b []byte) (int, error) {
	p.fn(int64(len(b)))
	return len(b), nil
}