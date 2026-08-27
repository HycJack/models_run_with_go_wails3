package ort

import (
	"os"
	"path/filepath"
	"testing"
)

// TestDownloadAndInit verifies the runtime provisioning path. It only runs
// when CPMM_RUNTIME_TEST=1 is set to avoid network access in normal runs.
func TestDownloadAndInit(t *testing.T) {
	if os.Getenv("CPMM_RUNTIME_TEST") != "1" {
		t.Skip("set CPMM_RUNTIME_TEST=1 to run")
	}
	dir := t.TempDir()
	path, err := DownloadRuntime(dir, "", nil)
	if err != nil {
		t.Fatalf("download: %v", err)
	}
	if err := Init(path); err != nil {
		t.Fatalf("init: %v", err)
	}
	if !IsInitialized() {
		t.Fatal("not initialized")
	}
	if Version() == "" {
		t.Fatal("empty version")
	}
	t.Logf("lib=%s version=%s", path, Version())
	_ = filepath.Base(path)
}