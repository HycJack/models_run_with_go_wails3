package main

import (
	"embed"
	"log"

	"github.com/wailsapp/wails/v3/pkg/application"

	"cpm_orc/internal/config"
	"cpm_orc/internal/ort"
	"cpm_orc/internal/paddleocr"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	cfg, err := config.Load(configPath())
	if err != nil {
		log.Printf("warning: failed to load config: %v", err)
	}
	if cfg == nil {
		cfg = config.Default()
	}
	_ = cfg.EnsureDirs()

	state := newState(cfg)
	// Apply the configured proxy to downloads from the start.
	ort.SetProxy(cfg.Proxy)
	// Enable CoreML execution provider for OCR on Apple Silicon.
	paddleocr.EnableCoreML = cfg.CoreML

	app := application.New(application.Options{
		Name:        "CPM OCR Studio",
		Description: "HuggingFace model manager, PaddleOCR and MiniCPM ONNX inference",
		Services: []application.Service{
			application.NewService(NewRuntimeService(state)),
			application.NewService(NewHFHubService(state)),
			application.NewService(NewOcrService(state)),
			application.NewService(NewLlmService(state)),
			application.NewService(NewAsrService(state)),
		},
		Assets: application.AssetOptions{
			Handler: application.AssetFileServerFS(assets),
		},
		Mac: application.MacOptions{
			// Keep running in the background when the window is closed so the
			// tray and global shortcuts keep working.
			ApplicationShouldTerminateAfterLastWindowClosed: false,
		},
	})

	state.SetApp(app)

	// Initialise ONNX Runtime synchronously so every service is ready when the
	// UI loads. If the shared library is missing the user is prompted to
	// download it in the Runtime tab.
	if err := ort.Init(cfg.OnnxLibPath); err != nil {
		log.Printf("ONNX Runtime not ready (%v); use the Runtime tab to download it", err)
	}

	mainWindow := app.Window.NewWithOptions(application.WebviewWindowOptions{
		Title:  "CPM OCR Studio",
		Width:  1280,
		Height: 820,
		Mac: application.MacWindow{
			InvisibleTitleBarHeight: 50,
			Backdrop:                application.MacBackdropTranslucent,
			TitleBar:                application.MacTitleBarHiddenInset,
		},
		BackgroundColour: application.NewRGB(10, 12, 22),
		URL:              "/",
	})
	state.SetMainWindow(mainWindow)

	// System tray + menu + global shortcuts (background OCR).
	setupTrayAndShortcuts(state)

	if err := app.Run(); err != nil {
		log.Fatal(err)
	}
}