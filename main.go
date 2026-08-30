package main

import (
	"embed"
	"log"

	"github.com/wailsapp/wails/v3/pkg/application"

	"cpm_orc/internal/app"
	"cpm_orc/internal/config"
	"cpm_orc/internal/ort"
	"cpm_orc/internal/paddleocr"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	cfg, err := config.Load(config.DefaultPath())
	if err != nil {
		log.Printf("warning: failed to load config: %v", err)
	}
	if cfg == nil {
		cfg = config.Default()
	}
	_ = cfg.EnsureDirs()

	state := app.New(cfg)
	// Apply the configured proxy to downloads from the start.
	ort.SetProxy(cfg.Proxy)
	// Enable CoreML execution provider for OCR on Apple Silicon.
	paddleocr.EnableCoreML = cfg.CoreML

	wailsApp := application.New(application.Options{
		Name:        "CPM OCR Studio",
		Description: "HuggingFace model manager, PaddleOCR and MiniCPM ONNX inference",
		Services: []application.Service{
			application.NewService(app.NewRuntimeService(state)),
			application.NewService(app.NewHFHubService(state)),
			application.NewService(app.NewOcrService(state)),
			application.NewService(app.NewLlmService(state)),
			application.NewService(app.NewAsrService(state)),
			application.NewService(app.NewMathService(state)),
			application.NewService(app.NewOllamaService(state)),
			application.NewService(app.NewYoloService(state)),
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

	state.SetApp(wailsApp)

	// Initialise ONNX Runtime synchronously so every service is ready when the
	// UI loads. If the shared library is missing the user is prompted to
	// download it in the Runtime tab.
	if err := ort.Init(cfg.OnnxLibPath); err != nil {
		log.Printf("ONNX Runtime not ready (%v); use the Runtime tab to download it", err)
	}

	mainWindow := wailsApp.Window.NewWithOptions(application.WebviewWindowOptions{
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
	app.SetupTrayAndShortcuts(state)

	if err := wailsApp.Run(); err != nil {
		log.Fatal(err)
	}
}
