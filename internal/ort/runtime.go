package ort

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// LibFileName returns the shared library file name of ONNX Runtime for the
// current platform.
func LibFileName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return "libonnxruntime.dylib"
	default:
		return "libonnxruntime.so"
	}
}

var (
	mu          sync.Mutex
	initialized bool
	libPath     string
)

// Init loads the ONNX Runtime shared library and initializes the global
// environment. It is safe to call multiple times.
func Init(lib string) error {
	mu.Lock()
	defer mu.Unlock()
	if initialized && libPath == lib {
		return nil
	}
	if _, err := os.Stat(lib); err != nil {
		return fmt.Errorf("ONNX Runtime library not found at %s: %w", lib, err)
	}
	// If a different library path was previously initialized we need a fresh
	// environment; onnxruntime only allows one shared-library load per process.
	if initialized {
		ort.DestroyEnvironment()
		initialized = false
	}
	ort.SetSharedLibraryPath(lib)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}
	initialized = true
	libPath = lib
	return nil
}

// IsInitialized reports whether the runtime has been initialized.
func IsInitialized() bool {
	mu.Lock()
	defer mu.Unlock()
	return initialized
}

// Version returns the ONNX Runtime version string if available.
func Version() string {
	ort.InitializeEnvironment()
	return ort.GetVersion()
}

// Session wraps a DynamicAdvancedSession so that input/output tensor shapes may
// change between calls. This is what the OCR and LLM engines use.
type Session struct {
	ds        *ort.DynamicAdvancedSession
	inputs    []string
	outputs   []string
	destroyed bool
}

// NewSession creates a dynamic session from an ONNX file. When enableCoreML is
// true (and the platform is macOS), the CoreML execution provider is
// registered so supported operators run on the Apple GPU / Neural Engine.
func NewSession(modelPath string, inputs, outputs []string, threads int, enableCoreML ...bool) (*Session, error) {
	if !IsInitialized() {
		return nil, fmt.Errorf("ONNX Runtime is not initialized")
	}
	useCoreML := false
	if len(enableCoreML) > 0 {
		useCoreML = enableCoreML[0]
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	opts.SetIntraOpNumThreads(threads)
	if useCoreML && runtime.GOOS == "darwin" {
		// MLProgram format gives the best Apple Silicon / ANE coverage.
		if err := opts.AppendExecutionProviderCoreMLV2(map[string]string{"ModelFormat": "MLProgram"}); err != nil {
			return nil, fmt.Errorf("register CoreML EP: %w", err)
		}
	}
	ds, err := ort.NewDynamicAdvancedSession(modelPath, inputs, outputs, opts)
	if err != nil {
		return nil, err
	}
	return &Session{ds: ds, inputs: inputs, outputs: outputs}, nil
}

// Inputs returns the model input names.
func (s *Session) Inputs() []string { return s.inputs }

// Outputs returns the model output names.
func (s *Session) Outputs() []string { return s.outputs }

// Run executes the session with the given input/output tensors.
func (s *Session) Run(inputs, outputs []ort.Value) error {
	if s.destroyed {
		return fmt.Errorf("session has been destroyed")
	}
	return s.ds.Run(inputs, outputs)
}

// Destroy releases the session.
func (s *Session) Destroy() error {
	if s.destroyed {
		return nil
	}
	s.destroyed = true
	return s.ds.Destroy()
}

// Tensor is a type alias for the underlying tensor type.
type Tensor[T ort.TensorData] = ort.Tensor[T]

// NewTensor creates a tensor backed by data.
func NewTensor[T ort.TensorData](shape []int64, data []T) (*ort.Tensor[T], error) {
	return ort.NewTensor(ort.NewShape(shape...), data)
}

// NewEmptyTensor creates a tensor with the given shape and zeroed data.
func NewEmptyTensor[T ort.TensorData](shape []int64) (*ort.Tensor[T], error) {
	return ort.NewEmptyTensor[T](ort.NewShape(shape...))
}

// Value is an ONNX value that can be used as a session input/output.
type Value = ort.Value