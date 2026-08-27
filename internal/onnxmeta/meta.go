package onnxmeta

import (
	"fmt"
	"os"
)

// ModelIO describes the input/output names and shapes of an ONNX model.
type ModelIO struct {
	Inputs  []string
	Outputs []string
	// InputShapes holds the tensor shape for each input, with -1 marking
	// symbolic (dynamic) dimensions.
	InputShapes [][]int64
}

// InputShape returns the shape of the input with the given name, or nil.
func (m *ModelIO) InputShape(name string) []int64 {
	for i, n := range m.Inputs {
		if n == name {
			return m.InputShapes[i]
		}
	}
	return nil
}

// Parse reads an ONNX model file and extracts the graph input/output names
// using a minimal protobuf wire-format parser. This avoids pulling in a full
// protobuf dependency just to introspect models.
func Parse(path string) (*ModelIO, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseBytes(data)
}

func parseBytes(data []byte) (*ModelIO, error) {
	// ModelProto: field 7 (graph) is a length-delimited GraphProto.
	var graph []byte
	i := 0
	for i < len(data) {
		field, wt, next, err := readTag(data, i)
		if err != nil {
			return nil, err
		}
		i = next
		switch wt {
		case 0: // varint
			_, i, err = readVarint(data, i)
		case 1: // 64-bit
			i += 8
		case 2: // length-delimited
			l, ni, err := readVarint(data, i)
			if err != nil {
				return nil, err
			}
			start := ni
			end := ni + int(l)
			if end > len(data) {
				return nil, fmt.Errorf("truncated message")
			}
			if field == 7 {
				graph = data[start:end]
			}
			i = end
		case 5: // 32-bit
			i += 4
		default:
			return nil, fmt.Errorf("unsupported wire type %d", wt)
		}
		if err != nil {
			return nil, err
		}
	}
	if graph == nil {
		return nil, fmt.Errorf("no graph found in ONNX model")
	}

	io := &ModelIO{}
	// GraphProto: field 11 = input, field 12 = output (repeated ValueInfoProto).
	j := 0
	for j < len(graph) {
		field, wt, next, err := readTag(graph, j)
		if err != nil {
			return nil, err
		}
		j = next
		switch wt {
		case 0:
			_, j, err = readVarint(graph, j)
		case 1:
			j += 8
		case 2:
			l, ni, err := readVarint(graph, j)
			if err != nil {
				return nil, err
			}
			start := ni
			end := ni + int(l)
			if end > len(graph) {
				return nil, fmt.Errorf("truncated graph")
			}
			switch field {
			case 11:
				name, shape, err := valueInfo(graph[start:end])
				if err != nil {
					return nil, err
				}
				io.Inputs = append(io.Inputs, name)
				io.InputShapes = append(io.InputShapes, shape)
			case 12:
				name, _, err := valueInfo(graph[start:end])
				if err != nil {
					return nil, err
				}
				io.Outputs = append(io.Outputs, name)
			}
			j = end
		case 5:
			j += 4
		default:
			return nil, fmt.Errorf("unsupported wire type %d", wt)
		}
		if err != nil {
			return nil, err
		}
	}
	if len(io.Inputs) == 0 || len(io.Outputs) == 0 {
		return nil, fmt.Errorf("model has no inputs/outputs")
	}
	return io, nil
}

// valueInfo extracts the name and tensor shape from a ValueInfoProto.
func valueInfo(data []byte) (string, []int64, error) {
	name := ""
	var shape []int64
	i := 0
	for i < len(data) {
		field, wt, next, err := readTag(data, i)
		if err != nil {
			return "", nil, err
		}
		i = next
		switch wt {
		case 0:
			_, i, err = readVarint(data, i)
		case 1:
			i += 8
		case 2:
			l, ni, err := readVarint(data, i)
			if err != nil {
				return "", nil, err
			}
			switch field {
			case 1: // name
				name = string(data[ni : ni+int(l)])
			case 2, 7: // type (TypeProto) — field number changed across ONNX versions
				shape, err = tensorShape(data[ni : ni+int(l)])
				if err != nil {
					return "", nil, err
				}
			}
			i = ni + int(l)
		case 5:
			i += 4
		default:
			return "", nil, fmt.Errorf("unsupported wire type %d", wt)
		}
		if err != nil {
			return "", nil, err
		}
	}
	return name, shape, nil
}

// tensorShape walks a TypeProto -> TensorTypeProto -> TensorShapeProto to
// extract concrete dimensions (-1 for symbolic dims).
func tensorShape(data []byte) ([]int64, error) {
	var shape []int64
	i := 0
	for i < len(data) {
		field, wt, next, err := readTag(data, i)
		if err != nil {
			return nil, err
		}
		i = next
		switch wt {
		case 0:
			_, i, err = readVarint(data, i)
		case 1:
			i += 8
		case 2:
			l, ni, err := readVarint(data, i)
			if err != nil {
				return nil, err
			}
			if field == 1 { // tensor_type
				shape, err = tensorShapeProto(data[ni : ni+int(l)])
				if err != nil {
					return nil, err
				}
			}
			i = ni + int(l)
		case 5:
			i += 4
		default:
			return nil, fmt.Errorf("unsupported wire type %d", wt)
		}
		if err != nil {
			return nil, err
		}
	}
	return shape, nil
}

func tensorShapeProto(data []byte) ([]int64, error) {
	var shape []int64
	i := 0
	for i < len(data) {
		field, wt, next, err := readTag(data, i)
		if err != nil {
			return nil, err
		}
		i = next
		switch wt {
		case 0:
			_, i, err = readVarint(data, i)
		case 1:
			i += 8
		case 2:
			l, ni, err := readVarint(data, i)
			if err != nil {
				return nil, err
			}
			if field == 2 { // shape
				dims, err := tensorShapeDims(data[ni : ni+int(l)])
				if err != nil {
					return nil, err
				}
				shape = dims
			}
			i = ni + int(l)
		case 5:
			i += 4
		default:
			return nil, fmt.Errorf("unsupported wire type %d", wt)
		}
		if err != nil {
			return nil, err
		}
	}
	return shape, nil
}

func tensorShapeDims(data []byte) ([]int64, error) {
	var dims []int64
	i := 0
	for i < len(data) {
		field, wt, next, err := readTag(data, i)
		if err != nil {
			return nil, err
		}
		i = next
		switch wt {
		case 0:
			_, i, err = readVarint(data, i)
		case 1:
			i += 8
		case 2:
			l, ni, err := readVarint(data, i)
			if err != nil {
				return nil, err
			}
			if field == 1 { // dim (Dimension message)
				dims = append(dims, parseDimension(data[ni:ni+int(l)]))
			}
			i = ni + int(l)
		case 5:
			i += 4
		default:
			return nil, fmt.Errorf("unsupported wire type %d", wt)
		}
		if err != nil {
			return nil, err
		}
	}
	return dims, nil
}

// parseDimension extracts a concrete dim_value, or -1 for a symbolic dim.
func parseDimension(data []byte) int64 {
	i := 0
	for i < len(data) {
		field, wt, next, err := readTag(data, i)
		if err != nil {
			return -1
		}
		i = next
		switch wt {
		case 0:
			v, ni, err := readVarint(data, i)
			if err != nil {
				return -1
			}
			if field == 1 { // dim_value
				return int64(v)
			}
			i = ni
		case 1:
			i += 8
		case 2:
			l, ni, err := readVarint(data, i)
			if err != nil {
				return -1
			}
			if field == 2 { // dim_param (symbolic)
				return -1
			}
			i = ni + int(l)
		case 5:
			i += 4
		default:
			return -1
		}
	}
	return -1
}

func readTag(b []byte, i int) (field int, wt int, next int, err error) {
	if i >= len(b) {
		return 0, 0, i, fmt.Errorf("unexpected end of data")
	}
	v, ni, err := readVarint(b, i)
	if err != nil {
		return 0, 0, i, err
	}
	return int(v >> 3), int(v & 7), ni, nil
}

func readVarint(b []byte, i int) (uint64, int, error) {
	var result uint64
	var shift uint
	for {
		if i >= len(b) {
			return 0, i, fmt.Errorf("unexpected end of data")
		}
		c := b[i]
		i++
		result |= uint64(c&0x7f) << shift
		if c&0x80 == 0 {
			return result, i, nil
		}
		shift += 7
		if shift >= 64 {
			return 0, i, fmt.Errorf("varint overflow")
		}
	}
}