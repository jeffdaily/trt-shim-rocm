// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// MIGraphX backend bridge. This is the only part of the shim that knows about
// MIGraphX and HIP; the nvinfer1 shim classes talk to it through plain C++
// types. It owns the ONNX -> compile -> run path and the engine serialization
// format (a magic header plus IO metadata plus the MIGraphX program).
#ifndef TRT_SHIM_ROCM_BACKEND_H
#define TRT_SHIM_ROCM_BACKEND_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace trtshim {

enum class IODir : uint8_t { kInput = 0, kOutput = 1 };

// A model input or output. datatype is the MIGraphX shape datatype enum value;
// the shim translates it to nvinfer1::DataType.
struct IOTensor {
    std::string name;
    std::vector<int64_t> dims;
    int datatype = 0;
    size_t bytes_per_elem = 0;
    IODir dir = IODir::kInput;
};

// Builder flags the shim forwards to the backend. Mirrors the subset of
// nvinfer1::BuilderFlag we act on; kept independent so backend.* needs no TRT
// headers.
struct BuildOptions {
    bool fp16 = false;
    bool int8 = false;
    bool bf16 = false;
};

// One int8 calibration batch: host data per input, sized to that input's byte
// size. Supplied by the shim's IInt8Calibrator wrapper.
struct CalibrationData {
    std::map<std::string, std::vector<char>> tensors;
};

// Abstract calibration data source, implemented by the shim over
// nvinfer1::IInt8Calibrator. Keeps backend.* free of TensorRT types.
class CalibrationSource {
public:
    virtual ~CalibrationSource() = default;
    // Fill `batch` with the next calibration batch (host data per input).
    // Return false when the calibrator is exhausted.
    virtual bool next(const std::vector<IOTensor>& inputs,
                      CalibrationData& batch) = 0;
};

// A compiled engine: the MIGraphX program plus its IO description.
class Engine {
public:
    virtual ~Engine() = default;
    virtual const std::vector<IOTensor>& ios() const = 0;

    // Run synchronously (stream == nullptr) or async on the given hipStream_t.
    // device_ptrs maps every IO tensor name to a caller-owned device buffer.
    // Outputs are copied into the caller's output buffers. Returns false on
    // error.
    virtual bool run(const std::map<std::string, void*>& device_ptrs,
                     void* stream) = 0;
};

// Parse an ONNX model and report its inputs/outputs. Used by the network token
// so the sample can query input/output dimensions after parsing. Output names
// are recovered from the ONNX bytes (MIGraphX does not expose them).
struct IOInfo {
    std::vector<IOTensor> inputs;
    std::vector<IOTensor> outputs;
};
IOInfo introspect(const void* onnx, size_t n);

// Build: parse ONNX, apply flags, compile for the GPU, and return a serialized
// engine blob (magic header + IO metadata + MIGraphX program). When opts.int8
// is set and calib is non-null, calibration batches drive migraphx int8
// quantization. Throws std::runtime_error on failure.
std::string build(const void* onnx, size_t n, const BuildOptions& opts,
                  CalibrationSource* calib = nullptr);

// Load a serialized engine blob produced by build(). Returns nullptr and sets
// err if the blob is not ours (e.g. an NVIDIA .engine).
std::unique_ptr<Engine> load(const void* blob, size_t n, std::string& err);

}  // namespace trtshim

#endif  // TRT_SHIM_ROCM_BACKEND_H
