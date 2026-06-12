# trt-shim-rocm

A TensorRT-API-compatible shim for AMD ROCm, backed by [MIGraphX](https://github.com/ROCm/AMDMIGraphX).

The goal is to let applications written against the NVIDIA TensorRT C++ API
(`nvinfer1::`) build and run inference on AMD GPUs with only a recompile and a
relink. TensorRT's own core builder and runtime (`libnvinfer`) are
closed-source, but the API headers, the ONNX parser interface, the samples, and
`trtexec` are open (Apache-2.0). This project re-implements those interfaces
over MIGraphX and ships `libtrtshim.so`, installed under the `libnvinfer.so` /
`libnvonnxparser.so` names so an unmodified TensorRT consumer links against it.

## Status

Early bring-up. Phase 0 (backend de-risk) is complete: MIGraphX parses, compiles,
and runs an ONNX model on a real AMD GPU (gfx90a, ROCm 7.2.1), matching the ONNX
reference output. The TensorRT API surface itself is the next phase.

The build is validated only on AMD GPUs where MIGraphX is available: Linux gfx90a
(primary) and gfx1100. Windows is not yet supported because MIGraphX has no
Windows build ([TheRock #1912](https://github.com/ROCm/TheRock/issues/1912)).

## Design at a glance

Real TensorRT apps treat `INetworkDefinition` as a write-only token between the
ONNX parser and `buildSerializedNetwork`. The shim exploits this: its ONNX
parser captures the raw model bytes and hands them to `migraphx::parse_onnx_buffer`,
so the engine is built from the ONNX graph directly and the 57+ layer-builder
methods are not needed for the common path.

```
createInferBuilder -> createNetworkV2 -> nvonnxparser::parseFromFile
  -> buildSerializedNetwork   (migraphx::parse_onnx_buffer -> compile(gpu))
  -> createInferRuntime -> deserializeCudaEngine
  -> createExecutionContext -> setTensorAddress -> enqueueV3   (run_async)
```

### Scope

Supported (target): ONNX models, fp16/int8, single-axis dynamic shapes (dynamic
batch), engine serialization. Validated with stock TensorRT samples, `trtexec`,
and the ONNX operator-conformance backend test.

Not supported: NVIDIA-built `.engine` blobs, CUDA plugin binaries, and DLA are
hard non-goals. Custom plugins, control-flow ops, and multi-axis dynamic shapes
are out of the committed scope (gaps in MIGraphX that are better fixed upstream);
the shim reports them with a clear error.

## Build

Requires ROCm with MIGraphX installed (`apt install migraphx migraphx-dev`).

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
This work was authored with the assistance of an AI coding assistant.
