# Architecture

How the shim is built, so you can extend it. Read the [README](../README.md)
first for the high-level picture. Everything here is validated on gfx90a /
ROCm 7.2.1 / MIGraphX 2.15.0.

## The big picture

Three layers:

1. **The nvinfer1 / nvonnxparser implementation** (`src/shim.cpp`) -- concrete
   C++ classes implementing the TensorRT interfaces, plus the `extern "C"`
   factory functions an app pulls from `libnvinfer`.
2. **The backend bridge** (`src/backend.{h,cpp}`) -- the only code that knows
   about MIGraphX and HIP. Turns ONNX bytes into a compiled `migraphx::program`,
   runs it, and owns the serialized-engine format. `backend.h` deliberately uses
   no TensorRT types, so the two layers stay decoupled.
3. **The CUDA->HIP compat shim** (`include/compat/`) -- forwarding headers so the
   vendored NVIDIA headers and the stock sample helpers compile and run on AMD.

## How an app reaches the shim: the apiv pattern

Modern TensorRT (8.x+) public classes do not have virtual methods you override.
Instead each `nvinfer1::IXxx` has non-virtual methods that forward to a protected
`apiv::VXxx* mImpl`:

```cpp
class ICudaEngine : public INoCopy {
public:
  Dims getTensorShape(char const* n) const noexcept { return mImpl->getTensorShape(n); }
  // ... all public methods forward to mImpl ...
protected:
  apiv::VCudaEngine* mImpl;   // the actual pure-virtual interface (NvInferImpl.h)
};
```

So the shim derives from **both** the public class and the `apiv::VXxx`
pure-virtual class, and points `mImpl` at itself:

```cpp
class ShimEngine : public nvinfer1::ICudaEngine, public nvinfer1::apiv::VCudaEngine {
public:
  ShimEngine(...) { mImpl = this; }                 // legal: mImpl is protected
  Dims getTensorShape(char const* n) const noexcept override { ... }  // implements VCudaEngine
  ...
};
```

`INoCopy` has a protected default constructor, so constructing it from the
derived class is fine. The factory returns `new ShimEngine` as an
`ICudaEngine*`; the app's `engine->getTensorShape(n)` forwards through `mImpl`
to the shim's override.

Notes that will bite you:
- Only `VRuntime`, `VCudaEngine`, `VExecutionContext` declare `getPImpl()`. Do
  **not** add a `getPImpl` override to a class whose V-base lacks it.
- The shim classes live in `namespace nvinfer1` (in `src/shim.cpp`, inside
  `namespace nvinfer1 { namespace shim { ... } }`) so the generated stub
  signatures -- which use unqualified `Dims`, `ITensor`, ... because the
  V-classes are declared in `nvinfer1` -- resolve.
- `nvonnxparser::IParser` is a plain pure-virtual interface (no apiv); implement
  it directly. `SubGraphCollection_t` is in the global namespace.

## The ONNX-capture trick

Implementing `INetworkDefinition`'s 57+ `addConvolution`/`addPooling`/... methods
would be enormous. We don't: real apps only use the network as a write-only
token between `nvonnxparser::parseFromFile` and `buildSerializedNetwork`. So:

- `ShimParser::parseFromFile/parse` reads the ONNX bytes and stores them in the
  bound `ShimNetwork` (and runs `introspect()` to populate input/output
  `ITensor`s so `network->getInput(0)->getDimensions()` works).
- `ShimNetwork`'s `addXxx` methods are all stubs.
- `buildSerializedNetwork` pulls the captured bytes and calls `backend::build`.

If a consumer builds a network via the layer API instead of ONNX, that path is
unimplemented (a Phase-4 item).

## Generated stubs

Each `apiv::VXxx` has dozens of pure virtuals; only ~30 are on the critical
path. `tools/gen_stubs.py` reads `NvInferImpl.h` and, for each V-class, emits
`src/generated/VXxx.stubs.inc` with trivial override definitions for every pure
virtual **not** in that class's hand-written exclude set. The `.inc` is
`#include`d inside the shim class body. Build wiring is a custom command in
`CMakeLists.txt` (target `gen_stubs`); the generated dir is gitignored.

Stub bodies: pointer->`nullptr`, bool->`false`, void->nothing, value types
(enums, `Dims`)->`{}`, reference returns->`trtshim_die(name)` (aborts; reaching
one means an unsupported path). `getPImpl`->`this`. Non-reference stubs log via
`trtshim_unimpl(name)` when `TRTSHIM_DEBUG` is set, which is how you discover
that an app needs a method you stubbed.

**To implement a stubbed method:** add its name to that class's set in
`gen_stubs.py` EXCLUDE, and hand-write the override in `src/shim.cpp`.

## The backend bridge

`backend.h` exposes a tiny TensorRT-free API:

- `introspect(onnx, n) -> IOInfo` -- parse + compile statically, return
  input/output names+dims+dtypes (for the network token).
- `build(onnx, n, opts, calib, dyn) -> std::string` -- parse, optionally
  quantize (fp16 / int8 calibration), compile, and pack a serialized blob.
- `load(blob, n, err) -> unique_ptr<Engine>` -- unpack a blob into a runnable
  engine; rejects anything without our magic header.
- `Engine::run(device_ptrs, concrete_shapes, stream)` -- bind IO and eval.

Key MIGraphX facts the bridge relies on (see `docs/migraphx-findings.md` for
the gaps):
- `parse_onnx_buffer(data, size, onnx_options)`; `onnx_options` pins free dims
  (`set_default_dim_value(1)`) or makes them dynamic (`set_default_dyn_dim_value`).
- The run path is **hybrid**, chosen at build:
  - **Static engines compile with `offload_copy=false`.** This exposes program
    outputs as bindable parameters named `main:#output_<N>`, so `Engine::run`
    binds the caller's device buffers for inputs AND outputs -- MIGraphX writes
    results straight into the caller's output buffer (**zero-copy**) -- and runs
    via `run_async` on the caller's stream. That run is **HIP-graph capturable**,
    so `enqueueV3` under CUDA-graph capture works (trtexec `--useCudaGraph`).
  - **Dynamic engines compile with `offload_copy=true`** and use a host-staging
    fallback (stage inputs device->host, `eval`, copy outputs host->device).
    MIGraphX's dynamic `select_module` dispatch crashes under `offload_copy=false`
    (a MIGraphX bug), so dynamic engines are not graph-capturable for now.
  `EngineImpl` picks the path by whether the program exposes `main:#output_<N>`
  parameters.
- `get_parameter_shapes()` gives input names+shapes; **outputs are positional**
  (no names). We recover ONNX output names with a ~60-line protobuf walk
  (`onnx_output_names` in `backend.cpp`) so the engine reports the real names
  (e.g. `Plus214_Output_0`) that consumers look up by.

### Serialized engine format

MIGraphX only serializes to a file, so the blob is: 8-byte magic `TRTSHIM\x01`,
then per-IO metadata (dir, dtype, bytes/elem, name, dims), then the MIGraphX
program bytes (round-tripped through a temp file). `load` validates the magic
and rebuilds IO metadata + program. A non-shim blob (e.g. an NVIDIA `.engine`)
is rejected with a clear error.

## int8 calibration bridge

`backend.h` declares an abstract `CalibrationSource`; `shim.cpp`'s
`ShimCalibSource` implements it over `nvinfer1::IInt8Calibrator`: it calls
`getBatch` (the app fills device pointers), copies each batch device->host, and
yields it. `build` feeds batches into `migraphx::quantize_int8_options::
add_calibration_data` and runs `quantize_int8` on the uncompiled program.
Lifetime gotcha: the host batches and `program_parameters` must stay alive until
`quantize_int8` returns (we keep them in vectors). The calibration *cache*
(persisting scales) is not implemented -- MIGraphX exposes no scale accessor.

## Dynamic shapes

`ShimOptimizationProfile` records per-input min/opt/max `Dims`; `dynamicAxis()`
finds the single axis where min != max. `build` parses that input as dynamic
(`set_default_dyn_dim_value`) and marks the dynamic axis -1 in the IO metadata.
At runtime `setInputShape` stores the concrete dims, which `executeV2`/
`enqueueV3` thread into `Engine::run`; run builds the migraphx argument at the
concrete shape instead of the static one. Single-axis only. Concat-heavy models
hit a MIGraphX dynamic codegen bug (see findings).

## The compat layer

`include/compat/cuda_runtime_api.h` (plus `cuda.h`, `cuda_runtime.h`,
`cuda_fp16.h`) is placed on the include path so `<cuda_runtime_api.h>` resolves
there on AMD. It aliases types (`cudaStream_t`=`hipStream_t`, ...) and forwards
the runtime calls the samples execute (malloc/memcpy/streams/events/device
queries) to HIP; CUDA-graph capture symbols, used only by helper code the simple
samples never call, are stubbed with CUDA-shaped signatures. On an NVIDIA build
this directory is absent and the real CUDA headers are used, so nothing here
perturbs CUDA. This is a shadow-header shim: the compat directory is placed
ahead of the system include path so `<cuda_runtime_api.h>` resolves here on an
AMD build, and is simply absent from the include path on an NVIDIA build.
Building `trtexec` grew this surface considerably -- device properties
(`cudaDeviceProp` aliased to `hipDeviceProp_t`), `cudaMemGetInfo`, managed/host
alloc, `cudaLaunchHostFunc`, pointer attributes, stream-capture status, and the
profiler stubs -- all in `include/compat/`.

## trtexec

The stock NVIDIA `trtexec` runs unmodified against the shim. Two things made it
work beyond the compat surface: (1) `trtexec` dlopens `libnvinfer.so.10` /
`libnvonnxparser.so.10` at runtime and `dlsym`s the factory symbols, so the
build creates those versioned aliases (and a no-op `libnvinfer_plugin.so.10`
that `trtexec` loads for the standard plugins); (2) it exercises more of the
engine/context API than the sample -- `getNbOptimizationProfiles` (return 1),
`setOptimizationProfileAsync`, `IExecutionContext::getEngine`/`getTensorShape`/
`getTensorStrides`, the `*V2` tensor-format queries, and the
`IStreamReader`/`IStreamReaderV2` deserialize overloads (drain the stream into
the blob deserializer). `TRTSHIM_DEBUG=1` was how each next-needed method was
found: run, see which stub it logs or which `trtshim_die` it hits, implement,
repeat.

## Testing

- `ctest` (8 cases) is the committed gate: MIGraphX smoke, the shim call path,
  the stock sampleOnnxMNIST, fp16, int8, serialize round-trip, dynamic shapes.
- `tools/sweep_models.sh` and `tools/onnx_backend_scoreboard.py` are the broader
  (network-dependent, uncommitted-model) validations.
- Set `TRTSHIM_DEBUG=1` to log every stubbed method an app actually hits -- the
  fastest way to find what to implement next for a new consumer.
