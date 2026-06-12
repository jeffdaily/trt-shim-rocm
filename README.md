# trt-shim-rocm

A TensorRT-API-compatible inference shim for AMD ROCm, backed by
[MIGraphX](https://github.com/ROCm/AMDMIGraphX).

## What this is and why

NVIDIA TensorRT is a closed-source inference library: its core builder and
runtime ship as a binary (`libnvinfer.so`). But the *interface* is open and
Apache-2.0 -- the `nvinfer1::` C++ headers, the ONNX parser interface
(`nvonnxparser`), the samples, and `trtexec`. This project re-implements those
interfaces on top of AMD MIGraphX and ships the result as `libtrtshim.so`,
installed under the `libnvinfer.so` / `libnvonnxparser.so` names.

The payoff: an application written against the TensorRT C++ API can build and
run inference on an AMD GPU with **only a recompile against the (vendored)
headers and a relink** -- no source changes. The headline proof is that the
stock, unmodified NVIDIA `sampleOnnxMNIST` builds against this shim and prints
`&&&& PASSED TensorRT.sample_onnx_mnist` on a Radeon/Instinct GPU.

This is useful for porting CUDA/TensorRT inference apps to ROCm without
rewriting their inference layer onto a different API (MIGraphX, ONNX Runtime).

## Status

Validated on **Linux gfx90a, ROCm 7.2.1, MIGraphX 2.15.0** (gfx1100 is the
intended second target). Windows is unsupported: MIGraphX has no Windows build
yet ([TheRock #1912](https://github.com/ROCm/TheRock/issues/1912)).

What works today, validated on real GPU:

| Capability | How it's validated |
|---|---|
| ONNX model -> build engine -> infer | stock `sampleOnnxMNIST`; `trtshim_run` |
| Link-level drop-in (`-lnvinfer`) | `sampleOnnxMNIST` links the alias, `ldd` resolves to libtrtshim |
| Stock `trtexec` (dlopen drop-in) | unmodified NVIDIA `trtexec --onnx/--fp16/--int8/--save+loadEngine` runs and reports metrics on AMD |
| CUDA graphs | `trtexec --useCudaGraph` really captures a HIP graph (zero-copy run path); ~4.6x throughput on the MNIST model |
| fp16 | `trtshim_run --fp16` on ResNet-50 (matches onnxruntime) |
| int8 (calibration) | `trtshim_run --int8`; IInt8Calibrator -> migraphx::quantize_int8 |
| Engine serialize/deserialize (cross-process) | `trtshim_run --save/--load`; rejects NVIDIA `.engine` |
| Single-axis dynamic shapes (dynamic batch) | `trtshim_dyn_run`: one engine at batch 1 and 2 |
| Operator breadth | 96 operators fully green on the ONNX backend test (see `test/onnx_scoreboard.md`); 4 CNN families (ResNet/SqueezeNet/MobileNet/GoogLeNet) match onnxruntime |

Deferred / not done (see [Roadmap](#roadmap)): the int8 calibration *cache*,
custom plugins, control-flow ops, and multi-axis dynamic shapes. Hard non-goals:
deserializing NVIDIA-built `.engine` blobs, running CUDA plugin binaries, DLA.

## Quick start

Requires ROCm with MIGraphX (`apt install migraphx migraphx-dev`) and a Python
env with `onnx`, `onnxruntime`, `numpy` for regenerating test assets.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j
ctest --test-dir build --output-on-failure        # 8/8 on a working setup
```

`test/run_gpu_tests.sh` does the same end to end (regenerates the small model,
builds, runs ctest).

## Using the shim in your own app

1. Compile your TensorRT app against the vendored headers in
   `third_party/tensorrt-headers/include` and `include/compat` (the latter
   resolves `<cuda_runtime_api.h>` to HIP).
2. Link against `build/libnvinfer.so` and `build/libnvonnxparser.so` (symlinks
   to `libtrtshim.so`).
3. Your CUDA buffer-management calls (`cudaMalloc`/`cudaMemcpy`/streams) resolve
   to HIP via `include/compat`; pass device pointers as usual.

`vendor/tensorrt-samples/sampleOnnxMNIST` is a worked example built exactly this
way (see its target in `CMakeLists.txt`).

## Tools

**`trtexec`** -- the stock NVIDIA TensorRT CLI, vendored unmodified and built
against the shim. It dlopens `libnvinfer.so.10` at runtime (the build installs
that alias), so it is a true drop-in. Run it like the real thing; it needs the
build dir on the library path:

```
LD_LIBRARY_PATH=build ./build/trtexec --onnx=test/mnist/mnist.onnx --fp16
```

Validated with `--onnx`, `--fp16`, `--int8`, `--saveEngine`, and `--loadEngine`;
it prints normal throughput/latency metrics. (Build with `-DBUILD_TRTEXEC=OFF`
to skip it.)

Bespoke drivers (the `trtshim_` prefix marks them as this project's, not NVIDIA
tools). All take ONNX models and run them through the shim on the GPU.

- **`trtshim_run <model> <input.bin> <golden.txt> [--fp16|--int8|--save f|--load f]`**
  -- build/serialize/run a classifier; checks argmax against a golden. The
  general-purpose validation driver.
- **`trtshim_dyn_run <model> <input2.bin> <golden>`** -- builds one dynamic-batch
  engine and runs it at batch 1 and 2 (dynamic-shape demo).
- **`trtshim_infer <model> <outdir> <name=infile>...`** -- generic runner: bind
  named raw inputs, write named raw outputs. Used by the conformance scoreboard.
- **`driver_smoke` / `migraphx_smoke`** -- minimal end-to-end and MIGraphX-only
  smoke tests.

Scripts:

- **`tools/sweep_models.sh`** -- downloads ResNet-50/SqueezeNet/MobileNet/
  GoogLeNet and checks each against onnxruntime through the shim.
- **`tools/onnx_backend_scoreboard.py build/trtshim_infer`** -- runs the
  standardized ONNX `backend.test` node cases through the shim and writes the
  per-operator scoreboard to `test/onnx_scoreboard.md`.
- **`tools/make_*.py`** -- regenerate the small committed test models/goldens.

## How it works (short version)

The key trick: real TensorRT apps treat `INetworkDefinition` as a write-only
token between the ONNX parser and `buildSerializedNetwork`. The shim's parser
captures the raw ONNX bytes and hands them to `migraphx::parse_onnx_buffer`, so
the engine is built from the ONNX graph directly -- the 57+ layer-builder
methods are never needed.

```
createInferBuilder -> createNetworkV2 -> nvonnxparser::parseFromFile (capture bytes)
  -> buildSerializedNetwork   (parse_onnx_buffer -> [quantize] -> compile(gpu))
  -> createInferRuntime -> deserializeCudaEngine
  -> createExecutionContext -> setTensorAddress -> executeV2/enqueueV3 (eval)
```

The public `nvinfer1::IXxx` classes forward to a protected `apiv::VXxx* mImpl`;
the shim derives from both and sets `mImpl = this`. The ~280 interface methods
not on the critical path are trivial stubs **generated** from the vendored
headers by `tools/gen_stubs.py`. See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
for the full internals (and how to extend the shim).

## Project layout

```
third_party/tensorrt-headers/  vendored NVIDIA TRT 10.7 + onnx-tensorrt headers
include/compat/                CUDA-runtime -> HIP shadow headers
src/
  shim.cpp                     nvinfer1/nvonnxparser implementations + factories
  backend.{h,cpp}              the only MIGraphX/HIP code; ONNX->compile->run
  generated/                   apiv stub overrides (built by gen_stubs.py)
tools/                         drivers, scoreboard, model/golden generators
vendor/tensorrt-samples/       stock NVIDIA sample + common, vendored unmodified
test/                          committed small models, goldens, scoreboard, harness
docs/                          ARCHITECTURE.md, migraphx-findings.md, reproducers
```

## Extending the shim

- **A method an app calls returns a default / aborts?** It's a generated stub.
  Add its name to the class's exclude set in `tools/gen_stubs.py` and hand-write
  it in `src/shim.cpp`. (Reference-returning stubs abort via `trtshim_die`;
  value stubs return defaults -- set `TRTSHIM_DEBUG=1` to log which stub is hit.)
- **New backend behavior** (a quantization mode, an IO mapping) goes in
  `src/backend.cpp`, which is the only file that touches MIGraphX.
- The architecture doc walks through adding both.

## Roadmap

Deferred work, in rough priority order (background and reproducers in
[docs/migraphx-findings.md](docs/migraphx-findings.md)):

- **int8 calibration cache** -- the int8 *math* works; persisting/reusing the
  calibration scales (`read/writeCalibrationCache`) is blocked on MIGraphX
  exposing the computed scales (a feature request filed in the findings).
- **Dynamic batch on concat-heavy models** -- blocked on a MIGraphX dynamic-shape
  codegen bug (reproducer in `docs/`).
- **Plugins (IPluginV3)** -- MIGraphX has a custom-op path (`migraphx::module` +
  a registered kernel), so it is tractable but unbuilt; the shim ships a no-op
  plugin lib so plugin-free consumers run.
- **Layer-builder API** (`INetworkDefinition::addConvolution`, ...) -- only
  needed by apps that build networks in code instead of from ONNX. Each layer
  maps to a `migraphx::operation`, so it is tractable but large; left until a
  consumer needs it.
- **Control-flow ops, multi-axis dynamic shapes** -- MIGraphX gaps better closed
  upstream than worked around.

## Known MIGraphX gaps found here

See [docs/migraphx-findings.md](docs/migraphx-findings.md): a dynamic-shape
codegen bug, no ONNX output-name accessor, no buffer-based program serialize,
and no int8-scale accessor. With reproducers, for the MIGraphX team.

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY.md](THIRD_PARTY.md) (provenance of vendored NVIDIA headers and
samples). This work was authored with the assistance of an AI coding assistant.
