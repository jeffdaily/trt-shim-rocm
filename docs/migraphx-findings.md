# MIGraphX gaps found building trt-shim-rocm

Consolidated feedback for the MIGraphX team, collected while building the
TensorRT-API shim (jeffdaily/trt-shim-rocm). Each item has a concrete reproducer
or a precise description. Verified on ROCm 7.2.1 / MIGraphX 2.15.0 / gfx90a
unless noted. Not yet filed upstream -- for Jeff to forward.

## 1. Dynamic-shape compile fails on ResNet-50 (codegen, real bug)

Parsing `resnet50-v2-7.onnx` with a dynamic batch dimension
(`onnx_options::set_default_dyn_dim_value(dynamic_dimension(1, 4))`) and
compiling for the gpu target fails inside MIGraphX's JIT kernel codegen. A fused
concat+pointwise kernel is generated with the wrong arity:

```
.../input/main.cpp:53:115: note: candidate function template not viable:
  requires at least 7 arguments, but 4 were provided
   transform_args(make_tensors(), rotate_last(), vectorize<1,3>())
     (private_p0,private_p1,private_p2,private_p3)([](auto y,
      auto pointwise00_concat_x0, ... auto... xs) { ...
1 error generated when compiling for gfx90a.
```

The same model compiles fine with a fixed batch
(`set_default_dim_value(1)`). A trivial CNN with a dynamic batch dim compiles and
runs correctly, so this is specific to the concat-fusion path under dynamic
shapes, not dynamic shapes in general.

Reproducer: `dynamic_concat_repro.cpp` in this directory (build against
`-lmigraphx_c`, run with `resnet50-v2-7.onnx`).

Impact on the shim: single-axis dynamic shapes work and are validated on a
synthetic dynamic-batch model, but dynamic batch on ResNet-50 is blocked by this
codegen bug. Severity: medium (real models with concat + dynamic batch).

## 2. No ONNX output-name accessor (API gap)

`migraphx::program::get_output_shapes()` is positional; there is no way to
recover the ONNX graph output names from a parsed/compiled program. TensorRT
consumers expect engine IO tensors to carry the real ONNX names. The shim works
around this by parsing `GraphProto.output[].name` out of the ONNX bytes itself
(a ~60-line protobuf walk). A `get_output_names()` (parallel to
`program_parameter_shapes::names()` for inputs) would remove that workaround.
Severity: low (workaround exists), but it is a clean API addition.

## 3. No buffer-based program serialize/load (API gap)

The C API exposes only file-based `migraphx_save`/`migraphx_load`. Serializing a
compiled program to or from an in-memory buffer requires a temp-file round-trip.
A `save_buffer`/`load_buffer` pair would let the shim implement
`IHostMemory`-based engine serialization without touching the filesystem.
Severity: low.

## Notes on items that turned out NOT to be problems

- int8 GPU calibration works. `migraphx::quantize_int8` on the gpu target
  quantizes correctly; the shim's IInt8Calibrator bridge validated on a small
  CNN and ResNet-50 with argmax preserved. (An earlier draft of the shim plan
  cited issue #3585 as an int8 reliability bug; #3585 is closed as not-a-bug.)
