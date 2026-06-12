# Third-party provenance

This project is Apache-2.0. It will vendor third-party source, kept under its
original license and copyright headers (unmodified). This file records what is
vendored and why.

## Planned vendored components (not yet imported)

- **NVIDIA TensorRT API headers** (`third_party/tensorrt-headers/`)
  - Source: https://github.com/NVIDIA/TensorRT (`include/`)
  - License: Apache-2.0 (the open-source TensorRT headers)
  - Pin: TensorRT 10.7 GA (to be confirmed at import)
  - Files: `NvInfer.h`, `NvInferRuntime.h`, `NvInferRuntimeBase.h`,
    `NvInferRuntimePlugin.h`, `NvInferLegacyDims.h`, `NvInferImpl.h`.
  - Rationale: the shim must be source- and ABI-compatible with what consumer
    apps include; the headers are vendored verbatim with their NVIDIA Apache-2.0
    banners intact. No AMD copyright line is added to these files.

- **ONNX-TensorRT parser header** (`third_party/tensorrt-headers/`)
  - Source: https://github.com/onnx/onnx-tensorrt (`NvOnnxParser.h`)
  - License: Apache-2.0

- **TensorRT samples and trtexec** (`vendor/tensorrt-samples/`, `vendor/trtexec/`)
  - Source: https://github.com/NVIDIA/TensorRT (`samples/`, `tools/trtexec/`)
  - License: Apache-2.0
  - Rationale: validation is driven by TensorRT's own samples and the `trtexec`
    tool, built against the shim, so correctness is judged by NVIDIA's own code
    rather than a bespoke harness.

## Test assets

- `test/models/mnist_cnn.onnx` is generated locally by
  `tools/make_model_and_golden.py` (deterministic, seeded); it is not a
  third-party model. Its golden output is computed by the ONNX
  `ReferenceEvaluator` (the spec reference backend, shipped with the `onnx`
  package, Apache-2.0).
