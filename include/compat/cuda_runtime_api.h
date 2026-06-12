// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// CUDA-to-HIP compatibility shim. The vendored NVIDIA TensorRT headers do
// `#include <cuda_runtime_api.h>` and use a handful of CUDA runtime TYPES in
// their signatures (cudaStream_t, cudaEvent_t). On AMD we put this directory on
// the include path so that include resolves here instead of to a real CUDA
// toolkit, and we alias those types onto their HIP equivalents. The HIP runtime
// types are usable from host translation units, so the shim compiles without a
// device compiler.
//
// This is the MOAT "Strategy A" compat-header pattern: on an NVIDIA build this
// directory is absent and the genuine CUDA header is used instead, so nothing
// here perturbs CUDA consumers.
#ifndef TRT_SHIM_ROCM_COMPAT_CUDA_RUNTIME_API_H
#define TRT_SHIM_ROCM_COMPAT_CUDA_RUNTIME_API_H

#include <hip/hip_runtime_api.h>

using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaError_t = hipError_t;

// A minimal slice of the CUDA error enum, sufficient for headers and simple
// host glue. Extend as consumers require.
constexpr cudaError_t cudaSuccess = hipSuccess;

#endif  // TRT_SHIM_ROCM_COMPAT_CUDA_RUNTIME_API_H
