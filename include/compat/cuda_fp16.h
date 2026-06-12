// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// CUDA half-precision compat: TensorRT sample helpers include <cuda_fp16.h>
// (mostly defensively). On AMD it resolves here and provides __half via HIP.
#ifndef TRT_SHIM_ROCM_COMPAT_CUDA_FP16_H
#define TRT_SHIM_ROCM_COMPAT_CUDA_FP16_H
#include <hip/hip_fp16.h>
#endif  // TRT_SHIM_ROCM_COMPAT_CUDA_FP16_H
