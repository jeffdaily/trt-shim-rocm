// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// Some TensorRT sample helpers include <cuda_runtime.h>; on AMD it resolves
// here and forwards to the CUDA-runtime-to-HIP compatibility shim.
#ifndef TRT_SHIM_ROCM_COMPAT_CUDA_RUNTIME_H
#define TRT_SHIM_ROCM_COMPAT_CUDA_RUNTIME_H
#include "cuda_runtime_api.h"
#endif  // TRT_SHIM_ROCM_COMPAT_CUDA_RUNTIME_H
