// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// CUDA driver-API header compat: some TensorRT sample helpers include <cuda.h>
// defensively. They do not use the driver API itself, so on AMD this forwards
// to the HIP runtime types via the runtime compat shim.
#ifndef TRT_SHIM_ROCM_COMPAT_CUDA_H
#define TRT_SHIM_ROCM_COMPAT_CUDA_H
#include "cuda_runtime_api.h"
#endif  // TRT_SHIM_ROCM_COMPAT_CUDA_H
