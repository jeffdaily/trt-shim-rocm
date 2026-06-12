// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// CUDA profiler API compat: trtexec's inference loop calls cudaProfilerStart/
// Stop around the measured region. On AMD these are no-ops (profiling is done
// out of band with rocprof); the calls just need to compile and succeed.
#ifndef TRT_SHIM_ROCM_COMPAT_CUDA_PROFILER_API_H
#define TRT_SHIM_ROCM_COMPAT_CUDA_PROFILER_API_H
#include "cuda_runtime_api.h"
inline cudaError_t cudaProfilerStart() { return cudaSuccess; }
inline cudaError_t cudaProfilerStop() { return cudaSuccess; }
#endif  // TRT_SHIM_ROCM_COMPAT_CUDA_PROFILER_API_H
