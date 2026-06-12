// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// CUDA-runtime-to-HIP compatibility shim. The vendored NVIDIA TensorRT headers
// and the stock TensorRT sample helpers (samples/common) include
// <cuda_runtime_api.h> and use a bounded slice of the CUDA runtime API. On AMD
// we put this directory on the include path so that include resolves here, and
// we provide the same API shape backed by HIP:
//   - types and the calls the samples actually execute (malloc/free/memcpy,
//     streams, events, device queries) forward directly to HIP;
//   - CUDA graph capture symbols, which appear only in helper code the simple
//     samples never call, are stubbed with CUDA-shaped signatures so they
//     compile without depending on HIP graph-API signature details.
//
// This is the MOAT "Strategy A" pattern: on an NVIDIA build this directory is
// absent and the genuine CUDA header is used, so nothing here perturbs CUDA.
#ifndef TRT_SHIM_ROCM_COMPAT_CUDA_RUNTIME_API_H
#define TRT_SHIM_ROCM_COMPAT_CUDA_RUNTIME_API_H

#include <cstddef>

#include <hip/hip_runtime_api.h>

// ---- types ----------------------------------------------------------------
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaError_t = hipError_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaDeviceAttr = hipDeviceAttribute_t;
using cudaLimit = hipLimit_t;
using cudaStreamCaptureMode = hipStreamCaptureMode;

// CUDA graph types are only referenced by helper code that the simple samples
// never instantiate; keep them as opaque handles independent of HIP graphs.
using cudaGraph_t = struct trtshimCudaGraph*;
using cudaGraphExec_t = struct trtshimCudaGraphExec*;

// ---- enum / constant values ----------------------------------------------
constexpr cudaError_t cudaSuccess = hipSuccess;
constexpr cudaError_t cudaErrorStreamCaptureInvalidated =
    hipErrorStreamCaptureInvalidated;
constexpr cudaMemcpyKind cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
constexpr unsigned int cudaStreamNonBlocking = hipStreamNonBlocking;
constexpr cudaStreamCaptureMode cudaStreamCaptureModeGlobal =
    hipStreamCaptureModeGlobal;
constexpr cudaDeviceAttr cudaDevAttrComputeCapabilityMajor =
    hipDeviceAttributeComputeCapabilityMajor;
constexpr cudaDeviceAttr cudaDevAttrComputeCapabilityMinor =
    hipDeviceAttributeComputeCapabilityMinor;
constexpr cudaDeviceAttr cudaDevAttrMaxPersistingL2CacheSize =
    hipDeviceAttributePersistingL2CacheMaxSize;
constexpr cudaLimit cudaLimitStackSize = hipLimitStackSize;

// ---- runtime calls that forward directly to HIP ---------------------------
inline cudaError_t cudaMalloc(void** p, size_t n) { return hipMalloc(p, n); }
inline cudaError_t cudaFree(void* p) { return hipFree(p); }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n,
                              cudaMemcpyKind k) {
    return hipMemcpy(dst, src, n, k);
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n,
                                   cudaMemcpyKind k, cudaStream_t s = 0) {
    return hipMemcpyAsync(dst, src, n, k, s);
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned f) {
    return hipStreamCreateWithFlags(s, f);
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    return hipStreamDestroy(s);
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    return hipStreamSynchronize(s);
}
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { return hipEventCreate(e); }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) {
    return hipEventDestroy(e);
}
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = 0) {
    return hipEventRecord(e, s);
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    return hipEventSynchronize(e);
}
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a,
                                        cudaEvent_t b) {
    return hipEventElapsedTime(ms, a, b);
}
inline cudaError_t cudaGetDevice(int* d) { return hipGetDevice(d); }
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr a, int d) {
    return hipDeviceGetAttribute(v, a, d);
}
inline cudaError_t cudaDeviceGetLimit(size_t* v, cudaLimit l) {
    return hipDeviceGetLimit(v, l);
}
inline const char* cudaGetErrorString(cudaError_t e) {
    return hipGetErrorString(e);
}
inline const char* cudaGetErrorName(cudaError_t e) {
    return hipGetErrorName(e);
}
inline cudaError_t cudaGetLastError() { return hipGetLastError(); }

// ---- CUDA graph capture: stubbed (compile-only for the simple samples) -----
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) {
    return cudaSuccess;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*) {
    return cudaSuccess;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, void*,
                                        void*, size_t) {
    return cudaSuccess;
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) {
    return cudaSuccess;
}
inline cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }

#endif  // TRT_SHIM_ROCM_COMPAT_CUDA_RUNTIME_API_H
