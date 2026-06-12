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
// This is a shadow-header shim: this directory is put ahead of the system
// include path so `<cuda_runtime_api.h>` resolves here on an AMD build. On an
// NVIDIA build this directory is simply not on the include path, so the genuine
// CUDA header is used and nothing here perturbs CUDA.
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

// ---- device / host / event surface (used by trtexec's framework) ----------
using cudaDeviceProp = hipDeviceProp_t;
using cudaUUID_t = hipUUID;
using cudaHostFn_t = hipHostFn_t;
constexpr unsigned int cudaEventBlockingSync = hipEventBlockingSync;
constexpr unsigned int cudaEventDefault = hipEventDefault;
constexpr unsigned int cudaDeviceScheduleSpin = hipDeviceScheduleSpin;
constexpr cudaStreamCaptureMode cudaStreamCaptureModeThreadLocal =
    hipStreamCaptureModeThreadLocal;

inline cudaError_t cudaDriverGetVersion(int* v) { return hipDriverGetVersion(v); }
inline cudaError_t cudaRuntimeGetVersion(int* v) { return hipRuntimeGetVersion(v); }
inline cudaError_t cudaGetDeviceCount(int* c) { return hipGetDeviceCount(c); }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int d) {
    return hipGetDeviceProperties(p, d);
}
inline cudaError_t cudaSetDevice(int d) { return hipSetDevice(d); }
inline cudaError_t cudaSetDeviceFlags(unsigned f) { return hipSetDeviceFlags(f); }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned f) {
    return hipEventCreateWithFlags(e, f);
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned f) {
    return hipStreamWaitEvent(s, e, f);
}
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { return hipStreamCreate(s); }
inline cudaError_t cudaMallocManaged(void** p, size_t n,
                                     unsigned f = hipMemAttachGlobal) {
    return hipMallocManaged(p, n, f);
}
inline cudaError_t cudaMallocHost(void** p, size_t n) { return hipHostMalloc(p, n, 0); }
inline cudaError_t cudaFreeHost(void* p) { return hipHostFree(p); }
inline cudaError_t cudaLaunchHostFunc(cudaStream_t s, cudaHostFn_t fn, void* u) {
    return hipLaunchHostFunc(s, fn, u);
}
inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    return hipMemGetInfo(free, total);
}

// stream-capture status query and pointer attributes
using cudaStreamCaptureStatus = hipStreamCaptureStatus;
constexpr cudaStreamCaptureStatus cudaStreamCaptureStatusNone =
    hipStreamCaptureStatusNone;
inline cudaError_t cudaStreamIsCapturing(cudaStream_t s,
                                         cudaStreamCaptureStatus* st) {
    return hipStreamIsCapturing(s, st);
}
using cudaPointerAttributes = hipPointerAttribute_t;
using cudaMemoryType = hipMemoryType;
constexpr cudaMemoryType cudaMemoryTypeHost = hipMemoryTypeHost;
constexpr cudaMemoryType cudaMemoryTypeDevice = hipMemoryTypeDevice;
constexpr cudaMemoryType cudaMemoryTypeUnregistered = hipMemoryTypeUnregistered;
inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* a,
                                            const void* p) {
    return hipPointerGetAttributes(a, p);
}

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
