// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// Stub libnvinfer_plugin. TensorRT consumers (e.g. trtexec) dlopen
// libnvinfer_plugin.so.10 and call initLibNvInferPlugins to register the
// standard NVIDIA plugins. The shim does not provide plugins (see the plugin
// non-goal in the README), so this is a no-op that lets such consumers load and
// run for models that use no plugins. A model that actually requires a plugin
// fails later with a clear "operator not supported" error from MIGraphX.
extern "C" bool initLibNvInferPlugins(void* /*logger*/,
                                      char const* /*libNamespace*/) {
    return true;
}
