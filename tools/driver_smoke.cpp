// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// Milestone 1a driver: exercise the full TensorRT API call path through the
// shim (linked as libnvinfer / libnvonnxparser) on a real AMD GPU, using our
// own ONNX model so IO names are under our control. This is the same sequence a
// stock TensorRT app uses: build a serialized engine from ONNX, deserialize it,
// bind device buffers, run, and check the result. Proves the shim mechanism end
// to end before the stock sampleOnnxMNIST (Milestone 1b) adds ONNX-name and
// data fidelity.
//
// Usage: driver_smoke <model.onnx> <input.bin> <golden.txt>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace {

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, char const* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[trt] %s\n", msg);
    }
};

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto n = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::string buf(static_cast<size_t>(n), '\0');
    f.read(buf.data(), n);
    return buf;
}

int64_t volume(const nvinfer1::Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model.onnx> <input.bin> <golden.txt>\n",
                     argv[0]);
        return 2;
    }
    Logger logger;

    // ---- build a serialized engine from ONNX ----
    auto builder = nvinfer1::createInferBuilder(logger);
    auto network = builder->createNetworkV2(0);
    auto config = builder->createBuilderConfig();
    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(argv[1], 0)) {
        std::fprintf(stderr, "parse failed\n");
        return 1;
    }
    auto plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        std::fprintf(stderr, "build failed\n");
        return 1;
    }

    // ---- deserialize and run ----
    auto runtime = nvinfer1::createInferRuntime(logger);
    auto engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    if (!engine) {
        std::fprintf(stderr, "deserialize failed\n");
        return 1;
    }
    auto context = engine->createExecutionContext();

    const std::string input_raw = read_file(argv[2]);

    const int32_t nio = engine->getNbIOTensors();
    std::vector<void*> device(nio, nullptr);
    std::vector<std::string> names(nio);
    int output_index = -1;
    int64_t output_elems = 0;

    for (int32_t i = 0; i < nio; ++i) {
        char const* name = engine->getIOTensorName(i);
        names[i] = name;
        auto dims = engine->getTensorShape(name);
        const int64_t elems = volume(dims);
        const int32_t bpe = engine->getTensorBytesPerComponent(name);
        hipMalloc(&device[i], static_cast<size_t>(elems) * bpe);
        context->setTensorAddress(name, device[i]);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            hipMemcpy(device[i], input_raw.data(), input_raw.size(),
                      hipMemcpyHostToDevice);
        } else {
            output_index = i;
            output_elems = elems;
        }
    }

    if (!context->executeV2(device.data())) {
        std::fprintf(stderr, "executeV2 failed\n");
        return 1;
    }

    std::vector<float> out(static_cast<size_t>(output_elems));
    hipMemcpy(out.data(), device[output_index], out.size() * sizeof(float),
              hipMemcpyDeviceToHost);
    for (void* p : device) hipFree(p);

    const int gpu_argmax =
        static_cast<int>(std::max_element(out.begin(), out.end()) - out.begin());
    int want = -1;
    std::ifstream(argv[3]) >> want;

    std::printf("gpu_argmax=%d golden_argmax=%d logits=[", gpu_argmax, want);
    for (size_t i = 0; i < out.size(); ++i) std::printf("%s%g", i ? " " : "", out[i]);
    std::printf("]\n");

    if (gpu_argmax != want) {
        std::printf("FAILED: argmax mismatch\n");
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
