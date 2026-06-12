// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// Validates single-axis dynamic shapes (IOptimizationProfile): builds one
// dynamic-batch engine, then runs it at two different batch sizes on the same
// execution context, checking each row's argmax against the golden. Exercises
// createOptimizationProfile/setDimensions/addOptimizationProfile and
// setInputShape.
//
// Usage: dyn_run <dyn_model.onnx> <input2.bin> <golden.txt(=a0 a1)>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace {
class Logger : public nvinfer1::ILogger {
    void log(Severity s, char const* m) noexcept override {
        if (s <= Severity::kWARNING) std::fprintf(stderr, "[trt] %s\n", m);
    }
};
std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    auto n = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::string b(static_cast<size_t>(n), '\0');
    f.read(b.data(), n);
    return b;
}
nvinfer1::Dims dims4(int b) {
    nvinfer1::Dims d{};
    d.nbDims = 4;
    d.d[0] = b;
    d.d[1] = 1;
    d.d[2] = 28;
    d.d[3] = 28;
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model.onnx> <input2.bin> <golden>\n",
                     argv[0]);
        return 2;
    }
    Logger logger;
    auto builder = nvinfer1::createInferBuilder(logger);
    auto network = builder->createNetworkV2(0);
    auto config = builder->createBuilderConfig();
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, dims4(1));
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, dims4(2));
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, dims4(4));
    config->addOptimizationProfile(profile);

    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(argv[1], 0)) { std::fprintf(stderr, "parse\n"); return 1; }
    auto plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) { std::fprintf(stderr, "build\n"); return 1; }
    auto runtime = nvinfer1::createInferRuntime(logger);
    auto engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    if (!engine) { std::fprintf(stderr, "deserialize\n"); return 1; }
    auto context = engine->createExecutionContext();

    const std::string in_raw = read_file(argv[2]);  // batch-2 input
    std::vector<int> golden(2);
    { std::ifstream g(argv[3]); g >> golden[0] >> golden[1]; }

    const int32_t nio = engine->getNbIOTensors();
    bool all_ok = true;
    for (int B : {1, 2}) {
        context->setInputShape("input", dims4(B));
        std::vector<void*> dev(nio, nullptr);
        int out_idx = -1;
        for (int32_t i = 0; i < nio; ++i) {
            char const* name = engine->getIOTensorName(i);
            bool is_in = engine->getTensorIOMode(name) ==
                         nvinfer1::TensorIOMode::kINPUT;
            size_t elems = is_in ? size_t(B) * 1 * 28 * 28 : size_t(B) * 10;
            hipMalloc(&dev[i], elems * sizeof(float));
            context->setTensorAddress(name, dev[i]);
            if (is_in) {
                hipMemcpy(dev[i], in_raw.data(), elems * sizeof(float),
                          hipMemcpyHostToDevice);
            } else {
                out_idx = i;
            }
        }
        if (!context->executeV2(dev.data())) { std::fprintf(stderr, "exec\n"); return 1; }
        std::vector<float> out(size_t(B) * 10);
        hipMemcpy(out.data(), dev[out_idx], out.size() * sizeof(float),
                  hipMemcpyDeviceToHost);
        for (void* p : dev) hipFree(p);
        for (int r = 0; r < B; ++r) {
            int am = static_cast<int>(
                std::max_element(out.begin() + r * 10, out.begin() + r * 10 + 10) -
                (out.begin() + r * 10));
            bool ok = am == golden[r];
            all_ok &= ok;
            std::printf("batch=%d row=%d argmax=%d golden=%d %s\n", B, r, am,
                        golden[r], ok ? "ok" : "MISMATCH");
        }
    }
    std::printf(all_ok ? "PASSED\n" : "FAILED\n");
    return all_ok ? 0 : 1;
}
