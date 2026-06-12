// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// Generic single-shot inference runner for the ONNX backend-test scoreboard.
// Builds an engine from an ONNX model through the shim, binds caller-provided
// raw input files by tensor name, runs, and writes each output as raw bytes.
// The Python harness compares the outputs against the ONNX expected results.
//
// Usage: trt_infer <model.onnx> <outdir> <name=infile> [<name=infile> ...]
// Prints one "OUT <name> <datatype> <d0,d1,...>" line per output.

#include <algorithm>
#include <cstdint>
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
        if (s <= Severity::kERROR) std::fprintf(stderr, "[trt] %s\n", m);
    }
};
std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("open " + p);
    auto n = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::string b(static_cast<size_t>(n), '\0');
    f.read(b.data(), n);
    return b;
}
int64_t volume(const nvinfer1::Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}
std::string sanitize(const std::string& s) {
    std::string o = s;
    for (char& c : o) if (c == '/' || c == ':' || c == ' ') c = '_';
    return o;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model> <outdir> [name=infile ...]\n",
                     argv[0]);
        return 2;
    }
    const std::string model = argv[1], outdir = argv[2];
    std::vector<std::pair<std::string, std::string>> ins;  // name -> file
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto eq = a.find('=');
        if (eq != std::string::npos) ins.emplace_back(a.substr(0, eq), a.substr(eq + 1));
    }

    Logger logger;
    try {
        auto builder = nvinfer1::createInferBuilder(logger);
        auto network = builder->createNetworkV2(0);
        auto config = builder->createBuilderConfig();
        auto parser = nvonnxparser::createParser(*network, logger);
        if (!parser->parseFromFile(model.c_str(), 0)) { std::fprintf(stderr, "parse\n"); return 1; }
        auto plan = builder->buildSerializedNetwork(*network, *config);
        if (!plan) { std::fprintf(stderr, "build\n"); return 1; }
        auto runtime = nvinfer1::createInferRuntime(logger);
        auto engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
        if (!engine) { std::fprintf(stderr, "deserialize\n"); return 1; }
        auto context = engine->createExecutionContext();

        const int32_t nio = engine->getNbIOTensors();
        std::vector<void*> dev(nio, nullptr);
        std::vector<std::pair<std::string, int>> outputs;  // name -> binding idx
        for (int32_t i = 0; i < nio; ++i) {
            char const* name = engine->getIOTensorName(i);
            auto dims = engine->getTensorShape(name);
            size_t bytes = static_cast<size_t>(volume(dims)) *
                           engine->getTensorBytesPerComponent(name);
            hipMalloc(&dev[i], bytes ? bytes : 1);
            context->setTensorAddress(name, dev[i]);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                std::string file;
                for (auto& kv : ins) if (kv.first == name) file = kv.second;
                if (file.empty() && ins.size() == 1) file = ins[0].second;
                auto raw = read_file(file);
                hipMemcpy(dev[i], raw.data(), std::min(raw.size(), bytes),
                          hipMemcpyHostToDevice);
            } else {
                outputs.emplace_back(name, i);
            }
        }
        if (!context->executeV2(dev.data())) { std::fprintf(stderr, "exec\n"); return 1; }

        for (auto& o : outputs) {
            char const* name = o.first.c_str();
            auto dims = engine->getTensorShape(name);
            size_t bytes = static_cast<size_t>(volume(dims)) *
                           engine->getTensorBytesPerComponent(name);
            std::string host(bytes, '\0');
            hipMemcpy(host.data(), dev[o.second], bytes, hipMemcpyDeviceToHost);
            std::ofstream(outdir + "/" + sanitize(o.first) + ".bin",
                          std::ios::binary)
                .write(host.data(), bytes);
            std::printf("OUT %s %d ", name,
                        static_cast<int>(engine->getTensorDataType(name)));
            for (int i = 0; i < dims.nbDims; ++i)
                std::printf("%s%ld", i ? "," : "", static_cast<long>(dims.d[i]));
            std::printf("\n");
        }
        for (void* p : dev) hipFree(p);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR %s\n", e.what());
        return 1;
    }
}
