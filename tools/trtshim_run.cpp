// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// Flexible TensorRT-API driver for validating the shim: builds an engine from
// ONNX (optionally fp16), optionally serializes it to disk and/or loads it back
// from disk, runs inference, and checks the argmax against a golden. Used to
// exercise the fp16 path and the cross-process serialize round-trip on real
// models.
//
// Usage: trtshim_run <model.onnx> <input.bin> <golden.txt>
//                [--fp16] [--save <engine>] [--load <engine>]
// With --load, the model argument is ignored for building; the engine is read
// from disk. Exit 0 iff argmax matches the golden.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace {

class Logger : public nvinfer1::ILogger {
    void log(Severity s, char const* msg) noexcept override {
        if (s <= Severity::kWARNING) std::fprintf(stderr, "[trt] %s\n", msg);
    }
};

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto n = static_cast<std::streamsize>(f.tellg());
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

// Minimal entropy calibrator that feeds a fixed input batch a few times. Enough
// to exercise the int8 calibration -> migraphx::quantize_int8 bridge.
class FixedCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    FixedCalibrator(const std::string& raw, int batches)
        : batches_(batches), bytes_(raw.size()) {
        hipMalloc(&device_, bytes_);
        hipMemcpy(device_, raw.data(), bytes_, hipMemcpyHostToDevice);
    }
    ~FixedCalibrator() override { hipFree(device_); }
    int32_t getBatchSize() const noexcept override { return 1; }
    bool getBatch(void* bindings[], char const*[], int32_t n) noexcept override {
        if (batches_-- <= 0) return false;
        for (int32_t i = 0; i < n; ++i) bindings[i] = device_;
        return true;
    }
    void const* readCalibrationCache(std::size_t& length) noexcept override {
        length = 0;
        return nullptr;
    }
    void writeCalibrationCache(void const*, std::size_t) noexcept override {}

private:
    int batches_;
    size_t bytes_;
    void* device_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <model.onnx> <input.bin> <golden.txt> "
                     "[--fp16] [--save <e>] [--load <e>]\n",
                     argv[0]);
        return 2;
    }
    const std::string model = argv[1], input = argv[2], golden = argv[3];
    bool fp16 = false, int8 = false;
    std::string save_path, load_path;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--fp16") fp16 = true;
        else if (a == "--int8") int8 = true;
        else if (a == "--save" && i + 1 < argc) save_path = argv[++i];
        else if (a == "--load" && i + 1 < argc) load_path = argv[++i];
    }

    Logger logger;
    const std::string input_raw = read_file(input);
    auto runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = nullptr;
    std::unique_ptr<FixedCalibrator> calibrator;

    if (!load_path.empty()) {
        const std::string blob = read_file(load_path);
        engine = runtime->deserializeCudaEngine(blob.data(), blob.size());
    } else {
        auto builder = nvinfer1::createInferBuilder(logger);
        auto network = builder->createNetworkV2(0);
        auto config = builder->createBuilderConfig();
        if (fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
        if (int8) {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            calibrator = std::make_unique<FixedCalibrator>(input_raw, 4);
            config->setInt8Calibrator(calibrator.get());
        }
        auto parser = nvonnxparser::createParser(*network, logger);
        if (!parser->parseFromFile(model.c_str(), 0)) {
            std::fprintf(stderr, "parse failed\n");
            return 1;
        }
        auto plan = builder->buildSerializedNetwork(*network, *config);
        if (!plan) {
            std::fprintf(stderr, "build failed\n");
            return 1;
        }
        if (!save_path.empty()) {
            std::ofstream(save_path, std::ios::binary)
                .write(static_cast<const char*>(plan->data()), plan->size());
            std::printf("saved engine (%zu bytes) to %s\n", plan->size(),
                        save_path.c_str());
        }
        engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    }
    if (!engine) {
        std::fprintf(stderr, "engine null\n");
        return 1;
    }

    auto context = engine->createExecutionContext();

    const int32_t nio = engine->getNbIOTensors();
    std::vector<void*> device(nio, nullptr);
    int out_idx = -1;
    int64_t out_elems = 0;
    for (int32_t i = 0; i < nio; ++i) {
        char const* name = engine->getIOTensorName(i);
        auto dims = engine->getTensorShape(name);
        const int64_t elems = volume(dims);
        const int32_t bpe = engine->getTensorBytesPerComponent(name);
        hipMalloc(&device[i], static_cast<size_t>(elems) * bpe);
        context->setTensorAddress(name, device[i]);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            hipMemcpy(device[i], input_raw.data(), input_raw.size(),
                      hipMemcpyHostToDevice);
        } else {
            out_idx = i;
            out_elems = elems;
        }
    }

    if (!context->executeV2(device.data())) {
        std::fprintf(stderr, "executeV2 failed\n");
        return 1;
    }

    std::vector<float> out(static_cast<size_t>(out_elems));
    hipMemcpy(out.data(), device[out_idx], out.size() * sizeof(float),
              hipMemcpyDeviceToHost);
    for (void* p : device) hipFree(p);

    const int argmax =
        static_cast<int>(std::max_element(out.begin(), out.end()) - out.begin());
    int want = -1;
    std::ifstream(golden) >> want;
    std::printf("argmax=%d golden=%d%s%s%s\n", argmax, want,
                fp16 ? " [fp16]" : "", int8 ? " [int8]" : "",
                load_path.empty() ? "" : " [loaded]");
    if (argmax != want) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
