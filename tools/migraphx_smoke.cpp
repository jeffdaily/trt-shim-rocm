// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// Phase 0 smoke test for trt-shim-rocm: prove the MIGraphX C++ toolchain runs
// an ONNX model end-to-end on the AMD GPU and agrees with the ONNX reference.
// This uses MIGraphX directly (no TensorRT shim yet) to de-risk the backend
// primitives the shim will build on: parse_onnx_buffer, compile(gpu),
// program_parameters, eval. Host IO via offload_copy keeps it minimal; the
// shim's enqueueV3 path will instead bind caller device pointers
// (offload_copy=false) for zero-copy, matching TensorRT setTensorAddress.
//
// Usage: migraphx_smoke <model.onnx> <input.bin> <golden.txt>
// Exit 0 and prints PASSED iff the GPU argmax matches the golden argmax.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <migraphx/migraphx.hpp>

namespace {

std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    const auto n = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::vector<char> buf(static_cast<size_t>(n));
    f.read(buf.data(), n);
    return buf;
}

int golden_argmax(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    int argmax = -1;
    f >> argmax;
    return argmax;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0]
                  << " <model.onnx> <input.bin> <golden.txt>\n";
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string input_path = argv[2];
    const std::string golden_path = argv[3];

    try {
        const auto model_bytes = read_file(model_path);
        auto prog = migraphx::parse_onnx_buffer(model_bytes.data(),
                                                model_bytes.size());

        migraphx::target gpu("gpu");
        migraphx::compile_options opts;
        opts.set_offload_copy(true);  // host IO; MIGraphX moves to/from device
        prog.compile(gpu, opts);

        const auto input_raw = read_file(input_path);

        migraphx::program_parameters params;
        auto param_shapes = prog.get_parameter_shapes();
        for (const char* name : param_shapes.names()) {
            auto shp = param_shapes[name];
            if (std::string(name) == "input") {
                params.add(name, migraphx::argument(
                                     shp, const_cast<char*>(input_raw.data())));
            } else {
                // Defensive: bind any other live parameter to a zero buffer.
                static std::vector<std::vector<char>> scratch;
                scratch.emplace_back(shp.bytes(), 0);
                params.add(name,
                           migraphx::argument(shp, scratch.back().data()));
            }
        }

        auto outputs = prog.eval(params);
        auto logits = outputs[0].as_vector<float>();

        const auto it = std::max_element(logits.begin(), logits.end());
        const int gpu_argmax = static_cast<int>(it - logits.begin());
        const int want = golden_argmax(golden_path);

        std::cout << "gpu_argmax=" << gpu_argmax << " golden_argmax=" << want
                  << " logits=[";
        for (size_t i = 0; i < logits.size(); ++i) {
            std::cout << (i ? " " : "") << logits[i];
        }
        std::cout << "]\n";

        if (gpu_argmax != want) {
            std::cout << "FAILED: argmax mismatch\n";
            return 1;
        }
        std::cout << "PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
