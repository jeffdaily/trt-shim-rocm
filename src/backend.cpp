// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
#include "backend.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.hpp>

namespace trtshim {
namespace {

constexpr char kMagic[8] = {'T', 'R', 'T', 'S', 'H', 'I', 'M', '\x01'};

size_t bytes_per_elem(const migraphx::shape& s) {
    const size_t n = s.elements();
    return n ? s.bytes() / n : 0;
}

// Serialize/deserialize a compiled program by round-tripping a temp file; the
// MIGraphX C API exposes only file-based save/load (no buffer variant yet).
std::string program_to_bytes(const migraphx::program& prog) {
    char path[] = "/tmp/trtshim_XXXXXX";
    int fd = ::mkstemp(path);
    if (fd < 0) throw std::runtime_error("trtshim: mkstemp failed");
    ::close(fd);
    migraphx::save(prog, path);
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const auto sz = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::string out(static_cast<size_t>(sz), '\0');
    f.read(out.data(), sz);
    ::unlink(path);
    return out;
}

migraphx::program program_from_bytes(const std::string& bytes) {
    char path[] = "/tmp/trtshim_XXXXXX";
    int fd = ::mkstemp(path);
    if (fd < 0) throw std::runtime_error("trtshim: mkstemp failed");
    ::close(fd);
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    f.close();
    auto prog = migraphx::load(path);
    ::unlink(path);
    return prog;
}

std::vector<IOTensor> collect_inputs(const migraphx::program& prog) {
    std::vector<IOTensor> ins;
    auto shapes = prog.get_parameter_shapes();
    for (const char* name : shapes.names()) {
        auto s = shapes[name];
        auto lengths = s.lengths();  // single call: begin()/end() must match
        IOTensor t;
        t.name = name;
        t.dims.assign(lengths.begin(), lengths.end());
        t.datatype = s.type();
        t.bytes_per_elem = bytes_per_elem(s);
        t.dir = IODir::kInput;
        ins.push_back(std::move(t));
    }
    return ins;
}

std::vector<IOTensor> collect_outputs(const migraphx::program& prog,
                                      const std::vector<std::string>& names) {
    std::vector<IOTensor> outs;
    auto shapes = prog.get_output_shapes();
    for (size_t i = 0; i < shapes.size(); ++i) {
        auto s = shapes[i];
        IOTensor t;
        t.name = (i < names.size() && !names[i].empty())
                     ? names[i]
                     : (shapes.size() == 1 ? std::string("output")
                                           : "output_" + std::to_string(i));
        auto lengths = s.lengths();  // single call: begin()/end() must match
        t.dims.assign(lengths.begin(), lengths.end());
        t.datatype = s.type();
        t.bytes_per_elem = bytes_per_elem(s);
        t.dir = IODir::kOutput;
        outs.push_back(std::move(t));
    }
    return outs;
}

void apply_quantization(migraphx::program& prog, const BuildOptions& opts) {
    if (opts.fp16) migraphx::quantize_fp16(prog);
    // int8 (calibration) and bf16 are wired in a later phase.
}

class EngineImpl : public Engine {
public:
    EngineImpl(migraphx::program prog, std::vector<IOTensor> ios)
        : prog_(std::move(prog)), ios_(std::move(ios)) {}

    const std::vector<IOTensor>& ios() const override { return ios_; }

    bool run(const std::map<std::string, void*>& device_ptrs,
             void* /*stream*/) override {
        // The program is compiled with offload_copy=true, so MIGraphX owns the
        // device scratch and eval takes/returns HOST buffers. TensorRT hands us
        // DEVICE pointers, so we stage each input device->host before eval and
        // copy each output host->device after. (A true zero-copy path using
        // offload_copy=false and caller device pointers is a later
        // optimization.)
        migraphx::program_parameters params;
        auto pshapes = prog_.get_parameter_shapes();
        auto names = pshapes.names();
        std::vector<std::vector<char>> staging;
        staging.reserve(names.size());  // keep argument pointers stable
        for (const char* name : names) {
            auto s = pshapes[name];
            auto it = device_ptrs.find(name);
            if (it == device_ptrs.end()) return false;  // unbound input
            staging.emplace_back(s.bytes());
            if (hipMemcpy(staging.back().data(), it->second, s.bytes(),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                return false;
            }
            params.add(name, migraphx::argument(s, staging.back().data()));
        }

        auto results = prog_.eval(params);

        size_t oi = 0;
        for (const auto& io : ios_) {
            if (io.dir != IODir::kOutput) continue;
            if (oi >= results.size()) return false;
            auto it = device_ptrs.find(io.name);
            if (it == device_ptrs.end()) return false;
            auto arg = results[oi++];
            const size_t nbytes = arg.get_shape().bytes();
            if (hipMemcpy(it->second, arg.data(), nbytes,
                          hipMemcpyHostToDevice) != hipSuccess) {
                return false;
            }
        }
        return true;
    }

private:
    migraphx::program prog_;
    std::vector<IOTensor> ios_;
};

std::string pack_blob(const std::vector<IOTensor>& ios, const std::string& mxr) {
    std::string b(kMagic, sizeof(kMagic));
    auto put_u32 = [&](uint32_t v) { b.append(reinterpret_cast<char*>(&v), 4); };
    auto put_u64 = [&](uint64_t v) { b.append(reinterpret_cast<char*>(&v), 8); };
    put_u32(static_cast<uint32_t>(ios.size()));
    for (const auto& io : ios) {
        b.push_back(static_cast<char>(io.dir));
        b.push_back(static_cast<char>(io.datatype));
        put_u32(static_cast<uint32_t>(io.bytes_per_elem));
        put_u32(static_cast<uint32_t>(io.name.size()));
        b.append(io.name);
        put_u32(static_cast<uint32_t>(io.dims.size()));
        for (int64_t d : io.dims) put_u64(static_cast<uint64_t>(d));
    }
    put_u64(mxr.size());
    b.append(mxr);
    return b;
}

}  // namespace

IOInfo introspect(const void* onnx, size_t n,
                  const std::vector<std::string>& output_names) {
    // Parameter and output shapes are only reliable after compilation, so
    // compile for the GPU target here as well. The network token needs this to
    // answer getInput/getOutput dimensions before buildSerializedNetwork runs.
    auto prog = migraphx::parse_onnx_buffer(onnx, n);
    migraphx::target gpu("gpu");
    migraphx::compile_options copts;
    copts.set_offload_copy(true);
    prog.compile(gpu, copts);
    IOInfo info;
    info.inputs = collect_inputs(prog);
    info.outputs = collect_outputs(prog, output_names);
    return info;
}

std::string build(const void* onnx, size_t n, const BuildOptions& opts,
                  const std::vector<std::string>& output_names) {
    auto prog = migraphx::parse_onnx_buffer(onnx, n);
    apply_quantization(prog, opts);

    auto outs = collect_outputs(prog, output_names);  // names before compile

    migraphx::target gpu("gpu");
    migraphx::compile_options copts;
    copts.set_offload_copy(true);
    prog.compile(gpu, copts);

    std::vector<IOTensor> ios = collect_inputs(prog);
    ios.insert(ios.end(), outs.begin(), outs.end());

    return pack_blob(ios, program_to_bytes(prog));
}

std::unique_ptr<Engine> load(const void* blob, size_t n, std::string& err) {
    const char* p = static_cast<const char*>(blob);
    if (n < sizeof(kMagic) || std::memcmp(p, kMagic, sizeof(kMagic)) != 0) {
        err = "not a trt-shim-rocm engine (NVIDIA .engine blobs are not "
              "supported)";
        return nullptr;
    }
    size_t off = sizeof(kMagic);
    auto get_u32 = [&](uint32_t& v) {
        std::memcpy(&v, p + off, 4);
        off += 4;
    };
    auto get_u64 = [&](uint64_t& v) {
        std::memcpy(&v, p + off, 8);
        off += 8;
    };
    uint32_t n_io = 0;
    get_u32(n_io);
    std::vector<IOTensor> ios(n_io);
    for (auto& io : ios) {
        io.dir = static_cast<IODir>(p[off++]);
        io.datatype = static_cast<unsigned char>(p[off++]);
        uint32_t bpe = 0, namelen = 0, ndim = 0;
        get_u32(bpe);
        io.bytes_per_elem = bpe;
        get_u32(namelen);
        io.name.assign(p + off, namelen);
        off += namelen;
        get_u32(ndim);
        io.dims.resize(ndim);
        for (uint32_t i = 0; i < ndim; ++i) {
            uint64_t d = 0;
            get_u64(d);
            io.dims[i] = static_cast<int64_t>(d);
        }
    }
    uint64_t mxr_len = 0;
    get_u64(mxr_len);
    std::string mxr(p + off, mxr_len);

    try {
        auto prog = program_from_bytes(mxr);
        return std::make_unique<EngineImpl>(std::move(prog), std::move(ios));
    } catch (const std::exception& e) {
        err = e.what();
        return nullptr;
    }
}

}  // namespace trtshim
