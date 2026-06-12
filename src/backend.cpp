// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
#include "backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.hpp>

namespace trtshim {
namespace {

constexpr char kMagic[8] = {'T', 'R', 'T', 'S', 'H', 'I', 'M', '\x01'};

// Minimal protobuf reader to recover ONNX graph output names. MIGraphX does not
// expose them (get_output_shapes is positional), but TensorRT consumers expect
// the engine's IO tensor names to be the real ONNX names. We walk just enough
// of the wire format: ModelProto.graph (field 7) -> GraphProto.output (field
// 12, repeated ValueInfoProto) -> ValueInfoProto.name (field 1, string).
class PbReader {
public:
    PbReader(const uint8_t* p, size_t n) : p_(p), end_(p + n) {}
    bool eof() const { return p_ >= end_; }
    uint64_t varint() {
        uint64_t r = 0;
        int shift = 0;
        while (p_ < end_ && shift < 64) {
            uint8_t b = *p_++;
            r |= static_cast<uint64_t>(b & 0x7f) << shift;
            if (!(b & 0x80)) break;
            shift += 7;
        }
        return r;
    }
    bool tag(uint32_t& field, uint32_t& wire) {
        if (eof()) return false;
        uint64_t t = varint();
        field = static_cast<uint32_t>(t >> 3);
        wire = static_cast<uint32_t>(t & 7);
        return true;
    }
    std::pair<const uint8_t*, size_t> bytes() {
        uint64_t len = varint();
        if (p_ + len > end_) len = static_cast<uint64_t>(end_ - p_);
        const uint8_t* s = p_;
        p_ += len;
        return {s, static_cast<size_t>(len)};
    }
    void skip(uint32_t wire) {
        switch (wire) {
            case 0: varint(); break;
            case 1: p_ = std::min(p_ + 8, end_); break;
            case 2: bytes(); break;
            case 5: p_ = std::min(p_ + 4, end_); break;
            default: p_ = end_; break;
        }
    }

private:
    const uint8_t* p_;
    const uint8_t* end_;
};

std::vector<std::string> onnx_output_names(const void* data, size_t n) {
    std::vector<std::string> names;
    PbReader top(static_cast<const uint8_t*>(data), n);
    const uint8_t* gp = nullptr;
    size_t gn = 0;
    uint32_t f = 0, w = 0;
    while (top.tag(f, w)) {
        if (f == 7 && w == 2) {  // ModelProto.graph
            auto r = top.bytes();
            gp = r.first;
            gn = r.second;
            break;
        }
        top.skip(w);
    }
    if (!gp) return names;
    PbReader g(gp, gn);
    while (g.tag(f, w)) {
        if (f == 12 && w == 2) {  // GraphProto.output (ValueInfoProto)
            auto vi = g.bytes();
            PbReader r(vi.first, vi.second);
            uint32_t f2 = 0, w2 = 0;
            while (r.tag(f2, w2)) {
                if (f2 == 1 && w2 == 2) {  // ValueInfoProto.name
                    auto s = r.bytes();
                    names.emplace_back(reinterpret_cast<const char*>(s.first),
                                       s.second);
                } else {
                    r.skip(w2);
                }
            }
        } else {
            g.skip(w);
        }
    }
    return names;
}

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
        : prog_(std::move(prog)), ios_(std::move(ios)) {
        for (const auto& io : ios_) {
            if (io.dir == IODir::kInput) dtype_[io.name] = io.datatype;
        }
    }

    const std::vector<IOTensor>& ios() const override { return ios_; }

    bool run(const std::map<std::string, void*>& device_ptrs,
             const std::map<std::string, std::vector<int64_t>>& concrete,
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
            // Use the caller-supplied concrete shape for a dynamic input;
            // otherwise the program's static parameter shape.
            migraphx::shape s = pshapes[name];
            auto cit = concrete.find(name);
            if (cit != concrete.end()) {
                std::vector<size_t> lengths(cit->second.begin(),
                                            cit->second.end());
                auto dt = dtype_.find(name);
                s = migraphx::shape(
                    static_cast<migraphx_shape_datatype_t>(
                        dt == dtype_.end() ? 4 : dt->second),
                    lengths);
            }
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
    std::map<std::string, int> dtype_;
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

IOInfo introspect(const void* onnx, size_t n) {
    // Parameter and output shapes are only reliable after compilation, so
    // compile for the GPU target here as well. The network token needs this to
    // answer getInput/getOutput dimensions before buildSerializedNetwork runs.
    migraphx::onnx_options oopts;
    oopts.set_default_dim_value(1);  // pin free dims (e.g. dynamic batch) to 1
    auto prog = migraphx::parse_onnx_buffer(onnx, n, oopts);
    migraphx::target gpu("gpu");
    migraphx::compile_options copts;
    copts.set_offload_copy(true);
    prog.compile(gpu, copts);
    IOInfo info;
    info.inputs = collect_inputs(prog);
    info.outputs = collect_outputs(prog, onnx_output_names(onnx, n));
    return info;
}

std::string build(const void* onnx, size_t n, const BuildOptions& opts,
                  CalibrationSource* calib, const DynamicAxis* dyn) {
    migraphx::onnx_options oopts;
    if (dyn) {
        oopts.set_default_dyn_dim_value(migraphx::dynamic_dimension(
            static_cast<size_t>(dyn->min), static_cast<size_t>(dyn->max),
            migraphx::optimals{static_cast<size_t>(dyn->opt)}));
    } else {
        oopts.set_default_dim_value(1);  // pin free dims (e.g. dynamic batch)
    }
    auto prog = migraphx::parse_onnx_buffer(onnx, n, oopts);

    // int8 quantization runs on the uncompiled program: drive calibration
    // batches from the source into migraphx::quantize_int8 before compiling.
    if (opts.int8 && calib) {
        auto inputs = collect_inputs(prog);
        migraphx::target gpu_t("gpu");
        migraphx::quantize_int8_options qopts;
        std::vector<std::unique_ptr<CalibrationData>> kept;
        std::vector<migraphx::program_parameters> kept_pp;
        while (true) {
            auto cd = std::make_unique<CalibrationData>();
            if (!calib->next(inputs, *cd)) break;
            migraphx::program_parameters pp;
            for (const auto& in : inputs) {
                auto it = cd->tensors.find(in.name);
                if (it == cd->tensors.end()) continue;
                migraphx::shape s(
                    static_cast<migraphx_shape_datatype_t>(in.datatype),
                    std::vector<size_t>(in.dims.begin(), in.dims.end()));
                pp.add(in.name.c_str(), migraphx::argument(s, it->second.data()));
            }
            qopts.add_calibration_data(pp);
            kept_pp.push_back(std::move(pp));  // keep handles alive for quantize
            kept.push_back(std::move(cd));     // keep host data alive
        }
        if (!kept.empty()) migraphx::quantize_int8(prog, gpu_t, qopts);
    }
    apply_quantization(prog, opts);

    migraphx::target gpu("gpu");
    migraphx::compile_options copts;
    copts.set_offload_copy(true);
    prog.compile(gpu, copts);

    std::vector<IOTensor> ios;
    if (dyn) {
        // A dynamic program's shapes are not directly enumerable, so take
        // static metadata from a fixed-dim parse and mark the dynamic axes -1.
        auto sinfo = introspect(onnx, n);
        ios = std::move(sinfo.inputs);
        ios.insert(ios.end(), sinfo.outputs.begin(), sinfo.outputs.end());
        for (auto& io : ios) {
            if (io.dir == IODir::kInput && io.name == dyn->input &&
                dyn->axis >= 0 && dyn->axis < static_cast<int>(io.dims.size())) {
                io.dims[dyn->axis] = -1;
            }
            if (io.dir == IODir::kOutput && !io.dims.empty()) {
                io.dims[0] = -1;  // output batch tracks the dynamic input batch
            }
        }
    } else {
        ios = collect_inputs(prog);
        auto outs = collect_outputs(prog, onnx_output_names(onnx, n));
        ios.insert(ios.end(), outs.begin(), outs.end());
    }

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
