// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// The nvinfer1 / nvonnxparser shim: concrete implementations of the TensorRT
// interfaces backed by MIGraphX (via backend.h), plus the extern "C" factory
// entry points an app pulls from libnvinfer. Each public IXxx interface
// forwards to a protected apiv::VXxx; the shim classes derive from both and set
// mImpl=this. Critical-path methods are implemented here; the rest are trivial
// overrides included from src/generated/*.stubs.inc.

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "backend.h"

namespace trtshim {

// Called by generated stubs for unimplemented methods that can safely return a
// default. Silent unless TRTSHIM_DEBUG is set.
void trtshim_unimpl(const char* what) noexcept {
    if (std::getenv("TRTSHIM_DEBUG")) {
        std::fprintf(stderr, "[trtshim] unimplemented (returning default): %s\n",
                     what);
    }
}

// Called by generated stubs for reference-returning methods, which cannot
// return a default. Reaching one means an app used an unsupported path.
[[noreturn]] void trtshim_die(const char* what) noexcept {
    std::fprintf(stderr, "[trtshim] fatal: unsupported method reached: %s\n",
                 what);
    std::abort();
}

}  // namespace trtshim

// The shim classes live in namespace nvinfer1 so the generated apiv stub
// overrides (whose signatures use unqualified nvinfer1 type names like Dims and
// ITensor, since the V-classes are declared in namespace nvinfer1) resolve.
namespace nvinfer1 {
namespace shim {

using namespace ::trtshim;

nvinfer1::DataType to_trt_dtype(int mgx) {
    using DT = nvinfer1::DataType;
    switch (mgx) {
        case 4: return DT::kFLOAT;   // float_type
        case 3: return DT::kHALF;    // half_type
        case 7: return DT::kINT8;    // int8_type
        case 10: return DT::kINT32;  // int32_type
        case 2: return DT::kBOOL;    // bool_type
        case 6: return DT::kUINT8;   // uint8_type
        case 11: return DT::kINT64;  // int64_type
        case 16: return DT::kBF16;   // bf16_type
        default: return DT::kFLOAT;
    }
}

nvinfer1::Dims to_dims(const std::vector<int64_t>& v) {
    nvinfer1::Dims d{};
    d.nbDims = static_cast<int32_t>(v.size());
    for (size_t i = 0; i < v.size() && i < nvinfer1::Dims::MAX_DIMS; ++i) {
        d.d[i] = v[i];
    }
    return d;
}

// ---- ITensor -------------------------------------------------------------
class ShimTensor : public nvinfer1::ITensor, public nvinfer1::apiv::VTensor {
public:
    ShimTensor(std::string name, std::vector<int64_t> dims, int mgx_dtype)
        : name_(std::move(name)), dims_(std::move(dims)), dtype_(mgx_dtype) {
        mImpl = this;
    }
    nvinfer1::Dims getDimensions() const noexcept override {
        return to_dims(dims_);
    }
    char const* getName() const noexcept override { return name_.c_str(); }
    void setName(char const* name) noexcept override { name_ = name; }
    nvinfer1::DataType getType() const noexcept override {
        return to_trt_dtype(dtype_);
    }
#include "generated/VTensor.stubs.inc"

private:
    std::string name_;
    std::vector<int64_t> dims_;
    int dtype_;
};

// ---- IHostMemory ---------------------------------------------------------
class ShimHostMemory : public nvinfer1::IHostMemory,
                       public nvinfer1::apiv::VHostMemory {
public:
    explicit ShimHostMemory(std::string blob) : blob_(std::move(blob)) {
        mImpl = this;
    }
    void* data() const noexcept override {
        return const_cast<char*>(blob_.data());
    }
    std::size_t size() const noexcept override { return blob_.size(); }
    nvinfer1::DataType type() const noexcept override {
        return nvinfer1::DataType::kINT8;
    }

private:
    std::string blob_;
};

// ---- INetworkDefinition (token) ------------------------------------------
class ShimNetwork : public nvinfer1::INetworkDefinition,
                    public nvinfer1::apiv::VNetworkDefinition {
public:
    ShimNetwork() { mImpl = this; }
    char const* getName() const noexcept override { return "trtshim_network"; }

    int32_t getNbInputs() const noexcept override {
        return static_cast<int32_t>(inputs_.size());
    }
    int32_t getNbOutputs() const noexcept override {
        return static_cast<int32_t>(outputs_.size());
    }
    nvinfer1::ITensor* getInput(int32_t i) const noexcept override {
        return (i >= 0 && i < (int32_t)inputs_.size()) ? inputs_[i].get()
                                                       : nullptr;
    }
    nvinfer1::ITensor* getOutput(int32_t i) const noexcept override {
        return (i >= 0 && i < (int32_t)outputs_.size()) ? outputs_[i].get()
                                                        : nullptr;
    }

    // Non-virtual shim plumbing (not part of the nvinfer1 interface).
    void capture(std::string bytes) { onnx_ = std::move(bytes); }
    const std::string& onnx() const { return onnx_; }
    void populate(const IOInfo& info) {
        inputs_.clear();
        outputs_.clear();
        for (const auto& t : info.inputs)
            inputs_.push_back(std::make_unique<ShimTensor>(t.name, t.dims,
                                                           t.datatype));
        for (const auto& t : info.outputs)
            outputs_.push_back(std::make_unique<ShimTensor>(t.name, t.dims,
                                                            t.datatype));
    }
#include "generated/VNetworkDefinition.stubs.inc"

private:
    std::string onnx_;
    std::vector<std::unique_ptr<ShimTensor>> inputs_;
    std::vector<std::unique_ptr<ShimTensor>> outputs_;
};

// ---- IBuilderConfig ------------------------------------------------------
class ShimConfig : public nvinfer1::IBuilderConfig,
                   public nvinfer1::apiv::VBuilderConfig {
public:
    ShimConfig() { mImpl = this; }
    void setFlag(nvinfer1::BuilderFlag f) noexcept override {
        flags_ |= (1U << static_cast<uint32_t>(f));
    }
    bool getFlag(nvinfer1::BuilderFlag f) const noexcept override {
        return (flags_ >> static_cast<uint32_t>(f)) & 1U;
    }
    void clearFlag(nvinfer1::BuilderFlag f) noexcept override {
        flags_ &= ~(1U << static_cast<uint32_t>(f));
    }
    BuildOptions options() const {
        BuildOptions o;
        o.fp16 = getFlag(nvinfer1::BuilderFlag::kFP16);
        o.int8 = getFlag(nvinfer1::BuilderFlag::kINT8);
        o.bf16 = getFlag(nvinfer1::BuilderFlag::kBF16);
        return o;
    }
#include "generated/VBuilderConfig.stubs.inc"

private:
    uint32_t flags_ = 0;
};

// ---- IExecutionContext ---------------------------------------------------
class ShimContext : public nvinfer1::IExecutionContext,
                    public nvinfer1::apiv::VExecutionContext {
public:
    ShimContext(Engine* engine, const std::vector<IOTensor>* ios)
        : engine_(engine), ios_(ios) {
        mImpl = this;
    }
    nvinfer1::IExecutionContext* getPImpl() noexcept override { return this; }

    bool setTensorAddress(char const* name, void* data) noexcept override {
        addrs_[name] = data;
        return true;
    }
    bool setInputShape(char const*, nvinfer1::Dims const&) noexcept override {
        return true;  // static shapes in this phase
    }
    bool executeV2(void* const* bindings) noexcept override {
        std::map<std::string, void*> ptrs;
        for (size_t i = 0; i < ios_->size(); ++i)
            ptrs[(*ios_)[i].name] = bindings[i];
        return engine_->run(ptrs, nullptr);
    }
    bool enqueueV3(cudaStream_t stream) noexcept override {
        return engine_->run(addrs_, static_cast<void*>(stream));
    }
#include "generated/VExecutionContext.stubs.inc"

private:
    Engine* engine_;
    const std::vector<IOTensor>* ios_;
    std::map<std::string, void*> addrs_;
};

// ---- ICudaEngine ---------------------------------------------------------
class ShimEngine : public nvinfer1::ICudaEngine,
                   public nvinfer1::apiv::VCudaEngine {
public:
    ShimEngine(std::unique_ptr<Engine> engine, std::string blob)
        : engine_(std::move(engine)), blob_(std::move(blob)) {
        mImpl = this;
    }
    nvinfer1::ICudaEngine* getPImpl() noexcept override { return this; }

    int32_t getNbIOTensors() const noexcept override {
        return static_cast<int32_t>(ios().size());
    }
    char const* getIOTensorName(int32_t i) const noexcept override {
        return (i >= 0 && i < (int32_t)ios().size()) ? ios()[i].name.c_str()
                                                     : nullptr;
    }
    nvinfer1::Dims getTensorShape(char const* name) const noexcept override {
        const IOTensor* t = find(name);
        return t ? to_dims(t->dims) : nvinfer1::Dims{};
    }
    nvinfer1::DataType getTensorDataType(char const* name) const noexcept override {
        const IOTensor* t = find(name);
        return t ? to_trt_dtype(t->datatype) : nvinfer1::DataType::kFLOAT;
    }
    nvinfer1::TensorIOMode getTensorIOMode(char const* name) const noexcept override {
        const IOTensor* t = find(name);
        if (!t) return nvinfer1::TensorIOMode::kNONE;
        return t->dir == IODir::kInput ? nvinfer1::TensorIOMode::kINPUT
                                       : nvinfer1::TensorIOMode::kOUTPUT;
    }
    int32_t getTensorComponentsPerElement(char const*) const noexcept override {
        return 1;
    }
    int32_t getTensorVectorizedDim(char const*) const noexcept override {
        return -1;
    }
    int32_t getTensorBytesPerComponent(char const* name) const noexcept override {
        const IOTensor* t = find(name);
        return t ? static_cast<int32_t>(t->bytes_per_elem) : 0;
    }
    nvinfer1::IExecutionContext* createExecutionContext(
        nvinfer1::ExecutionContextAllocationStrategy) noexcept override {
        return new ShimContext(engine_.get(), &ios());
    }
    nvinfer1::IExecutionContext* createExecutionContextWithoutDeviceMemory()
        noexcept override {
        return new ShimContext(engine_.get(), &ios());
    }
    nvinfer1::IHostMemory* serialize() const noexcept override {
        return new ShimHostMemory(blob_);
    }
    char const* getName() const noexcept override { return "trtshim_engine"; }
    int32_t getNbLayers() const noexcept override { return 1; }
#include "generated/VCudaEngine.stubs.inc"

private:
    const std::vector<IOTensor>& ios() const { return engine_->ios(); }
    const IOTensor* find(const char* name) const {
        for (const auto& t : ios())
            if (t.name == name) return &t;
        return nullptr;
    }
    std::unique_ptr<Engine> engine_;
    std::string blob_;
};

// ---- IRuntime ------------------------------------------------------------
class ShimRuntime : public nvinfer1::IRuntime, public nvinfer1::apiv::VRuntime {
public:
    explicit ShimRuntime(nvinfer1::ILogger* logger) : logger_(logger) {
        mImpl = this;
    }
    nvinfer1::IRuntime* getPImpl() noexcept override { return this; }
    nvinfer1::ILogger* getLogger() const noexcept override { return logger_; }

    nvinfer1::ICudaEngine* deserializeCudaEngine(void const* blob,
                                                 std::size_t size) noexcept override {
        std::string err;
        auto engine = load(blob, size, err);
        if (!engine) {
            if (logger_) {
                logger_->log(nvinfer1::ILogger::Severity::kERROR, err.c_str());
            }
            return nullptr;
        }
        return new ShimEngine(std::move(engine),
                              std::string(static_cast<const char*>(blob), size));
    }
    nvinfer1::ICudaEngine* deserializeCudaEngine(
        nvinfer1::IStreamReader&) noexcept override {
        return nullptr;  // stream-reader deserialize not supported in this phase
    }
#include "generated/VRuntime.stubs.inc"

private:
    nvinfer1::ILogger* logger_;
};

// ---- IBuilder ------------------------------------------------------------
class ShimBuilder : public nvinfer1::IBuilder, public nvinfer1::apiv::VBuilder {
public:
    explicit ShimBuilder(nvinfer1::ILogger* logger) : logger_(logger) {
        mImpl = this;
    }
    nvinfer1::ILogger* getLogger() const noexcept override { return logger_; }

    nvinfer1::IBuilderConfig* createBuilderConfig() noexcept override {
        return new ShimConfig();
    }
    nvinfer1::INetworkDefinition* createNetworkV2(
        nvinfer1::NetworkDefinitionCreationFlags) noexcept override {
        return new ShimNetwork();
    }
    nvinfer1::IHostMemory* buildSerializedNetwork(
        nvinfer1::INetworkDefinition& network,
        nvinfer1::IBuilderConfig& config) noexcept override {
        auto& net = static_cast<ShimNetwork&>(network);
        auto& cfg = static_cast<ShimConfig&>(config);
        try {
            std::string blob = build(net.onnx().data(), net.onnx().size(),
                                     cfg.options(), /*output_names=*/{});
            return new ShimHostMemory(std::move(blob));
        } catch (const std::exception& e) {
            if (logger_) {
                logger_->log(nvinfer1::ILogger::Severity::kERROR, e.what());
            }
            return nullptr;
        }
    }
#include "generated/VBuilder.stubs.inc"

private:
    nvinfer1::ILogger* logger_;
};

// ---- nvonnxparser::IParser (plain interface, no apiv) ---------------------
class ShimParser : public nvonnxparser::IParser {
public:
    ShimParser(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger)
        : network_(static_cast<ShimNetwork*>(network)), logger_(logger) {}

    bool parse(void const* data, size_t size,
               const char* /*model_path*/) noexcept override {
        return ingest(static_cast<const char*>(data), size);
    }
    bool parseFromFile(const char* file, int /*verbosity*/) noexcept override {
        std::FILE* f = std::fopen(file, "rb");
        if (!f) return false;
        std::fseek(f, 0, SEEK_END);
        long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::string bytes(static_cast<size_t>(n), '\0');
        size_t got = std::fread(bytes.data(), 1, static_cast<size_t>(n), f);
        std::fclose(f);
        if (got != static_cast<size_t>(n)) return false;
        return ingest(bytes.data(), bytes.size());
    }
    bool parseWithWeightDescriptors(void const* data,
                                    size_t size) noexcept override {
        return ingest(static_cast<const char*>(data), size);
    }
    bool supportsModel(void const*, size_t, SubGraphCollection_t&,
                       const char*) noexcept override {
        return true;
    }
    bool supportsModelV2(void const*, size_t, char const*) noexcept override {
        return true;
    }
    bool supportsOperator(const char*) const noexcept override { return true; }
    int getNbErrors() const noexcept override { return 0; }
    nvonnxparser::IParserError const* getError(int) const noexcept override {
        return nullptr;
    }
    void clearErrors() noexcept override {}
    char const* const* getUsedVCPluginLibraries(
        int64_t& nbPluginLibs) const noexcept override {
        nbPluginLibs = 0;
        return nullptr;
    }
    void setFlags(nvonnxparser::OnnxParserFlags) noexcept override {}
    nvonnxparser::OnnxParserFlags getFlags() const noexcept override { return 0; }
    void clearFlag(nvonnxparser::OnnxParserFlag) noexcept override {}
    void setFlag(nvonnxparser::OnnxParserFlag) noexcept override {}
    bool getFlag(nvonnxparser::OnnxParserFlag) const noexcept override {
        return false;
    }
    nvinfer1::ITensor const* getLayerOutputTensor(char const*,
                                                  int64_t) noexcept override {
        return nullptr;
    }
    int64_t getNbSubgraphs() noexcept override { return 0; }
    bool isSubgraphSupported(int64_t) noexcept override { return false; }
    int64_t* getSubgraphNodes(int64_t, int64_t& n) noexcept override {
        n = 0;
        return nullptr;
    }

private:
    bool ingest(const char* data, size_t size) {
        try {
            network_->capture(std::string(data, size));
            network_->populate(introspect(data, size, /*output_names=*/{}));
            return true;
        } catch (const std::exception& e) {
            if (logger_) {
                logger_->log(nvinfer1::ILogger::Severity::kERROR, e.what());
            }
            return false;
        }
    }
    ShimNetwork* network_;
    nvinfer1::ILogger* logger_;
};

}  // namespace shim
}  // namespace nvinfer1

// ---- factory entry points (the symbols an app pulls from libnvinfer) ------
extern "C" {

void* createInferBuilder_INTERNAL(void* logger, int /*version*/) noexcept {
    return new nvinfer1::shim::ShimBuilder(static_cast<nvinfer1::ILogger*>(logger));
}

void* createInferRuntime_INTERNAL(void* logger, int /*version*/) noexcept {
    return new nvinfer1::shim::ShimRuntime(static_cast<nvinfer1::ILogger*>(logger));
}

void* createNvOnnxParser_INTERNAL(void* network, void* logger,
                                  int /*version*/) noexcept {
    return new nvinfer1::shim::ShimParser(
        static_cast<nvinfer1::INetworkDefinition*>(network),
        static_cast<nvinfer1::ILogger*>(logger));
}

int getNvOnnxParserVersion() noexcept { return NV_ONNX_PARSER_VERSION; }

}  // extern "C"
