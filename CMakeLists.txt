# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>
cmake_minimum_required(VERSION 3.21)
project(trt-shim-rocm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_PREFIX_PATH AND DEFINED ENV{ROCM_PATH})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
endif()
if(NOT CMAKE_PREFIX_PATH)
  list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
endif()

find_package(migraphx REQUIRED)
find_package(hip REQUIRED)

# Generate the apiv stub overrides from the vendored TensorRT impl header.
set(GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)
add_custom_command(
  OUTPUT ${GEN_DIR}/.stamp
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_DIR}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_stubs.py
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt-headers/include/NvInferImpl.h
          ${GEN_DIR}
  COMMAND ${CMAKE_COMMAND} -E touch ${GEN_DIR}/.stamp
  DEPENDS tools/gen_stubs.py
          third_party/tensorrt-headers/include/NvInferImpl.h
  COMMENT "Generating apiv stub overrides")
add_custom_target(gen_stubs DEPENDS ${GEN_DIR}/.stamp)

# The shim library. Installed under the libnvinfer / libnvonnxparser names so an
# unmodified TensorRT consumer links against it.
add_library(trtshim SHARED src/shim.cpp src/backend.cpp)
add_dependencies(trtshim gen_stubs)
target_include_directories(trtshim PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt-headers/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/compat
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(trtshim PRIVATE migraphx::c hip::host)

# Link-compatibility aliases -> libtrtshim.so. The unversioned names are for
# link-time (-lnvinfer); the versioned .so.10 names are what runtime dlopen
# consumers (trtexec) look up (libnvinfer.so.<NV_TENSORRT_MAJOR>).
set(ALIASES libnvinfer.so libnvonnxparser.so libnvinfer.so.10 libnvonnxparser.so.10)
set(ALIAS_FILES "")
foreach(a ${ALIASES})
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/${a}
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:trtshim>
            ${CMAKE_BINARY_DIR}/${a}
    DEPENDS trtshim
    COMMENT "Aliasing ${a} -> libtrtshim.so")
  list(APPEND ALIAS_FILES ${CMAKE_BINARY_DIR}/${a})
endforeach()
add_custom_target(aliases ALL DEPENDS ${ALIAS_FILES})

# Stub libnvinfer_plugin: consumers dlopen libnvinfer_plugin.so.10 to register
# the standard NVIDIA plugins. The shim ships no plugins, so this is a no-op so
# such consumers (trtexec) load; plugin-using models fail later with a clear
# unsupported-operator error.
add_library(nvinfer_plugin SHARED src/plugin_stub.cpp)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/libnvinfer_plugin.so.10
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:nvinfer_plugin>
          ${CMAKE_BINARY_DIR}/libnvinfer_plugin.so.10
  DEPENDS nvinfer_plugin)
add_custom_target(plugin_alias ALL DEPENDS ${CMAKE_BINARY_DIR}/libnvinfer_plugin.so.10)

# Phase 0: MIGraphX-only backend smoke (no shim).
add_executable(migraphx_smoke tools/migraphx_smoke.cpp)
target_link_libraries(migraphx_smoke PRIVATE migraphx::c)

# Milestone 1a: drive the full TensorRT API path through the shim, linked via
# the libnvinfer / libnvonnxparser aliases to prove link-level drop-in.
add_executable(driver_smoke tools/driver_smoke.cpp)
add_dependencies(driver_smoke aliases)
target_include_directories(driver_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt-headers/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/compat)
target_link_libraries(driver_smoke PRIVATE
  ${CMAKE_BINARY_DIR}/libnvinfer.so
  ${CMAKE_BINARY_DIR}/libnvonnxparser.so
  hip::host)
set_target_properties(driver_smoke PROPERTIES
  BUILD_RPATH ${CMAKE_BINARY_DIR})

# Milestone 1b: the STOCK TensorRT sampleOnnxMNIST, vendored unmodified, built
# against the shim. Proves an unmodified TensorRT consumer (its common/ helpers
# call the CUDA runtime, resolved to HIP by include/compat) builds and runs on
# AMD.
set(SAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/vendor/tensorrt-samples)
add_executable(sampleOnnxMNIST
  ${SAMPLES}/sampleOnnxMNIST/sampleOnnxMNIST.cpp
  ${SAMPLES}/common/logger.cpp
  ${SAMPLES}/utils/timingCache.cpp
  ${SAMPLES}/utils/fileLock.cpp)
add_dependencies(sampleOnnxMNIST aliases)
target_include_directories(sampleOnnxMNIST PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt-headers/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/compat
  ${SAMPLES}
  ${SAMPLES}/common
  ${SAMPLES}/utils
  ${SAMPLES}/sampleOnnxMNIST)
target_link_libraries(sampleOnnxMNIST PRIVATE
  ${CMAKE_BINARY_DIR}/libnvinfer.so
  ${CMAKE_BINARY_DIR}/libnvonnxparser.so
  hip::host)
set_target_properties(sampleOnnxMNIST PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR})

# Flexible driver for fp16 + serialize round-trip validation on real models.
foreach(drv trtshim_run trtshim_dyn_run trtshim_infer)
  add_executable(${drv} tools/${drv}.cpp)
  add_dependencies(${drv} aliases)
  target_include_directories(${drv} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt-headers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/compat)
  target_link_libraries(${drv} PRIVATE
    ${CMAKE_BINARY_DIR}/libnvinfer.so
    ${CMAKE_BINARY_DIR}/libnvonnxparser.so
    hip::host)
  set_target_properties(${drv} PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR})
endforeach()

# The stock TensorRT trtexec CLI, vendored unmodified, built against the shim.
# Needs the full sample inference framework and a larger CUDA->HIP compat
# surface (include/compat).
option(BUILD_TRTEXEC "Build the vendored trtexec against the shim" ON)
if(BUILD_TRTEXEC)
  add_executable(trtexec
    ${SAMPLES}/trtexec/trtexec.cpp
    ${SAMPLES}/common/logger.cpp
    ${SAMPLES}/common/sampleDevice.cpp
    ${SAMPLES}/common/sampleEngines.cpp
    ${SAMPLES}/common/sampleInference.cpp
    ${SAMPLES}/common/sampleOptions.cpp
    ${SAMPLES}/common/sampleReporting.cpp
    ${SAMPLES}/common/sampleUtils.cpp
    ${SAMPLES}/common/getOptions.cpp
    ${SAMPLES}/common/bfloat16.cpp
    ${SAMPLES}/utils/timingCache.cpp
    ${SAMPLES}/utils/fileLock.cpp)
  add_dependencies(trtexec aliases)
  target_include_directories(trtexec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt-headers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/compat
    ${SAMPLES} ${SAMPLES}/common ${SAMPLES}/utils ${SAMPLES}/trtexec)
  target_link_libraries(trtexec PRIVATE
    ${CMAKE_BINARY_DIR}/libnvinfer.so
    ${CMAKE_BINARY_DIR}/libnvonnxparser.so
    hip::host)
  set_target_properties(trtexec PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR})
endif()

enable_testing()
add_test(NAME migraphx_smoke COMMAND migraphx_smoke
  ${CMAKE_CURRENT_SOURCE_DIR}/test/models/mnist_cnn.onnx
  ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/input.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/golden.txt)
add_test(NAME driver_smoke COMMAND driver_smoke
  ${CMAKE_CURRENT_SOURCE_DIR}/test/models/mnist_cnn.onnx
  ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/input.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/golden.txt)
add_test(NAME sampleOnnxMNIST
  COMMAND sampleOnnxMNIST -d ${CMAKE_CURRENT_SOURCE_DIR}/test/mnist)
set_tests_properties(sampleOnnxMNIST PROPERTIES
  PASS_REGULAR_EXPRESSION "PASSED TensorRT.sample_onnx_mnist")

set(MODEL ${CMAKE_CURRENT_SOURCE_DIR}/test/models/mnist_cnn.onnx)
set(INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/input.bin)
set(GOLD ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/golden.txt)
add_test(NAME trtshim_run_fp16 COMMAND trtshim_run ${MODEL} ${INPUT} ${GOLD} --fp16)
add_test(NAME trtshim_run_int8 COMMAND trtshim_run ${MODEL} ${INPUT} ${GOLD} --int8)
# Cross-process serialize round-trip: save in one run, load in the next.
add_test(NAME trtshim_run_save
  COMMAND trtshim_run ${MODEL} ${INPUT} ${GOLD} --save ${CMAKE_BINARY_DIR}/rt.engine)
add_test(NAME trtshim_run_load
  COMMAND trtshim_run ${MODEL} ${INPUT} ${GOLD} --load ${CMAKE_BINARY_DIR}/rt.engine)
set_tests_properties(trtshim_run_save PROPERTIES FIXTURES_SETUP RTENGINE)
set_tests_properties(trtshim_run_load PROPERTIES FIXTURES_REQUIRED RTENGINE)
add_test(NAME trtshim_dyn_run COMMAND trtshim_dyn_run
  ${CMAKE_CURRENT_SOURCE_DIR}/test/models/dyn_cnn.onnx
  ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/dyn_input2.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/dyn_golden.txt)
if(BUILD_TRTEXEC)
  # The stock trtexec dlopens libnvinfer.so.10 from the build dir at runtime.
  add_test(NAME trtexec
    COMMAND trtexec --onnx=${CMAKE_CURRENT_SOURCE_DIR}/test/mnist/mnist.onnx)
  set_tests_properties(trtexec PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}"
    PASS_REGULAR_EXPRESSION "PASSED TensorRT.trtexec")
  # --useCudaGraph must really capture (zero-copy run path is HIP-graph
  # capturable), not just pass via fallback.
  add_test(NAME trtexec_cudagraph
    COMMAND trtexec --onnx=${CMAKE_CURRENT_SOURCE_DIR}/test/mnist/mnist.onnx
            --useCudaGraph)
  set_tests_properties(trtexec_cudagraph PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}"
    PASS_REGULAR_EXPRESSION "Successfully captured CUDA graph")
endif()
