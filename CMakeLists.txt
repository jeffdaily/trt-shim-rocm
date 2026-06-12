# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>
cmake_minimum_required(VERSION 3.21)
project(trt-shim-rocm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_PREFIX_PATH AND DEFINED ENV{ROCM_PATH})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
endif()
if(NOT CMAKE_PREFIX_PATH)
  list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
endif()

find_package(migraphx REQUIRED)

# Phase 0: backend smoke test (MIGraphX only, no TensorRT shim yet).
add_executable(migraphx_smoke tools/migraphx_smoke.cpp)
target_link_libraries(migraphx_smoke PRIVATE migraphx::c)

enable_testing()
add_test(
  NAME migraphx_smoke
  COMMAND migraphx_smoke
    ${CMAKE_CURRENT_SOURCE_DIR}/test/models/mnist_cnn.onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/input.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/golden.txt)
