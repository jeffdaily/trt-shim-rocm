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

# Link-compatibility aliases: libnvinfer.so / libnvonnxparser.so -> libtrtshim.so
set(ALIASES libnvinfer.so libnvonnxparser.so)
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

enable_testing()
add_test(NAME migraphx_smoke COMMAND migraphx_smoke
  ${CMAKE_CURRENT_SOURCE_DIR}/test/models/mnist_cnn.onnx
  ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/input.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/golden.txt)
add_test(NAME driver_smoke COMMAND driver_smoke
  ${CMAKE_CURRENT_SOURCE_DIR}/test/models/mnist_cnn.onnx
  ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/input.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/test/golden/golden.txt)
