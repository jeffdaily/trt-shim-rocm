#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>
#
# Build and run the trt-shim-rocm validation suite on a real AMD GPU.
# This is the authoritative gate: a passing run means the shim (and, in Phase 0,
# the MIGraphX backend it builds on) produces correct results on this hardware.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${here}/build"
rocm="${ROCM_PATH:-/opt/rocm}"

echo "== regenerating model + golden (deterministic) =="
python3 "${here}/tools/make_model_and_golden.py"

echo "== configure + build =="
cmake -S "${here}" -B "${build}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${rocm}"
cmake --build "${build}" -j

echo "== run GPU tests =="
ctest --test-dir "${build}" --output-on-failure
