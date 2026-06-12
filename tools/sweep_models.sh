#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>
#
# Model breadth sweep: download a set of diverse ONNX vision classifiers, build
# and run each through the shim on the GPU, and check the argmax against the
# onnxruntime CPU reference. Demonstrates feature-completeness across distinct
# CNN architectures (residual, fire, depthwise-separable, multi-branch concat).
# Models are large, so they are downloaded to /tmp and not committed.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run="${here}/build/trtshim_run"
tmp=/tmp

declare -A URL=(
  [resnet50]="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
  [squeezenet]="https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
  [mobilenet]="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
  [inception]="https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx"
)

pass=0; total=0
for name in resnet50 squeezenet mobilenet inception; do
  total=$((total+1))
  m="${tmp}/${name}.onnx"
  [ -f "$m" ] || curl -sL --max-time 120 -o "$m" "${URL[$name]}"
  python3 "${here}/tools/make_golden.py" "$m" "${tmp}/${name}" >/dev/null 2>&1
  r=$("$run" "$m" "${tmp}/${name}_input.bin" "${tmp}/${name}_golden.txt" 2>/dev/null | tail -1)
  echo "${name}: ${r}"
  [[ "$r" == PASSED ]] && pass=$((pass+1))
done
echo "sweep: ${pass}/${total} models match onnxruntime"
[ "$pass" -eq "$total" ]
