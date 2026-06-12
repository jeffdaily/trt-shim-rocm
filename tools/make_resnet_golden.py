#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>
#
# Generate a deterministic input and reference argmax for ResNet-50, so the
# shim's GPU output can be checked against the ONNX reference on a real,
# representative model. Uses a fixed random input (no image-data wrangling); the
# point is shim-vs-reference agreement, not a specific class.
#
# Input: /tmp/resnet50.onnx  Output (gitignored, large model not committed):
#   /tmp/resnet_input.bin, /tmp/resnet_golden.txt

import numpy as np
import onnxruntime as ort

RNG = np.random.default_rng(20260612)
M = "/tmp/resnet50.onnx"

sess = ort.InferenceSession(M, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
x = (RNG.standard_normal((1, 3, 224, 224)) * 0.3).astype(np.float32)
open("/tmp/resnet_input.bin", "wb").write(x.tobytes(order="C"))

out = np.asarray(sess.run(None, {iname: x})[0]).reshape(-1)
argmax = int(np.argmax(out))
open("/tmp/resnet_golden.txt", "w").write(f"{argmax}\n")
print(f"resnet input name={iname} reference argmax={argmax} top={out[argmax]:.4f}")
