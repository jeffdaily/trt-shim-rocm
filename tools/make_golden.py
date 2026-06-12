#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>
#
# Generic golden generator for the model breadth sweep: given an ONNX model,
# write a deterministic input and the onnxruntime CPU reference argmax, so the
# shim's GPU output can be checked for agreement on diverse architectures.
#
# Usage: make_golden.py <model.onnx> <out_prefix>
# Writes <out_prefix>_input.bin and <out_prefix>_golden.txt.

import sys
import numpy as np
import onnxruntime as ort

model, prefix = sys.argv[1], sys.argv[2]
RNG = np.random.default_rng(20260612)

sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
x = (RNG.standard_normal(shape) * 0.3).astype(np.float32)
open(f"{prefix}_input.bin", "wb").write(x.tobytes(order="C"))

out = np.asarray(sess.run(None, {inp.name: x})[0]).reshape(-1)
argmax = int(np.argmax(out))
open(f"{prefix}_golden.txt", "w").write(f"{argmax}\n")
print(f"{model}: input {inp.name} {shape}, ref argmax={argmax}")
