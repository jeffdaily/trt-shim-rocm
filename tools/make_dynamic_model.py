#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>
#
# Generate a small CNN with a genuinely dynamic (symbolic) batch dimension, plus
# a 2-batch input and per-row golden, to validate the shim's single-axis dynamic
# shape support (IOptimizationProfile). The batch axis uses a dim_param ("N") so
# MIGraphX treats it as free.
#
# Outputs:
#   test/models/dyn_cnn.onnx
#   test/golden/dyn_input2.bin     (batch-2 input, raw float32)
#   test/golden/dyn_golden.txt     (argmax row0, argmax row1)

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20260612)
C, H, W, K, FC = 1, 28, 28, 8, 10


def init(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name)


inits = [
    init("c_w", RNG.standard_normal((K, C, 3, 3)) * 0.1),
    init("c_b", np.zeros(K)),
    init("fc_w", RNG.standard_normal((FC, K * 14 * 14)) * 0.05),
    init("fc_b", np.zeros(FC)),
]
nodes = [
    helper.make_node("Conv", ["input", "c_w", "c_b"], ["c"],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["c"], ["r"]),
    helper.make_node("MaxPool", ["r"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
    helper.make_node("Flatten", ["p"], ["f"], axis=1),
    helper.make_node("Gemm", ["f", "fc_w", "fc_b"], ["output"], transB=1),
]
graph = helper.make_graph(
    nodes, "dyn_cnn",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", C, H, W])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", FC])],
    initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.checker.check_model(model)
(ROOT / "test/models").mkdir(parents=True, exist_ok=True)
onnx.save(model, str(ROOT / "test/models/dyn_cnn.onnx"))

x = (RNG.standard_normal((2, C, H, W)) * 0.5).astype(np.float32)
(ROOT / "test/golden/dyn_input2.bin").write_bytes(x.tobytes(order="C"))
sess = ort.InferenceSession(str(ROOT / "test/models/dyn_cnn.onnx"),
                            providers=["CPUExecutionProvider"])
out = np.asarray(sess.run(None, {"input": x})[0])
a0, a1 = int(np.argmax(out[0])), int(np.argmax(out[1]))
(ROOT / "test/golden/dyn_golden.txt").write_text(f"{a0} {a1}\n")
print(f"dynamic CNN: batch-2 argmax row0={a0} row1={a1}")
