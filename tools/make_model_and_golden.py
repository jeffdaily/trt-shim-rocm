#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>
#
# Generate a small, deterministic MNIST-shaped CNN ONNX model plus a fixed
# input and a reference output (the "golden"). The golden is computed with the
# ONNX ReferenceEvaluator, the spec-defined reference backend, so the smoke
# test checks MIGraphX against the ONNX standard rather than against another
# accelerator. Self-contained: no network, no onnxruntime, fully reproducible.
#
# Outputs (relative to the repo root):
#   test/models/mnist_cnn.onnx   the model
#   test/golden/input.bin        raw float32 input, shape [1,1,28,28], C order
#   test/golden/golden.txt       "argmax\n" followed by the 10 logits

import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20260612)

N, C, H, W = 1, 1, 28, 28
K1, K2, FC = 8, 16, 10


def conv_w(out_c, in_c, k):
    return (RNG.standard_normal((out_c, in_c, k, k)) * 0.1).astype(np.float32)


def init(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name)


def build_model():
    # input -> Conv(8,3x3,pad1) -> Relu -> MaxPool/2 -> Conv(16,3x3,pad1)
    #       -> Relu -> MaxPool/2 -> Flatten -> Gemm(10) -> output
    inits = [
        init("c1_w", conv_w(K1, C, 3)),
        init("c1_b", np.zeros(K1)),
        init("c2_w", conv_w(K2, K1, 3)),
        init("c2_b", np.zeros(K2)),
        init("fc_w", (RNG.standard_normal((FC, K2 * 7 * 7)) * 0.05)),
        init("fc_b", np.zeros(FC)),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "c1_w", "c1_b"], ["c1"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("MaxPool", ["r1"], ["p1"],
                         kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["p1", "c2_w", "c2_b"], ["c2"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c2"], ["r2"]),
        helper.make_node("MaxPool", ["r2"], ["p2"],
                         kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Flatten", ["p2"], ["f"], axis=1),
        helper.make_node("Gemm", ["f", "fc_w", "fc_b"], ["output"],
                         transB=1),
    ]
    graph = helper.make_graph(
        nodes, "mnist_cnn", [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [N, C, H, W]),
        ], [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [N, FC]),
        ], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def main():
    model = build_model()
    (ROOT / "test/models").mkdir(parents=True, exist_ok=True)
    (ROOT / "test/golden").mkdir(parents=True, exist_ok=True)
    model_path = ROOT / "test/models/mnist_cnn.onnx"
    onnx.save(model, str(model_path))

    x = (RNG.standard_normal((N, C, H, W)) * 0.5 + 0.1).astype(np.float32)
    (ROOT / "test/golden/input.bin").write_bytes(x.tobytes(order="C"))

    sess = ReferenceEvaluator(str(model_path))
    out = sess.run(None, {"input": x})[0].reshape(-1).astype(np.float32)
    argmax = int(np.argmax(out))

    with open(ROOT / "test/golden/golden.txt", "w") as f:
        f.write(f"{argmax}\n")
        f.write(" ".join(f"{v:.6f}" for v in out) + "\n")

    print(f"wrote {model_path} ({model_path.stat().st_size} bytes)")
    print(f"reference argmax={argmax} logits={out}")


if __name__ == "__main__":
    main()
