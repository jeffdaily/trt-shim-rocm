#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>
#
# Generate the data the stock TensorRT sampleOnnxMNIST expects: mnist.onnx plus
# 0.pgm..9.pgm. For each digit we pick the MNIST test image the model
# classifies most confidently (softmax > 0.9) so the sample's self-check passes
# regardless of which digit its RNG selects. The sample reads a P5 PGM and
# computes the model input as (1 - pixel/255), so the PGM stores the inverted
# image (digit dark on light); we therefore write 255 - mnist_pixel.
#
# Inputs (downloaded out of band to /tmp):
#   /tmp/mnist.onnx, /tmp/t10k-images.gz, /tmp/t10k-labels.gz
# Output: <repo>/test/mnist/{mnist.onnx,0.pgm..9.pgm}

import gzip
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test/mnist"


def load_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def load_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = load_idx_images("/tmp/t10k-images.gz")
    labels = load_idx_labels("/tmp/t10k-labels.gz")

    model_path = OUT / "mnist.onnx"
    if not model_path.exists():
        onnx.save(onnx.load("/tmp/mnist.onnx"), str(model_path))
    sess = ReferenceEvaluator(str(model_path))

    def confidence(img):
        x = (img.astype(np.float32) / 255.0).reshape(1, 1, 28, 28)
        out = np.asarray(sess.run(None, {"Input3": x})[0]).reshape(-1)
        p = softmax(out)
        return int(np.argmax(p)), float(p.max())

    for digit in range(10):
        idxs = np.where(labels == digit)[0]
        best_i, best_conf = None, -1.0
        for i in idxs[:60]:  # scan a handful; clean test images classify easily
            pred, conf = confidence(images[i])
            if pred == digit and conf > best_conf:
                best_i, best_conf = i, conf
            if best_conf > 0.99:
                break
        assert best_i is not None, f"no confident sample for digit {digit}"
        pgm = (255 - images[best_i]).astype(np.uint8)  # invert for the sample
        with open(OUT / f"{digit}.pgm", "wb") as f:
            f.write(b"P5\n28 28\n255\n")
            f.write(pgm.tobytes())
        print(f"digit {digit}: test idx {best_i}, confidence {best_conf:.4f}")


if __name__ == "__main__":
    main()
