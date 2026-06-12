#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>
#
# ONNX operator-conformance scoreboard for the shim. Drives the standardized
# onnx.backend.test node test cases (shipped with the onnx package) through the
# shim's generic runner (trtshim_infer) on the GPU and compares each output to the
# ONNX expected result. Produces a per-operator pass/fail/skip scoreboard --
# the empirical feature-completeness picture.
#
# Usage: onnx_backend_scoreboard.py <trtshim_infer> [limit] [filter]
# Writes a summary to stdout and test/onnx_scoreboard.md.

import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np
import onnx
from onnx import numpy_helper

RUNNER = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
FILTER = sys.argv[3] if len(sys.argv) > 3 else ""
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = os.path.join(os.path.dirname(onnx.__file__), "backend", "test", "data", "node")

TRT_TO_NP = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.int32, 4: bool,
             5: np.uint8, 7: np.bfloat16 if hasattr(np, "bfloat16") else np.float16,
             8: np.int64}


def load_case(d):
    model = os.path.join(d, "model.onnx")
    ds = os.path.join(d, "test_data_set_0")
    if not os.path.isfile(model) or not os.path.isdir(ds):
        return None
    m = onnx.load(model)
    inits = {i.name for i in m.graph.initializer}
    in_names = [i.name for i in m.graph.input if i.name not in inits]
    ins, outs = [], []
    for f in sorted(os.listdir(ds)):
        t = onnx.load_tensor(os.path.join(ds, f))
        arr = numpy_helper.to_array(t)
        (ins if f.startswith("input") else outs).append(arr)
    op = m.graph.node[0].op_type if m.graph.node else "?"
    return model, op, in_names, ins, outs


def run_case(d, tmp):
    c = load_case(d)
    if c is None:
        return "skip", "no-data"
    model, op, in_names, ins, outs = c
    if len(ins) != len(in_names) or not outs:
        return "skip", "io-shape"
    if any(a.dtype == object for a in ins + outs):
        return "skip", "non-tensor"
    args = [RUNNER, model, tmp]
    for name, arr in zip(in_names, ins):
        p = os.path.join(tmp, f"in_{len(args)}.bin")
        open(p, "wb").write(np.ascontiguousarray(arr).tobytes())
        args.append(f"{name}={p}")
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "fail", "timeout"
    if r.returncode != 0:
        return "fail", "run:" + (r.stderr.strip().split("\n")[-1][:40] or "rc")
    # compare first output
    exp = outs[0]
    san = "".join("_" if ch in "/: " else ch for ch in
                  [ln.split()[1] for ln in r.stdout.splitlines()
                   if ln.startswith("OUT")][0])
    binf = os.path.join(tmp, san + ".bin")
    if not os.path.isfile(binf):
        return "fail", "no-output"
    got = np.frombuffer(open(binf, "rb").read(), dtype=exp.dtype.type)
    if got.size != exp.size:
        return "fail", f"size {got.size}!={exp.size}"
    got = got.reshape(exp.shape)
    if np.issubdtype(exp.dtype, np.floating):
        ok = np.allclose(got, exp, rtol=1e-2, atol=1e-3, equal_nan=True)
    else:
        ok = np.array_equal(got, exp)
    return ("pass" if ok else "fail"), ("" if ok else "mismatch")


def main():
    dirs = sorted(d for d in os.listdir(NODE)
                  if d.startswith("test_") and (FILTER in d))
    if LIMIT:
        dirs = dirs[:LIMIT]
    tally = Counter()
    by_op = defaultdict(lambda: Counter())
    reasons = Counter()
    for name in dirs:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                c = load_case(os.path.join(NODE, name))
                op = c[1] if c else name
                status, why = run_case(os.path.join(NODE, name), tmp)
            except Exception as e:
                status, why, op = "fail", "harness:" + str(e)[:30], name
        tally[status] += 1
        by_op[op][status] += 1
        if why:
            reasons[why] += 1

    total = sum(tally.values())
    lines = ["# ONNX backend-test scoreboard (shim on gfx90a)", "",
             f"Ran {total} node test cases through trtshim_infer.", "",
             f"- pass: {tally['pass']}", f"- fail: {tally['fail']}",
             f"- skip: {tally['skip']}", ""]
    # operators fully passing vs failing
    full = sorted(op for op, c in by_op.items() if c["pass"] and not c["fail"])
    partial = sorted(op for op, c in by_op.items() if c["pass"] and c["fail"])
    none = sorted(op for op, c in by_op.items() if not c["pass"] and c["fail"])
    lines += [f"## Operators fully passing ({len(full)})", ", ".join(full), "",
              f"## Operators partially passing ({len(partial)})",
              ", ".join(partial), "",
              f"## Operators with no pass ({len(none)})", ", ".join(none), "",
              "## Top failure reasons"]
    for why, n in reasons.most_common(15):
        lines += [f"- {why}: {n}"]
    out = "\n".join(lines) + "\n"
    open(os.path.join(ROOT, "test", "onnx_scoreboard.md"), "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
