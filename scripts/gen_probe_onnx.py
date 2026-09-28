#!/usr/bin/env python3
"""Minimal ONNX used to probe whether a TensorRT build actually supports the
host GPU's compute capability (spec §4.1).

Conv1d + LayerNorm + MatMul + LogSoftmax over a dynamic time axis. Deliberately
Conformer-shaped rather than a bare Conv: a single convolution can build on a
TRT that would still fall over on attention-style graphs, and the whole point of
the probe is to fail here instead of three steps later.

Weights are deterministic (fixed seed) so the probe file hashes identically
across runs -- the hash goes into cache/env.json as evidence of what was checked.
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

C_IN, C_OUT, V = 64, 128, 256


def build() -> onnx.ModelProto:
    rng = np.random.default_rng(0)

    initializers = [
        numpy_helper.from_array(
            (rng.standard_normal((C_OUT, C_IN, 3)) * 0.02).astype(np.float32), "w_conv"
        ),
        numpy_helper.from_array(np.ones(C_OUT, dtype=np.float32), "w_ln"),
        numpy_helper.from_array(np.zeros(C_OUT, dtype=np.float32), "b_ln"),
        numpy_helper.from_array(
            (rng.standard_normal((C_OUT, V)) * 0.02).astype(np.float32), "w_proj"
        ),
    ]

    nodes = [
        helper.make_node(
            "Conv", ["features", "w_conv"], ["conv_out"],
            kernel_shape=[3], pads=[1, 1], strides=[2],
        ),
        helper.make_node("Relu", ["conv_out"], ["act"]),
        helper.make_node("Transpose", ["act"], ["act_t"], perm=[0, 2, 1]),
        helper.make_node(
            "LayerNormalization", ["act_t", "w_ln", "b_ln"], ["ln"], axis=-1
        ),
        helper.make_node("MatMul", ["ln", "w_proj"], ["logits"]),
        helper.make_node("LogSoftmax", ["logits"], ["log_probs"], axis=-1),
    ]

    graph = helper.make_graph(
        nodes,
        "sm_probe",
        [helper.make_tensor_value_info(
            "features", TensorProto.FLOAT, ["batch", C_IN, "seq"])],
        [helper.make_tensor_value_info(
            "log_probs", TensorProto.FLOAT, ["batch", "seq_out", V])],
        initializers,
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "probe.onnx"
    onnx.save(build(), out_path)
    print(f"wrote {out_path}")
