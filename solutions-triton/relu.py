"""Triton solution for `relu` — mirrors kernel-harnesses/relu.cu (elementwise over n*m).

Run:
  ../.venv/bin/python relu.py            # benchmark
  ../.venv/bin/python relu.py --check    # correctness vs torch reference
Sizes honor TENSOR_SCALE / TENSOR_N / TENSOR_M like the CUDA harness.
"""
import os
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb


@triton.jit
def _relu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


def solution(x, y, total):
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _relu_kernel[grid](x, y, total, BLOCK=BLOCK)


def main(do_check):
    n = tb.bench_size("N", 64)
    m = tb.bench_size("M", 64)
    total = n * m
    x = tb.rand(total)
    y = torch.empty_like(x)
    tb.benchmark(lambda: solution(x, y, total))
    tb.preview(y, "d_output")
    if do_check:
        return tb.check("relu", y, torch.relu(x))
    return 0


if __name__ == "__main__":
    tb.run("relu", main)
