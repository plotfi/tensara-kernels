"""Triton solution for `hard-sigmoid` — elementwise over n*m (mirrors solutions-cuda/hard-sigmoid.cu)."""
import os
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb


@triton.jit
def _tanh(x):
    # exact tanh via sigmoid: 2*sigmoid(2x) - 1
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.where(x <= -3.0, 0.0, tl.where(x >= 3.0, 1.0, (x + 3.0) / 6.0))
    tl.store(y_ptr + offs, y, mask=mask)


def solution(x, y, total):
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _kernel[grid](x, y, total, BLOCK=BLOCK)


def main(do_check):
    n = tb.bench_size("N", 64)
    m = tb.bench_size("M", 64)
    total = n * m
    x = tb.rand(total)
    y = torch.empty_like(x)
    tb.benchmark(lambda: solution(x, y, total))
    tb.preview(y, "d_output")
    if do_check:
        ALPHA = 0.0
        return tb.check("hard-sigmoid", y, torch.nn.functional.hardsigmoid(x))
    return 0


if __name__ == "__main__":
    tb.run("hard-sigmoid", main)
