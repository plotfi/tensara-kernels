"""Triton solution for `threshold` — mirrors solutions-cuda/threshold.cu."""
import os
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb


@triton.jit
def _kernel(x_ptr, y_ptr, n, thr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.where(x > thr, 1.0, 0.0), mask=mask)


def solution(x, y, n, thr=0.5):
    BLOCK = 1024
    _kernel[(triton.cdiv(n, BLOCK),)](x, y, n, thr, BLOCK=BLOCK)


def main(do_check):
    h = tb.bench_size("HEIGHT", 64)
    w = tb.bench_size("WIDTH", 64)
    n = h * w
    x = tb.rand(n)
    y = torch.empty_like(x)
    tb.benchmark(lambda: solution(x, y, n, 0.5))
    tb.preview(y, "output_image")
    if do_check:
        return tb.check("threshold", y, (x > 0.5).float())
    return 0


if __name__ == "__main__":
    tb.run("threshold", main)
