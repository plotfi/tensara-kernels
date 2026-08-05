"""Triton solution for `l2-norm` — mirrors solutions-cuda/l2-norm.cu."""
import os
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb


@triton.jit
def _kernel(x_ptr, y_ptr, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    s = tl.sqrt(tl.sum(x * x)); y = x / s
    tl.store(y_ptr + base, y, mask=mask)


def solution(x, y, rows, D):
    BLOCK = triton.next_power_of_2(D)
    _kernel[(rows,)](x, y, D, BLOCK=BLOCK)


def main(do_check):
    R = tb.bench_size("B", 8)
    D = tb.bench_size("D", 64)
    x = tb.rand(R * D)
    y = torch.empty_like(x)
    tb.benchmark(lambda: solution(x, y, R, D))
    tb.preview(y, "Y")
    if do_check:
        xv = x.view(R, D)
        return tb.check("l2-norm", y.view(R, D), xv / xv.pow(2).sum(1, keepdim=True).sqrt())
    return 0


if __name__ == "__main__":
    tb.run("l2-norm", main)
