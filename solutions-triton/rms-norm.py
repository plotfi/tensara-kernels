"""Triton solution for `rms-norm` — mirrors solutions-cuda/rms-norm.cu."""
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
    ms = tl.sum(x * x) / D; y = x * (1.0 / tl.sqrt(ms + 1e-5))
    tl.store(y_ptr + base, y, mask=mask)


def solution(x, y, rows, D):
    BLOCK = triton.next_power_of_2(D)
    _kernel[(rows,)](x, y, D, BLOCK=BLOCK)


def main(do_check):
    R = tb.bench_size("B", 8)
    D = tb.bench_size("N", 64)
    x = tb.rand(R * D)
    y = torch.empty_like(x)
    tb.benchmark(lambda: solution(x, y, R, D))
    tb.preview(y, "Y")
    if do_check:
        xv = x.view(R, D)
        return tb.check("rms-norm", y.view(R, D), xv * torch.rsqrt(xv.pow(2).mean(1, keepdim=True) + 1e-5))
    return 0


if __name__ == "__main__":
    tb.run("rms-norm", main)
