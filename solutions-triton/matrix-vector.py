"""Triton solution for `matrix-vector` — mirrors solutions-cuda/matrix-vector.cu."""
import os
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb


@triton.jit
def _kernel(a_ptr, b_ptr, c_ptr, K, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    a = tl.load(a_ptr + row * K + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(c_ptr + row, tl.sum(a * b))


def solution(a, b, c, m, k):
    BLOCK = triton.next_power_of_2(k)
    _kernel[(m,)](a, b, c, k, BLOCK=BLOCK)


def main(do_check):
    m = tb.bench_size("M", 64)
    k = tb.bench_size("K", 64)
    a = tb.rand(m * k)
    b = tb.rand(k)
    c = torch.empty(m, device="cuda")
    tb.benchmark(lambda: solution(a, b, c, m, k))
    tb.preview(c, "output_c")
    if do_check:
        return tb.check("matrix-vector", c, a.view(m, k) @ b)
    return 0


if __name__ == "__main__":
    tb.run("matrix-vector", main)
