"""Triton solution for `vector-addition` — mirrors kernel-harnesses/vector-addition.cu.

Run:
  ../.venv/bin/python vector-addition.py            # benchmark
  ../.venv/bin/python vector-addition.py --check    # correctness vs torch reference
Sizes honor TENSOR_SCALE / TENSOR_N just like the CUDA harness.
"""
import os
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb


@triton.jit
def _add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)


# The backend "solution" — same contract as the CUDA solution(): device tensors + n.
def solution(a, b, c, n):
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _add_kernel[grid](a, b, c, n, BLOCK=BLOCK)


def main(do_check):
    n = tb.bench_size("N", 1024)
    a = tb.rand(n)
    b = tb.rand(n)
    c = torch.empty_like(a)
    tb.benchmark(lambda: solution(a, b, c, n))
    tb.preview(c, "d_output")
    if do_check:
        return tb.check("vector-addition", c, a + b)
    return 0


if __name__ == "__main__":
    tb.run("vector-addition", main)
