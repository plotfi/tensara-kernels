"""Triton solution for `frobenius-norm` — mirrors solutions-cuda/frobenius-norm.cu."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb

@triton.jit
def _sumsq(x_ptr, s_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.atomic_add(s_ptr, tl.sum(tl.where(mask, x * x, 0.0)))

@triton.jit
def _norm(x_ptr, y_ptr, s_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    inv = 1.0 / tl.sqrt(tl.load(s_ptr))
    tl.store(y_ptr + offs, x * inv, mask=mask)

def solution(x, y, n, s):
    s.zero_()
    BLOCK = 1024; g = (triton.cdiv(n, BLOCK),)
    _sumsq[g](x, s, n, BLOCK=BLOCK)
    _norm[g](x, y, s, n, BLOCK=BLOCK)

def main(do_check):
    n = tb.bench_size("SIZE", 4096)
    x = tb.rand(n); y = torch.empty_like(x); s = torch.zeros(1, device="cuda")
    tb.benchmark(lambda: solution(x, y, n, s))
    tb.preview(y, "Y")
    if do_check:
        return tb.check("frobenius-norm", y, x / x.norm(), rtol=1e-3, atol=1e-4)
    return 0
if __name__ == "__main__": tb.run("frobenius-norm", main)
