"""Triton solution for `mse-loss` — mirrors solutions-cuda/mse-loss.cu."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb

@triton.jit
def _k(p_ptr, t_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    t = tl.load(t_ptr + offs, mask=mask, other=0.0)
    d = p - t
    tl.atomic_add(out_ptr, tl.sum(tl.where(mask, d * d, 0.0)) / n)

def solution(p, t, out, n):
    out.zero_()
    BLOCK = 1024
    _k[(triton.cdiv(n, BLOCK),)](p, t, out, n, BLOCK=BLOCK)

def main(do_check):
    # mse-loss reduces over M*N (matches the CUDA harness's 2-D shape).
    n = tb.bench_size("M", 64) * tb.bench_size("N", 64)
    p = tb.rand(n); t = tb.rand(n); out = torch.zeros(1, device="cuda")
    tb.benchmark(lambda: solution(p, t, out, n))
    tb.preview(out, "output")
    if do_check:
        return tb.check("mse-loss", out, ((p - t) ** 2).mean().view(1), rtol=1e-2, atol=1e-3)
    return 0
if __name__ == "__main__": tb.run("mse-loss", main)
