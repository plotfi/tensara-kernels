"""Triton solution for `kl-loss` — mirrors solutions-cuda/kl-loss.cu."""
import os
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb


@triton.jit
def _kernel(p_ptr, t_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    t = tl.load(t_ptr + offs, mask=mask, other=0.0)
    h = tl.log(t) - tl.log(p)
    h = t * h
    h = tl.where(mask, h, 0.0)
    tl.atomic_add(out_ptr, tl.sum(h) / n)


def solution(p, t, out, n):
    out.zero_()
    BLOCK = 1024
    _kernel[(triton.cdiv(n, BLOCK),)](p, t, out, n, BLOCK=BLOCK)


def main(do_check):
    n = tb.bench_size("N", 1024)
    # KL takes log(p), log(t) -> inputs MUST be strictly positive (tb.rand is [-1,1]).
    p = torch.rand(n, device="cuda") * 0.999 + 1e-3   # (0, 1]
    t = torch.rand(n, device="cuda") * 0.999 + 1e-3   # (0, 1]
    out = torch.zeros(1, device="cuda")
    tb.benchmark(lambda: solution(p, t, out, n))
    tb.preview(out, "output")
    if do_check:
        d = t * (t.log() - p.log())
        ref = d.mean().view(1)
        return tb.check("kl-loss", out, ref, rtol=1e-2, atol=1e-3)
    return 0


if __name__ == "__main__":
    tb.run("kl-loss", main)
