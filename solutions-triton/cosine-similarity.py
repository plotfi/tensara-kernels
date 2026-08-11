"""Triton solution for `cosine-similarity` — mirrors solutions-cuda/cosine-similarity.cu."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb

@triton.jit
def _k(p_ptr, t_ptr, out_ptr, d, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < d
    p = tl.load(p_ptr + row * d + offs, mask=mask, other=0.0)
    t = tl.load(t_ptr + row * d + offs, mask=mask, other=0.0)
    dot = tl.sum(p * t); npp = tl.sum(p * p); ntt = tl.sum(t * t)
    tl.store(out_ptr + row, dot / (tl.sqrt(npp) * tl.sqrt(ntt)))

def solution(p, t, out, n, d):
    _k[(n,)](p, t, out, d, BLOCK=triton.next_power_of_2(d))

def main(do_check):
    n = tb.bench_size("N", 64); d = tb.bench_size("D", 128)
    p = tb.rand(n * d); t = tb.rand(n * d); out = torch.empty(n, device="cuda")
    tb.benchmark(lambda: solution(p, t, out, n, d))
    tb.preview(out, "output")
    if do_check:
        pv = p.view(n, d); tv = t.view(n, d)
        ref = (pv * tv).sum(1) / (pv.norm(dim=1) * tv.norm(dim=1))
        return tb.check("cosine-similarity", out, ref, rtol=1e-3, atol=1e-4)
    return 0
if __name__ == "__main__": tb.run("cosine-similarity", main)
