"""Triton solution for `gemm-relu` — mirrors solutions-cuda/gemm-relu.cu."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb

@triton.jit
def _mm(a, wt, bias, c, Bn, M, N, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)          # rows over B, cols over M(out)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptrs = a + rm[:, None] * N + rk[None, :]            # A[b, n]
    w_ptrs = wt + rn[None, :] * N + rk[:, None]           # W[j, n] as (k=n, col=j)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, N, BK):
        av = tl.load(a_ptrs, mask=(rm[:, None] < Bn) & (rk[None, :] + k0 < N), other=0.0)
        wv = tl.load(w_ptrs, mask=(rk[:, None] + k0 < N) & (rn[None, :] < M), other=0.0)
        acc += tl.dot(av, wv)
        a_ptrs += BK; w_ptrs += BK
    bv = tl.load(bias + rn, mask=rn < M, other=0.0)
    out = tl.maximum(acc + bv[None, :], 0.0)
    tl.store(c + rm[:, None] * M + rn[None, :], out, mask=(rm[:, None] < Bn) & (rn[None, :] < M))


def solution(a, w, b, c, Bn, N, M):
    grid = (triton.cdiv(Bn, 64), triton.cdiv(M, 64))
    _mm[grid](a, w, b, c, Bn, M, N, BM=64, BN=64, BK=32)


def main(do_check):
    Bn = tb.bench_size("B", 16); N = tb.bench_size("N", 64); M = tb.bench_size("M", 64)
    a = tb.rand(Bn * N); w = tb.rand(M * N); b = tb.rand(M); c = torch.empty(Bn * M, device="cuda")
    tb.benchmark(lambda: solution(a, w, b, c, Bn, N, M))
    tb.preview(c, "C")
    if do_check:
        ref = torch.relu(a.view(Bn, N) @ w.view(M, N).t() + b)
        return tb.check("gemm-relu", c.view(Bn, M), ref, rtol=1e-2, atol=1e-2)
    return 0

if __name__ == "__main__":
    tb.run("gemm-relu", main)
