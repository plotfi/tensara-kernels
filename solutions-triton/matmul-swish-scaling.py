"""Triton solution for `matmul-swish-scaling` — mirrors solutions-cuda/matmul-swish-scaling.cu."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb

@triton.jit
def _mm(a, b, c, M, N, K, scale, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptrs = a + rm[:, None] * K + rk[None, :]
    b_ptrs = b + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        av = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
        bv = tl.load(b_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(av, bv)
        a_ptrs += BK; b_ptrs += BK * N
    out = acc * tl.sigmoid(acc) * scale
    tl.store(c + rm[:, None] * N + rn[None, :], out, mask=(rm[:, None] < M) & (rn[None, :] < N))


def solution(a, b, c, M, N, K, scale=1.0):
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _mm[grid](a, b, c, M, N, K, scale, BM=64, BN=64, BK=32)


def main(do_check):
    M = tb.bench_size("M", 64); N = tb.bench_size("N", 64); K = tb.bench_size("K", 64)
    scale = 1.0
    a = tb.rand(M * K); b = tb.rand(K * N); c = torch.empty(M * N, device="cuda")
    tb.benchmark(lambda: solution(a, b, c, M, N, K, scale))
    tb.preview(c, "output")
    if do_check:
        ref = torch.nn.functional.silu(a.view(M, K) @ b.view(K, N)) * scale
        return tb.check("matmul-swish-scaling", c.view(M, N), ref, rtol=1e-2, atol=1e-2)
    return 0

if __name__ == "__main__":
    tb.run("matmul-swish-scaling", main)
