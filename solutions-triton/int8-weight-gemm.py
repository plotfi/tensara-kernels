"""Triton solution for `int8-weight-gemm` — mirrors solutions-cuda/int8-weight-gemm.cu."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb

@triton.jit
def _mm(a, wq, scale, c, M, N, K, NG, BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, G)
    a_ptrs = a + rm[:, None] * K + rk[None, :]           # A[m, k]
    w_ptrs = wq + rn[None, :] * K + rk[:, None]          # Wq[n, k] as (k, col=n), int8
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, G):                            # one group per BK step (BK == G)
        g = k0 // G
        av = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
        wv = tl.load(w_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0)
        sc = tl.load(scale + rn * NG + g, mask=rn < N, other=0.0)   # scale[n, g]
        deq = wv.to(tl.float32) * sc[None, :]
        acc += tl.dot(av, deq)
        a_ptrs += G; w_ptrs += G
    tl.store(c + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def solution(a, wq, scale, c, M, N, K, group_size):
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _mm[grid](a, wq, scale, c, M, N, K, K // group_size, BM=64, BN=64, G=group_size)


def main(do_check):
    M = tb.bench_size("M", 64); N = tb.bench_size("N", 64); K = tb.bench_size("K", 256)
    G = 64; NG = K // G
    a = tb.rand(M * K)
    wq = torch.randint(-128, 128, (N * K,), dtype=torch.int8, device="cuda")
    scale = tb.rand(N * NG).abs() * 0.05 + 0.01
    c = torch.empty(M * N, device="cuda")
    tb.benchmark(lambda: solution(a, wq, scale, c, M, N, K, G))
    tb.preview(c, "C")
    if do_check:
        w = wq.view(N, K).float() * scale.view(N, NG).repeat_interleave(G, dim=1)
        ref = a.view(M, K) @ w.t()
        return tb.check("int8-weight-gemm", c.view(M, N), ref, rtol=1e-2, atol=1e-1)
    return 0

if __name__ == "__main__":
    tb.run("int8-weight-gemm", main)
