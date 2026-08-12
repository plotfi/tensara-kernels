"""Triton solution for `cs4803dgc-lab1-matmul` (C = A*B, square 32x32).
Triton has no naive-vs-tiled distinction (tl.dot is always tiled), so lab1 and
lab2 are the same Triton kernel — these exist for CUDA-vs-Triton parity. Fixed
size 32 matches the CUDA harness so `run-bench.sh -C` compares like-for-like."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb


@triton.jit
def _mm(a, b, c, N, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a + offs_m[:, None] * N + offs_k[None, :]
    b_ptrs = b + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, N, BK):
        am = (offs_m[:, None] < N) & (offs_k[None, :] + k0 < N)
        bm = (offs_k[:, None] + k0 < N) & (offs_n[None, :] < N)
        acc += tl.dot(tl.load(a_ptrs, mask=am, other=0.0), tl.load(b_ptrs, mask=bm, other=0.0))
        a_ptrs += BK; b_ptrs += BK * N
    cm = (offs_m[:, None] < N) & (offs_n[None, :] < N)
    tl.store(c + offs_m[:, None] * N + offs_n[None, :], acc, mask=cm)


def solution(a, b, c, n):
    BM = BN = BK = 32
    grid = (triton.cdiv(n, BM), triton.cdiv(n, BN))
    _mm[grid](a, b, c, n, BM=BM, BN=BN, BK=BK)


def main(do_check):
    n = 32
    a = tb.rand(n * n); b = tb.rand(n * n); c = torch.empty(n * n, device="cuda")
    tb.benchmark(lambda: solution(a, b, c, n))
    tb.preview(c, "C")
    if do_check:
        return tb.check("cs4803dgc-lab1-matmul", c.view(n, n), a.view(n, n) @ b.view(n, n), rtol=1e-2, atol=1e-2)
    return 0


if __name__ == "__main__":
    tb.run("cs4803dgc-lab1-matmul", main)
