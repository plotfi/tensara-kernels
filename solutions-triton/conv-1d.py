"""Triton solution for `conv-1d` — mirrors solutions-cuda/conv-1d.cu."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb

@triton.jit
def _kernel(a, b, c, N, K, RC, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for j in range(0, K):
        idx = i + j - RC
        av = tl.load(a + idx, mask=(idx >= 0) & (idx < N) & (i < N), other=0.0)
        acc += av * tl.load(b + j)
    tl.store(c + i, acc, mask=i < N)


def solution(a, b, c, N, K):
    BLOCK = 1024
    _kernel[(triton.cdiv(N, BLOCK),)](a, b, c, N, K, (K - 1) // 2, BLOCK=BLOCK)


def main(do_check):
    N = tb.bench_size("N", 1024); K = 5
    a = tb.rand(N); b = tb.rand(K); c = torch.empty(N, device="cuda")
    tb.benchmark(lambda: solution(a, b, c, N, K))
    tb.preview(c, "C")
    if do_check:
        ref = torch.nn.functional.conv1d(a.view(1, 1, N), b.view(1, 1, K), padding=(K - 1) // 2).view(N)
        return tb.check("conv-1d", c, ref, rtol=1e-3, atol=1e-3)
    return 0

if __name__ == "__main__":
    tb.run("conv-1d", main)
