"""Triton solution for `conv1d-maxpool1d` — mirrors solutions-cuda/conv1d-maxpool1d.cu."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb

@triton.jit
def _kernel(inp, wt, out, N, K, RC, Lout, KS, STRIDE, PAD, DIL, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), float("-inf"), tl.float32)
    for m in range(0, KS):
        cpos = o * STRIDE - PAD + m * DIL              # conv index this pool tap reads
        valid = (cpos >= 0) & (cpos < N)
        conv = tl.zeros((BLOCK,), tl.float32)
        for j in range(0, K):
            idx = cpos + j - RC
            av = tl.load(inp + idx, mask=(idx >= 0) & (idx < N) & valid & (o < Lout), other=0.0)
            conv += av * tl.load(wt + j)
        acc = tl.maximum(acc, tl.where(valid, conv, float("-inf")))
    tl.store(out + o, acc, mask=o < Lout)


def solution(inp, wt, out, N, K, ks, stride, pad, dil, Lout):
    _kernel[(triton.cdiv(Lout, 256),)](inp, wt, out, N, K, (K - 1) // 2, Lout, ks, stride, pad, dil, BLOCK=256)


def main(do_check):
    N = tb.bench_size("N", 1024); K = 5
    ks, stride, pad, dil = 3, 2, 1, 1
    Lout = (N + 2 * pad - dil * (ks - 1) - 1) // stride + 1
    inp = tb.rand(N); wt = tb.rand(K); out = torch.empty(Lout, device="cuda")
    tb.benchmark(lambda: solution(inp, wt, out, N, K, ks, stride, pad, dil, Lout))
    tb.preview(out, "output")
    if do_check:
        conv = torch.nn.functional.conv1d(inp.view(1, 1, N), wt.view(1, 1, K), padding=(K - 1) // 2)
        ref = torch.nn.functional.max_pool1d(conv, ks, stride, pad, dil).view(Lout)
        return tb.check("conv1d-maxpool1d", out, ref, rtol=1e-3, atol=1e-3)
    return 0

if __name__ == "__main__":
    tb.run("conv1d-maxpool1d", main)
