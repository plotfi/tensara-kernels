"""Triton solution for `max-pool-1d` — mirrors solutions-cuda/max-pool-1d.cu."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb

@triton.jit
def _kernel(x, y, H, Lout, KS, STRIDE, PAD, DIL, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), float("-inf"), tl.float32)
    for m in range(0, KS):
        idx = o * STRIDE - PAD + m * DIL
        v = tl.load(x + idx, mask=(idx >= 0) & (idx < H) & (o < Lout), other=float("-inf"))
        acc = tl.maximum(acc, v)
    tl.store(y + o, acc, mask=o < Lout)


def solution(x, y, ks, stride, pad, dil, H, Lout):
    _kernel[(triton.cdiv(Lout, 256),)](x, y, H, Lout, ks, stride, pad, dil, BLOCK=256)


def main(do_check):
    H = tb.bench_size("H", 1024); ks, stride, pad, dil = 3, 1, 1, 1
    Lout = (H + 2 * pad - dil * (ks - 1) - 1) // stride + 1
    x = tb.rand(H); y = torch.empty(Lout, device="cuda")
    tb.benchmark(lambda: solution(x, y, ks, stride, pad, dil, H, Lout))
    tb.preview(y, "output")
    if do_check:
        ref = torch.nn.functional.max_pool1d(x.view(1, 1, H), ks, stride, pad, dil).view(Lout)
        return tb.check("max-pool-1d", y, ref, rtol=1e-3, atol=1e-3)
    return 0

if __name__ == "__main__":
    tb.run("max-pool-1d", main)
