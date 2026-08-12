"""Triton solution for `cs4803dgc-lab3-conv2d-shared` (2D 5x5 convolution, 512x512, zero-padded).
Triton manages shared/constant memory automatically, so the CUDA lab3 variants
(global / shared / shared-constant) collapse to one Triton kernel — these three
files are identical and exist for CUDA-vs-Triton parity. Fixed size 512 matches
the CUDA harness."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb


@triton.jit
def _conv(f_ptr, n_ptr, p_ptr, sz, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for m in range(5):
        for n in range(5):
            ri = rows[:, None] + m - 2
            ci = cols[None, :] + n - 2
            mask = (ri >= 0) & (ri < sz) & (ci >= 0) & (ci < sz)
            val = tl.load(n_ptr + ri * sz + ci, mask=mask, other=0.0)
            acc += val * tl.load(f_ptr + m * 5 + n)   # filter weight (scalar)
    om = (rows[:, None] < sz) & (cols[None, :] < sz)
    tl.store(p_ptr + rows[:, None] * sz + cols[None, :], acc, mask=om)


def solution(filter, N, P, size):
    BM = BN = 16
    grid = (triton.cdiv(size, BM), triton.cdiv(size, BN))
    _conv[grid](filter, N, P, size, BM=BM, BN=BN)


def main(do_check):
    sz = 512
    filt = tb.rand(25); N = tb.rand(sz * sz); P = torch.empty(sz * sz, device="cuda")
    tb.benchmark(lambda: solution(filt, N, P, sz))
    tb.preview(P, "P")
    if do_check:
        ref = torch.nn.functional.conv2d(N.view(1,1,sz,sz), filt.view(1,1,5,5), padding=2).view(sz, sz)
        return tb.check("cs4803dgc-lab3-conv2d-shared", P.view(sz, sz), ref, rtol=1e-2, atol=1e-2)
    return 0


if __name__ == "__main__":
    tb.run("cs4803dgc-lab3-conv2d-shared", main)
