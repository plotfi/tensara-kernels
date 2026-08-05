"""Triton solution for `grayscale` — mirrors solutions-cuda/grayscale.cu."""
import os
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb


@triton.jit
def _kernel(rgb_ptr, gray_ptr, npix, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < npix
    r = tl.load(rgb_ptr + 3 * p + 0, mask=mask)
    g = tl.load(rgb_ptr + 3 * p + 1, mask=mask)
    b = tl.load(rgb_ptr + 3 * p + 2, mask=mask)
    tl.store(gray_ptr + p, 0.299 * r + 0.587 * g + 0.114 * b, mask=mask)


def solution(rgb, gray, npix):
    BLOCK = 1024
    _kernel[(triton.cdiv(npix, BLOCK),)](rgb, gray, npix, BLOCK=BLOCK)


def main(do_check):
    h = tb.bench_size("HEIGHT", 64)
    w = tb.bench_size("WIDTH", 64)
    npix = h * w
    rgb = tb.rand(npix * 3)
    gray = torch.empty(npix, device="cuda")
    tb.benchmark(lambda: solution(rgb, gray, npix))
    tb.preview(gray, "grayscale_output")
    if do_check:
        wts = torch.tensor([0.299, 0.587, 0.114], device="cuda")
        return tb.check("grayscale", gray, rgb.view(npix, 3) @ wts)
    return 0


if __name__ == "__main__":
    tb.run("grayscale", main)
