"""Triton solution for `triplet-margin` — mirrors solutions-cuda/triplet-margin.cu."""
import os, sys, torch, triton
import triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensor-lib"))
import triton_bench as tb

@triton.jit
def _k(a_ptr, p_ptr, n_ptr, loss_ptr, E, margin, inv_B, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < E
    a = tl.load(a_ptr + b * E + offs, mask=mask, other=0.0)
    p = tl.load(p_ptr + b * E + offs, mask=mask, other=0.0)
    nn = tl.load(n_ptr + b * E + offs, mask=mask, other=0.0)
    dp = tl.sum((a - p) * (a - p)); dn = tl.sum((a - nn) * (a - nn))
    v = tl.maximum(0.0, tl.sqrt(dp) - tl.sqrt(dn) + margin)
    tl.atomic_add(loss_ptr, v * inv_B)

def solution(a, p, nn, loss, B, E, margin):
    loss.zero_()
    _k[(B,)](a, p, nn, loss, E, margin, 1.0 / B, BLOCK=triton.next_power_of_2(E))

def main(do_check):
    B = tb.bench_size("B", 8); E = tb.bench_size("E", 128); margin = 1.0
    a = tb.rand(B * E); p = tb.rand(B * E); nn = tb.rand(B * E); loss = torch.zeros(1, device="cuda")
    tb.benchmark(lambda: solution(a, p, nn, loss, B, E, margin))
    tb.preview(loss, "loss")
    if do_check:
        av = a.view(B, E); pv = p.view(B, E); nv = nn.view(B, E)
        d = torch.clamp((av - pv).norm(dim=1) - (av - nv).norm(dim=1) + margin, min=0)
        return tb.check("triplet-margin", loss, d.mean().view(1), rtol=1e-2, atol=1e-3)
    return 0
if __name__ == "__main__": tb.run("triplet-margin", main)
