"""Shared harness helpers for Triton kernels — the Python mirror of tensor.cuh.

A solutions-triton/<kernel>.py imports this, defines its @triton.jit kernel and a
solution(), then benchmarks/checks it. It deliberately reuses the C++ system's
conventions so the two backends interoperate:

  * the same size env knobs   — TENSOR_SCALE and TENSOR_<DIM> (k/m/g suffixes)
  * the same output format     — "=== <k> ===" / "Avg kernel time: X ms (over N iters)"
                                 / "<label> (first 10): ..." / "Done."
    so run-bench.sh's timing-table parser picks up Triton runs unchanged.
  * the same warmup/iters      — 3 warmup + 100 timed, CUDA-event timed.
"""
import os
import sys
import torch


# ---- env-controlled sizes: mirror of tensor::bench_size ----------------------
def _parse_size(s):
    s = s.strip().lower()
    mult = 1
    if s and s[-1] in "kmg":
        mult = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}[s[-1]]
        s = s[:-1]
    return int(s) * mult


def _scale():
    e = os.environ.get("TENSOR_SCALE")
    return _parse_size(e) if e else 1


def bench_size(dim, default):
    """TENSOR_<dim> overrides absolutely; else default * TENSOR_SCALE."""
    e = os.environ.get("TENSOR_" + dim)
    if e:
        return _parse_size(e)
    return default * _scale()


# ---- harness banner / rng / preview -----------------------------------------
def begin(name):
    print(f"=== {name} ===")
    torch.manual_seed(42)


def end():
    print("Done.")


def rand(*shape):
    """Uniform [-1, 1] on the GPU, matching the C++ fill_random(float)."""
    return torch.rand(*shape, device="cuda", dtype=torch.float32) * 2.0 - 1.0


def preview(t, label, k=10):
    vals = " ".join(f"{v:.6f}" for v in t.flatten()[:k].tolist())
    print(f"{label} (first {k}): {vals} ")


# ---- timing: mirror of BENCHMARK (3 warmup + 100 timed, CUDA events) --------
def benchmark(fn, warmup=3, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(stop) / iters
    print(f"Avg kernel time: {ms:.4f} ms (over {iters} iters)")


# ---- correctness against a torch reference ----------------------------------
def check(name, got, ref, rtol=1e-3, atol=1e-4):
    ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
    if ok:
        print(f"PASS: {name}")
        return 0
    diff = (got - ref).abs().max().item()
    print(f"FAIL: {name} (max abs diff {diff:g} > atol {atol})")
    return 1


def run(name, main_fn):
    """Standard entry: run main_fn(check=...) with --check honored."""
    do_check = "--check" in sys.argv
    begin(name)
    rc = main_fn(do_check)
    end()
    sys.exit(rc or 0)
