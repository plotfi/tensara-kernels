#pragma once

// Cross-platform launch/benchmark helpers for the Tensara harnesses.
//
// Every harness does the same thing: seed RNG, allocate GPU buffers, fill
// inputs, run the kernel under a timed loop, then preview the outputs. The
// device-specific allocation is hidden behind harness::Buffer<T> so one harness
// can target either CUDA or Metal. (Design borrowed from micro-tensor, folded
// in here so the harness has no external dependency.)
//
// Backend selection (compile-time):
//   * Define HARNESS_CUDA to force the CUDA backend, or
//   * Define HARNESS_METAL to force the Metal backend (via metal-cpp), or
//   * Otherwise auto-detect: __CUDACC__ -> CUDA, __APPLE__ -> Metal,
//     defaulting to CUDA.
//
// Common API on both backends:
//   harness::begin(name) / harness::end()
//   harness::Buffer<T>(n)   -- alloc + zero; .fill_random() .set({...}) .fill(v)
//                              .preview(label) .data() .size()
//   BENCHMARK(...)          -- warm up, time, print avg per launch
//
// Launch handoff (the one place the backends differ):
//   CUDA  -- Buffer converts to T* (device pointer); call a kernel launch:
//              BENCHMARK(solution(a, b, c, n));
//   Metal -- Buffer converts to MTL::Buffer*; dispatch a compiled shader:
//              auto pso = harness::pipeline("<kernel>");
//              BENCHMARK(harness::dispatch(pso, {a, b, c}, { harness::arg(n) }, grid));
//   Harnesses select between the two with #if HARNESS_METAL.

#if !defined(HARNESS_CUDA) && !defined(HARNESS_METAL)
#  if defined(__CUDACC__)
#    define HARNESS_CUDA 1
#  elif defined(__APPLE__)
#    define HARNESS_METAL 1
#  else
#    define HARNESS_CUDA 1   // tensara is CUDA-first; default to it
#  endif
#endif

#if defined(HARNESS_METAL)
#  include "detail/harness_metal.cuh"
#else
#  include "detail/harness_cuda.cuh"
#endif

// Benchmark a call expression, e.g. BENCHMARK(solution(a, b, c, m, n, k));
// The expression is captured by reference and replayed each iteration.
#define BENCHMARK(...) ::harness::benchmark([&]() { __VA_ARGS__; })
