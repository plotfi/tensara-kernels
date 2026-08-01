#pragma once

// Cross-platform launch/benchmark helpers for the kernel harnesses.
//
// Every harness does the same thing: seed RNG, allocate GPU buffers, fill
// inputs, run the kernel under a timed loop, then preview the outputs. The
// device-specific allocation is hidden behind tensor::Buffer<T> so one harness
// can target either CUDA or Metal.
//
// Backend selection (compile-time):
//   * Define TENSOR_CUDA to force the CUDA backend, or
//   * Define TENSOR_METAL to force the Metal backend (via metal-cpp), or
//   * Otherwise auto-detect: __CUDACC__ -> CUDA, __APPLE__ -> Metal,
//     defaulting to CUDA.
//
// Common API on both backends:
//   tensor::begin(name) / tensor::end()
//   tensor::Buffer<T>(n)   -- alloc + zero; .fill_random() .set({...}) .fill(v)
//                              .preview(label) .data() .size()
//   BENCHMARK(...)          -- warm up, time, print avg per launch
//
// Launch handoff (the one place the backends differ):
//   CUDA  -- Buffer converts to T* (device pointer); call a kernel launch:
//              BENCHMARK(solution(a, b, c, n));
//   Metal -- Buffer converts to MTL::Buffer*; dispatch a compiled shader:
//              auto pso = tensor::pipeline("<kernel>");
//              BENCHMARK(tensor::dispatch(pso, {a, b, c}, { tensor::arg(n) }, grid));
//   Harnesses select between the two with #if TENSOR_METAL.

#if !defined(TENSOR_CUDA) && !defined(TENSOR_METAL)
#  if defined(__CUDACC__)
#    define TENSOR_CUDA 1
#  elif defined(__APPLE__)
#    define TENSOR_METAL 1
#  else
#    define TENSOR_CUDA 1   // CUDA-first; default to it
#  endif
#endif

#if defined(TENSOR_METAL)
#  include "detail/tensor_metal.cuh"
#else
#  include "detail/tensor_cuda.cuh"
#endif

// Benchmark a call expression, e.g. BENCHMARK(solution(a, b, c, m, n, k));
// The expression is captured by reference and replayed each iteration.
#define BENCHMARK(...) ::tensor::benchmark([&]() { __VA_ARGS__; })
