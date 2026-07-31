// Metal solution wrapper stub for "ecc-point-negation" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const uint64_t* xs, const uint64_t* ys, uint64_t p, uint64_t* out_xy, size_t n) {
    auto pso = harness::pipeline("ecc-point-negation");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
