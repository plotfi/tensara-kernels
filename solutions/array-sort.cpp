// Metal solution wrapper stub for "array-sort" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const int* a, int* b, size_t n) {
    auto pso = harness::pipeline("array-sort");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
