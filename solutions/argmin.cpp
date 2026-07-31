// Metal solution wrapper stub for "argmin" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int dim, int* output, const int* shape, int ndim) {
    auto pso = harness::pipeline("argmin");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
