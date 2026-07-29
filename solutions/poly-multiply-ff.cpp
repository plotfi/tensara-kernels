// Metal solution wrapper stub for "poly-multiply-ff" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (harness::buf(ptr)) and scalars (harness::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const uint32_t* d_input1, const uint32_t* d_input2, uint32_t* d_output, size_t n) {
    auto pso = harness::pipeline("poly-multiply-ff");
    harness::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
