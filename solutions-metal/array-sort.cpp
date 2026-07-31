// Metal solution wrapper stub for "array-sort" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const int* a, int* b, size_t n) {
    auto pso = tensor::pipeline("array-sort");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
