// Metal solution wrapper stub for "argmax" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, int dim, int* output, const int* shape, int ndim) {
    auto pso = tensor::pipeline("argmax");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
