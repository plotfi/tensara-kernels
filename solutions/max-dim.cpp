// Metal solution wrapper stub for "max-dim" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim) {
    auto pso = tensor::pipeline("max-dim");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
