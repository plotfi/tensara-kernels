// Metal solution wrapper stub for "vector-multiply-ff" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const uint32_t* d_input1, const uint32_t* d_input2, uint32_t* d_output, size_t n) {
    auto pso = tensor::pipeline("vector-multiply-ff");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
