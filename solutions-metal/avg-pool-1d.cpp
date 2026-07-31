// Metal solution wrapper for "avg-pool-1d".
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    auto pso = tensor::pipeline("avg-pool-1d");
    int Hi = static_cast<int>(H);
    int Hout = (Hi + 2 * padding - kernel_size) / stride + 1;
    tensor::dispatch(pso, { tensor::buf(input), tensor::buf(output) },
                      { tensor::arg(kernel_size), tensor::arg(stride), tensor::arg(padding), tensor::arg(Hi) },
                      static_cast<size_t>(Hout));
}
