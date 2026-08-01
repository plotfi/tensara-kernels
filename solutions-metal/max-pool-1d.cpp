// Metal solution wrapper for "max-pool-1d". Mirrors the CUDA solution: dispatch
// one thread per output element over the generalized pool1d<MaxPoolOp> shader.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H) {
    auto pso = tensor::pipeline("max-pool-1d");
    int Hi = static_cast<int>(H);
    int Hout = (Hi + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    tensor::dispatch(pso, { tensor::buf(input), tensor::buf(output) },
                      { tensor::arg(kernel_size), tensor::arg(stride), tensor::arg(padding),
                        tensor::arg(dilation), tensor::arg(Hi) },
                      static_cast<size_t>(Hout));
}
