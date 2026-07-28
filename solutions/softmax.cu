// Solution for "softmax": y = exp(x) / sum(exp(x)) over the last (contiguous)
// axis, per row (covers the harness's 2D dim=1 case). Row length must be a
// multiple of 8.
#include "../kernel-implementation/reduction.cuh"

extern "C" void solution(const float* input, int dim, float* output,
                         const size_t* shape, size_t ndim) {
    size_t h_shape[8] = {0};
    size_t nd = ndim < 8 ? ndim : 8;
    cudaMemcpy(h_shape, shape, nd * sizeof(size_t), cudaMemcpyDeviceToHost);

    int D = static_cast<int>(h_shape[dim]);
    size_t rows = 1;
    for (size_t i = 0; i < nd; ++i)
        if (i != static_cast<size_t>(dim)) rows *= h_shape[i];

    launch_reduce<SoftmaxOps, 256>(input, output, static_cast<int>(rows), D);
}
