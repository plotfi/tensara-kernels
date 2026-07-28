#include <cuda_runtime.h>

// Softmax over the last (contiguous) axis: y = exp(x) / sum(exp(x)), per row.
// Same reduction shape as log-softmax, but the writeback divides by the sum of
// exponentials instead of subtracting its log.
#define MAP_OP(X)         __expf(X)
#define FINALIZE(ACC, D)  (ACC)
#define WRITE_OP(X, ACC)  __fdividef(__expf(X), ACC)
#include "../kernel-implementation/reduction.cuh"

// Note: input, output, and shape are device pointers.
// Reduces over `dim`, treated as the innermost (contiguous) axis — this covers
// the harness's 2D dim=1 case. Row length shape[dim] must be a multiple of 4.
extern "C" void solution(const float* input, int dim, float* output,
                         const size_t* shape, size_t ndim) {
    size_t h_shape[8] = {0};
    size_t nd = ndim < 8 ? ndim : 8;
    cudaMemcpy(h_shape, shape, nd * sizeof(size_t), cudaMemcpyDeviceToHost);

    int D = static_cast<int>(h_shape[dim]);
    size_t rows = 1;
    for (size_t i = 0; i < nd; ++i)
        if (i != static_cast<size_t>(dim)) rows *= h_shape[i];

    launch_reduction(input, output, static_cast<int>(rows), D);
}
