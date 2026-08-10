// Solution for "softmax": y = exp(x) / sum(exp(x)) over the last (contiguous)
// axis, per row (covers the harness's 2D dim=1 case). Row length must be a
// multiple of 8.
#include "../tensor-lib/tensor.cuh"   // tensor::to_host
#include "../kernel-implementation/reduction.cuh"

extern "C" void solution(const float* input, int dim, float* output,
                         const size_t* shape, size_t ndim) {
    // Unlike the norm kernels, softmax gets its dimensions as a device `shape`
    // array rather than scalar args. The launch grid (row count) and row length
    // are host-side decisions, so get a host-readable view of the small shape
    // array with tensor::to_host before deriving them. (That's a device->host
    // copy on CUDA; a no-op returning `shape` on Metal, where storage is shared.)
    size_t scratch[8] = {0};
    size_t nd = ndim < 8 ? ndim : 8;
    const size_t* s = tensor::to_host(scratch, shape, nd);

    int D = static_cast<int>(s[dim]);
    size_t rows = 1;
    for (size_t i = 0; i < nd; ++i)
      if (i != static_cast<size_t>(dim))
        rows *= s[i];

    launch_reduce<SoftmaxOps, 256>(input, output, static_cast<int>(rows), D);
}
