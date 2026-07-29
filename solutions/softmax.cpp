// Metal solution wrapper for "softmax". Reduces over the last (contiguous) axis;
// reads the shape buffer (shared memory) to get row length D and row count.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim) {
    auto pso = harness::pipeline("softmax");
    size_t D = shape[dim];
    size_t rows = 1;
    for (size_t i = 0; i < ndim; ++i)
        if (i != static_cast<size_t>(dim)) rows *= shape[i];
    uint32_t Dv = static_cast<uint32_t>(D);
    const size_t tpg = 256;
    harness::dispatch(pso, { harness::buf(input), harness::buf(output) }, { harness::arg(Dv) }, rows * tpg, tpg);
}
