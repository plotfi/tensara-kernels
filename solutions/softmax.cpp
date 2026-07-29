// Metal solution wrapper for "softmax". Reduces over the last (contiguous) axis.
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim) {
    auto pso = harness::pipeline("softmax");

    // Same as the CUDA solution: get a host-readable view of the dimensions to
    // size the launch. Here harness::to_host is a no-op returning `shape`
    // (Metal storage is shared); on CUDA it's a device->host copy -- the call
    // site is identical either way.
    size_t scratch[8] = {0};
    size_t nd = ndim < 8 ? ndim : 8;
    const size_t* s = harness::to_host(scratch, shape, nd);

    size_t D = s[dim];
    size_t rows = 1;
    for (size_t i = 0; i < nd; ++i)
        if (i != static_cast<size_t>(dim)) rows *= s[i];

    uint32_t Dv = static_cast<uint32_t>(D);
    const size_t tpg = 256;
    harness::dispatch(pso, { harness::buf(input), harness::buf(output) }, { harness::arg(Dv) }, rows * tpg, tpg);
}
