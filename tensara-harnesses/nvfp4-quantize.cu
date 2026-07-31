#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const half* a, float sf_g, uint8_t* q, __nv_fp8_e4m3* scale, size_t m, size_t k);

int main() {
    tensor::begin("nvfp4-quantize");

    float sf_g = 1.0f;
    size_t m = 64;
    size_t k = 64;

    tensor::Buffer<half> a(m * k);
    tensor::Buffer<uint8_t> q(m * k / 2);
    tensor::Buffer<__nv_fp8_e4m3> scale(m * (k / 16));

    a.fill_random();

    BENCHMARK(solution(a, sf_g, q, scale, m, k));

    q.preview("q");
    scale.preview("scale");

    tensor::end();
}
