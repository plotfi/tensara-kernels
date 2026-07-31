#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const uint8_t* q_a, const uint8_t* scale_a, const uint8_t* q_b, const uint8_t* scale_b, float* c, size_t m, size_t n, size_t k);

int main() {
    tensor::begin("mxfp4-gemm");

    size_t m = 64;
    size_t n = 64;
    size_t k = 64;

    tensor::Buffer<uint8_t> q_a(m * k / 2);
    tensor::Buffer<uint8_t> scale_a(m * (k / 32));
    tensor::Buffer<uint8_t> q_b(k * n / 2);
    tensor::Buffer<uint8_t> scale_b(k * (n / 32));
    tensor::Buffer<float> c(m * n);

    q_a.fill_random();
    scale_a.fill_random();
    q_b.fill_random();
    scale_b.fill_random();

    BENCHMARK(solution(q_a, scale_a, q_b, scale_b, c, m, n, k));

    c.preview("c");

    tensor::end();
}
