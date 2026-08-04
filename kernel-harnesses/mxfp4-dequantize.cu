#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const uint8_t* q, const uint8_t* scale, float* out, size_t m, size_t k);

int main() {
    tensor::begin("mxfp4-dequantize");

    size_t m = tensor::bench_size("M", 64);
    size_t k = tensor::bench_size("K", 64);

    tensor::Buffer<uint8_t> q(m * k / 2);
    tensor::Buffer<uint8_t> scale(m * (k / 32));
    tensor::Buffer<float> out(m * k);

    q.fill_random();
    scale.fill_random();

    BENCHMARK(solution(q, scale, out, m, k));

    out.preview("out");

    tensor::end();
}
