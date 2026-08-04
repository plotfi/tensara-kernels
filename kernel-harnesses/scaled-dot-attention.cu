#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* Q, const float* K, const float* V, float* output, size_t B, size_t H, size_t S, size_t E);

int main() {
    tensor::begin("scaled-dot-attention");

    size_t B = tensor::bench_size("B", 2);
    size_t H = tensor::bench_size("H", 4);
    size_t S = tensor::bench_size("S", 32);
    size_t E = tensor::bench_size("E", 64);

    tensor::Buffer<float> Q(B * H * S * E);
    tensor::Buffer<float> K(B * H * S * E);
    tensor::Buffer<float> V(B * H * S * E);
    tensor::Buffer<float> output(B * H * S * E);

    Q.fill_random();
    K.fill_random();
    V.fill_random();

    BENCHMARK(solution(Q, K, V, output, B, H, S, E));

    output.preview("output");

    tensor::end();
}
