#include "../tensor-lib/tensor.cuh"

// Weight-only int8 group-quantized GEMM (a prologue-fusion example): the int8
// weight is dequantized inside the matmul load, so the fp32 weight is never
// materialized. Wq is [N,K] int8; scale is [N, K/group_size] fp32.
extern "C" void solution(const float* A, const int8_t* Wq, const float* scale,
                         float* C, size_t M, size_t N, size_t K, int group_size);

int main() {
    tensor::begin("int8-weight-gemm");

    size_t M = tensor::bench_size("M", 64);
    size_t N = tensor::bench_size("N", 64);
    size_t K = tensor::bench_size("K", 256);
    int group_size = 64;                 // K/group_size = 4 groups per output row

    tensor::Buffer<float>  A(M * K);
    tensor::Buffer<int8_t> Wq(N * K);
    tensor::Buffer<float>  scale(N * (K / group_size));
    tensor::Buffer<float>  C(M * N);

    A.fill_random();
    Wq.fill_random();
    scale.fill_random();

    BENCHMARK(solution(A, Wq, scale, C, M, N, K, group_size));

    C.preview("C");

    tensor::end();
}
