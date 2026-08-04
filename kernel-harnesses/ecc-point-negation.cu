#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const uint64_t* xs, const uint64_t* ys, uint64_t p, uint64_t* out_xy, size_t n);

int main() {
    tensor::begin("ecc-point-negation");

    uint64_t p = 18446744073709551557ULL;
    size_t n = tensor::bench_size("N", 1024);

    tensor::Buffer<uint64_t> xs(n);
    tensor::Buffer<uint64_t> ys(n);
    tensor::Buffer<uint64_t> out_xy(n * 2);

    xs.fill_random();
    ys.fill_random();

    BENCHMARK(solution(xs, ys, p, out_xy, n));

    out_xy.preview("out_xy");

    tensor::end();
}
