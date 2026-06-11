#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw);

int main() {
    harness::begin("conv-2d");

    size_t H = 64;
    size_t W = 64;
    size_t Kh = 3;
    size_t Kw = 3;

    harness::Buffer<float> A(H * W);
    harness::Buffer<float> B(Kh * Kw);
    harness::Buffer<float> C(H * W);

    A.fill_random();
    B.fill_random();

    BENCHMARK(solution(A, B, C, H, W, Kh, Kw));

    C.preview("C");

    harness::end();
}
