#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n);

int main() {
    harness::begin("vector-addition");

    size_t n = 1024;

    harness::Buffer<float> d_input1(n);
    harness::Buffer<float> d_input2(n);
    harness::Buffer<float> d_output(n);

    d_input1.fill_random();
    d_input2.fill_random();

    BENCHMARK(solution(d_input1, d_input2, d_output, n));

    d_output.preview("d_output");

    harness::end();
}
