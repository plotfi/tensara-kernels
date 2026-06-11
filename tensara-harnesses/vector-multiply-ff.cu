#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const uint32_t* d_input1, const uint32_t* d_input2, uint32_t* d_output, size_t n);

int main() {
    harness::begin("vector-multiply-ff");

    size_t n = 1024;

    harness::Buffer<uint32_t> d_input1(n);
    harness::Buffer<uint32_t> d_input2(n);
    harness::Buffer<uint32_t> d_output(n);

    d_input1.fill_random();
    d_input2.fill_random();

    BENCHMARK(solution(d_input1, d_input2, d_output, n));

    d_output.preview("d_output");

    printf("Done.\n");
    return 0;
}
