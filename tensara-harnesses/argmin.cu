#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int dim, int* output, const int* shape, int ndim);

int main() {
    harness::begin("argmin");

    int dim = 1;
    int ndim = 2;

    harness::Buffer<float> input(64 * 64);
    harness::Buffer<int> output(64);
    harness::Buffer<int> shape(ndim);

    input.fill_random();
    shape.set({64, 64});

    BENCHMARK(solution(input, dim, output, shape, ndim));

    output.preview("output");

    printf("Done.\n");
    return 0;
}
