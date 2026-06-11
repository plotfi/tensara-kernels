#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim);

int main() {
    harness::begin("min-dim");

    int dim = 1;
    size_t ndim = 2;

    harness::Buffer<float> input(64 * 64);
    harness::Buffer<float> output(64);
    harness::Buffer<size_t> shape(ndim);

    input.fill_random();
    shape.set({64, 64});

    BENCHMARK(solution(input, dim, output, shape, ndim));

    output.preview("output");

    printf("Done.\n");
    return 0;
}
