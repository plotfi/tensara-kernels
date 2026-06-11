#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const int* a, int* b, size_t n);

int main(int argc, char** argv) {
    printf("=== array-sort ===\n");
    srand(42);

    size_t n = 1024;

    size_t a_count = n;
    size_t b_count = n;

    int* h_a = new int[a_count];
    int* h_b = new int[b_count];

    for (size_t i = 0; i < a_count; i++)
        h_a[i] = rand() % 201 - 100;
    memset(h_b, 0, b_count * sizeof(int));

    int* d_a;
    cudaMalloc(&d_a, a_count * sizeof(int));
    int* d_b;
    cudaMalloc(&d_b, b_count * sizeof(int));

    cudaMemcpy(d_a, h_a, a_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, b_count * sizeof(int));

    BENCHMARK(solution(d_a, d_b, n));

    cudaMemcpy(h_b, d_b, b_count * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Output b (first 10): ");
    for (size_t i = 0; i < 10 && i < b_count; i++)
        printf("%d ", h_b[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;

    printf("Done.\n");
    return 0;
}
