#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#include <math_constants.h>

// ReLU
#define BLOCK_SIZE 512
#define ACTIVATION ReLU
#define SOLUTION() multi_solution(input, output, n, m, 0.0f)

// LeakyReLU

// ELU

// GELU

#define ELU(X) \
    X = (X > 0.0f) ? X : alpha * (__expf(X) - 1)

#define ReLU(v) fmax(v, 0.0f)

#if 0
#define kSqrt2OverPi sqrtf(2.0f / M_PI)
#else
#define kSqrt2OverPi 0.7978845608028654f
#endif
#define kCoef 0.044715f
#define GELU(x) \
    ((0.5f * x) * \
     (1.0f + fast_tanh(kSqrt2OverPi * (x + (kCoef * (x * x * x))))))

__device__ __forceinline__ float fast_tanh(float x) {
    float e2x = __expf(2.0f * x);
    return __fdividef(e2x - 1.0f, e2x + 1.0f);
}

__global__ void activation_kernelx8(const float* __restrict__ A,
                                   float* __restrict__ C,
                                   int n,
                                   float alpha) {
    int base =  (threadIdx.x + blockIdx.x * blockDim.x) * 8;

    if (base + 7 >= n) {
        for (int j = base; j < n; ++j) {
            const float x = A[j];
            C[j] = ReLU(x);
        }
        return;
    }

    // Apparently the hardware can issue 256 bits at a time, so do 2 float4s
    float4 x0 = *reinterpret_cast<const float4*>(A + base);
    float4 x1 = *reinterpret_cast<const float4*>(A + base + 4);

    x0.x = ACTIVATION(x0.x); x0.y = ACTIVATION(x0.y);
    x0.z = ACTIVATION(x0.z); x0.w = ACTIVATION(x0.w);
    x1.x = ACTIVATION(x1.x); x1.y = ACTIVATION(x1.y);
    x1.z = ACTIVATION(x1.z); x1.w = ACTIVATION(x1.w);

    *reinterpret_cast<float4*>(C + base    ) = x0;
    *reinterpret_cast<float4*>(C + base + 4) = x1;
}

void multi_solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    int N = static_cast<int>(n * m);
    if (N == 0) return;

    // Each thread does 8 elements
    size_t threads_needed = (static_cast<int>(N) + 7) / 8;
    const int grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;

    activation_kernelx8<<<grid, BLOCK_SIZE>>>(input, output, N, alpha);
}

/*
extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
  SOLUTION();
}
*/

// Note: input, output are device pointers
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
  SOLUTION();
}


int main(int argc, char** argv) {
    printf("=== relu ===\n");
    srand(42);

    size_t n = 64;
    size_t m = 64;

    size_t input_count = n * m;
    size_t output_count = n * m;

    float* h_input = new float[input_count];
    float* h_output = new float[output_count];

    for (size_t i = 0; i < input_count; i++)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(float));

    float* d_input;
    cudaMalloc(&d_input, input_count * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, output_count * sizeof(float));

    cudaMemcpy(d_input, h_input, input_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(float));

    solution(d_input, d_output, n, m);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output (first 10): ");
    for (size_t i = 0; i < 10 && i < output_count; i++)
        printf("%f ", h_output[i]);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}
