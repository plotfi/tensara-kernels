#include <cuda_runtime.h>
#include <math_constants.h>

// Uncomment for ReLU
// #define BLOCK_SIZE 512
// #define ACTIVATION ReLU
// #define SOLUTION() multi_solution(input, output, n, m, 0.0f)

// Uncomment for LeakyReLU
// #define BLOCK_SIZE 512
// #define ACTIVATION LeakyReLU
// #define SOLUTION() multi_solution(input, output, n, m, alpha)

// Uncomment for SELU
// #define BLOCK_SIZE 1024
// #define ACTIVATION SELU
// #define SOLUTION() multi_solution(input, output, n, m, 0.0f)

// Uncomment for Softplus
// #define BLOCK_SIZE 1024
// #define ACTIVATION SOFTPLUS
// #define SOLUTION() multi_solution(input, output, n, m, 0.0f)

// Uncomment for Swish
#define BLOCK_SIZE 1024
#define ACTIVATION SWISH
#define SOLUTION() multi_solution(input, output, n, m, 0.0f)

// Uncomment for ELU
// #define BLOCK_SIZE 256
// #define ACTIVATION ELU
// #define SOLUTION() multi_solution(input, output, n, m, alpha)

// Uncomment for GELU
// #define BLOCK_SIZE 512
// #define ACTIVATION GELU
// #define SOLUTION() multi_solution(input, output, n, m, 0.0f)

// Uncomment for sigmoid
// #define BLOCK_SIZE 512
// #define ACTIVATION SIGMOID
// #define SOLUTION() multi_solution(input, output, n, m, 0.0f)

// Uncomment for tanh
// #define BLOCK_SIZE 512
// #define ACTIVATION fast_tanh
// #define SOLUTION() multi_solution(input, output, n, m, 0.0f)

#define SOFTPLUS(X) \
    __logf(1 + __expf(X))

#define SELU(X) \
    1.0507f * (fmax(0.0f, X) + fmin(0.0f, 1.67326f * (__expf(X) - 1)))

#define ELU(X) \
    X > 0.0f ? X : (alpha * (__expf(X) - 1))

#define SIGMOID(X) __fdividef(1.0f, 1.0f + __expf(-(X)))

#define SWISH(X) \
  (X * (SIGMOID(X)))

#define LeakyReLU(X) \
  fmax(X, X * alpha)

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

// Uncomment for ReLU, SELU, sigmoid, softplus, swish, and TANH
// Note: input, output are device pointers
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
  SOLUTION();
}

// Uncomment for LeakyReLU
/*
// Note: input, output are device pointers
extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
  SOLUTION();
}
*/

// Uncomment for GELU
/*
// Note: input, output are device pointers
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
  SOLUTION();
}
*/

// Uncomment for ELU
/*
// Note: input, output are device pointers
extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
  SOLUTION();
}
*/

