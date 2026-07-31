// Correctness test for the elementwise activation kernels.
//
// Selected at compile time with the same -DACT_* flag that activations.cu uses,
// so one source tests every activation. The CPU reference below mirrors the
// device formulas; tolerances are loose because the kernel uses fast-math
// intrinsics (__expf, __fdividef, a fast tanh approximation).

#include "test_utils.cuh"

// solution() is linked from solutions-cuda/<activation>.cu; its signature varies by op.
#if defined(ACT_LEAKY_RELU)
extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m);
#elif defined(ACT_ELU)
extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha);
#else
extern "C" void solution(const float* input, float* output, size_t n, size_t m);
#endif

#if defined(ACT_RELU)
#  define ACT_NAME "relu"
#elif defined(ACT_LEAKY_RELU)
#  define ACT_NAME "leaky-relu"
#elif defined(ACT_ELU)
#  define ACT_NAME "elu"
#elif defined(ACT_GELU)
#  define ACT_NAME "gelu"
#elif defined(ACT_SELU)
#  define ACT_NAME "selu"
#elif defined(ACT_SIGMOID)
#  define ACT_NAME "sigmoid"
#elif defined(ACT_HARD_SIGMOID)
#  define ACT_NAME "hard-sigmoid"
#elif defined(ACT_TANH)
#  define ACT_NAME "tanh"
#elif defined(ACT_SOFTPLUS)
#  define ACT_NAME "soft-plus"
#else
#  define ACT_NAME "swish"
#endif

static float ref_activation(float x, float alpha) {
    (void)alpha;
#if defined(ACT_RELU)
    return fmaxf(x, 0.0f);
#elif defined(ACT_LEAKY_RELU)
    return fmaxf(x, x * alpha);
#elif defined(ACT_ELU)
    return x > 0.0f ? x : alpha * (expf(x) - 1.0f);
#elif defined(ACT_GELU)
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
#elif defined(ACT_SELU)
    return 1.0507f * (fmaxf(0.0f, x) + fminf(0.0f, 1.67326f * (expf(x) - 1.0f)));
#elif defined(ACT_SIGMOID)
    return 1.0f / (1.0f + expf(-x));
#elif defined(ACT_HARD_SIGMOID)
    if (x <= -3.0f) return 0.0f;
    if (x >= 3.0f)  return 1.0f;
    return (x + 3.0f) / 6.0f;
#elif defined(ACT_TANH)
    return tanhf(x);
#elif defined(ACT_SOFTPLUS)
    return logf(1.0f + expf(x));
#else // swish
    return x / (1.0f + expf(-x));
#endif
}

int main() {
    test::seed();

    size_t n = 64, m = 64;
    size_t total = n * m;
    float alpha = 0.01f; // used by leaky-relu / elu references

    float* h_in  = new float[total];
    float* h_out = new float[total];
    float* h_ref = new float[total];
    test::fill_random(h_in, total);

    test::DBuf<float> d_in(total), d_out(total);
    d_in.upload(h_in);

#if defined(ACT_LEAKY_RELU)
    solution(d_in, alpha, d_out, n, m);
#elif defined(ACT_ELU)
    alpha = 1.0f;
    solution(d_in, d_out, n, m, alpha);
#else
    solution(d_in, d_out, n, m);
#endif
    test::check_cuda(ACT_NAME);

    d_out.download(h_out);
    for (size_t i = 0; i < total; i++)
        h_ref[i] = ref_activation(h_in[i], alpha);

    int bad = test::compare(ACT_NAME, h_out, h_ref, total, 1e-2f, 1e-3f);
    int rc = test::report(ACT_NAME, bad, total);

    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
