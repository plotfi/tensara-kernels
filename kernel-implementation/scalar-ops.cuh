#pragma once
#include <cuda_runtime.h>

// Single source of truth for the elementwise scalar ops shared across the repo:
//   * activation.cuh   -- its function-pointer kernel calls each op's static
//                         apply(x, alpha) via a thin free-function wrapper.
//   * gemm-epilogue.cuh -- its fused GEMM applies these as functors (operator()
//                         and Compose<>) in the store epilogue.
//
// Each op is therefore a functor exposing both forms:
//   static apply(x, alpha) -- the math, for the (float,float) function-pointer path.
//   operator()(x)          -- the functor path (epilogue / Compose). Parameterized
//                             ops carry their parameter as a member.
// This mirrors the Metal-side functor layout in activation.metal.h.

__device__ __forceinline__ float fast_tanh(float x) {
    float e2x = __expf(2.0f * x);
    return __fdividef(e2x - 1.0f, e2x + 1.0f);
}

struct Relu {
    static __device__ __forceinline__ float apply(float x, float) { return x > 0.0f ? x : 0.0f; }
    __device__ __forceinline__ float operator()(float x) const { return apply(x, 0.0f); }
};
struct LeakyRelu {
    float alpha = 0.0f;
    static __device__ __forceinline__ float apply(float x, float a) { return x > 0.0f ? x : x * a; }
    __device__ __forceinline__ float operator()(float x) const { return apply(x, alpha); }
};
struct Elu {
    float alpha = 1.0f;
    static __device__ __forceinline__ float apply(float x, float a) { return x > 0.0f ? x : a * (__expf(x) - 1.0f); }
    __device__ __forceinline__ float operator()(float x) const { return apply(x, alpha); }
};
struct Sigmoid {
    static __device__ __forceinline__ float apply(float x, float) { return __fdividef(1.0f, 1.0f + __expf(-x)); }
    __device__ __forceinline__ float operator()(float x) const { return apply(x, 0.0f); }
};
struct Swish {
    static __device__ __forceinline__ float apply(float x, float) { return x * __fdividef(1.0f, 1.0f + __expf(-x)); }
    __device__ __forceinline__ float operator()(float x) const { return apply(x, 0.0f); }
};
struct TanhAct {
    static __device__ __forceinline__ float apply(float x, float) { return fast_tanh(x); }
    __device__ __forceinline__ float operator()(float x) const { return apply(x, 0.0f); }
};
struct Gelu {
    static __device__ __forceinline__ float apply(float x, float) {
        constexpr float kSqrt2OverPi = 0.7978845608028654f;
        constexpr float kCoef = 0.044715f;
        return 0.5f * x * (1.0f + fast_tanh(kSqrt2OverPi * (x + kCoef * x * x * x)));
    }
    __device__ __forceinline__ float operator()(float x) const { return apply(x, 0.0f); }
};
struct Selu {
    static __device__ __forceinline__ float apply(float x, float) {
        return 1.0507f * (fmaxf(0.0f, x) + fminf(0.0f, 1.67326f * (__expf(x) - 1.0f)));
    }
    __device__ __forceinline__ float operator()(float x) const { return apply(x, 0.0f); }
};
struct Softplus {
    static __device__ __forceinline__ float apply(float x, float) { return __logf(1.0f + __expf(x)); }
    __device__ __forceinline__ float operator()(float x) const { return apply(x, 0.0f); }
};
struct HardSigmoid {
    static __device__ __forceinline__ float apply(float x, float) {
        return x <= -3.0f ? 0.0f : (x >= 3.0f ? 1.0f : __fdividef(x + 3.0f, 6.0f));
    }
    __device__ __forceinline__ float operator()(float x) const { return apply(x, 0.0f); }
};
// Not a neural activation, but shares the elementwise shape: binary threshold
// with alpha = threshold value.
struct Threshold {
    float t = 0.0f;
    static __device__ __forceinline__ float apply(float x, float a) { return x > a ? 1.0f : 0.0f; }
    __device__ __forceinline__ float operator()(float x) const { return apply(x, t); }
};

// Combinators with no activation-kernel counterpart (epilogue use only).
struct Identity { __device__ __forceinline__ float operator()(float x) const { return x; } };
struct Scale    { float s;  __device__ __forceinline__ float operator()(float x) const { return x * s; } };

// Compose two ops: apply F then G, both inlined -> G(F(x)).
template <class F, class G>
struct Compose {
    F f; G g;
    __device__ __forceinline__ float operator()(float x) const { return g(f(x)); }
};
