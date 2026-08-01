// Unified activation shader header (Metal port of kernel-implementation/
// activation.cuh). Each activation is a functor with a uniform
// (float x, float alpha) signature; stateless ones ignore alpha. A shader
// selects one and calls activation_x8<Functor>. Included by every activation
// .metal via the harness's local-include inliner.
#pragma once
#include <metal_stdlib>
using namespace metal;

struct Relu        { static inline float apply(float x, float)   { return fmax(x, 0.0f); } };
struct LeakyRelu   { static inline float apply(float x, float a) { return x > 0.0f ? x : x * a; } };
struct Elu         { static inline float apply(float x, float a) { return x > 0.0f ? x : a * (exp(x) - 1.0f); } };
struct Sigmoid     { static inline float apply(float x, float)   { return 1.0f / (1.0f + exp(-x)); } };
struct Swish       { static inline float apply(float x, float)   { return x / (1.0f + exp(-x)); } };
struct TanhAct     { static inline float apply(float x, float)   { return tanh(x); } };
struct Gelu        { static inline float apply(float x, float)   { return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); } };
struct Selu        { static inline float apply(float x, float)   { return 1.0507f * (fmax(0.0f, x) + fmin(0.0f, 1.67326f * (exp(x) - 1.0f))); } };
struct Softplus    { static inline float apply(float x, float)   { return log(1.0f + exp(x)); } };
struct HardSigmoid { static inline float apply(float x, float)   { return x <= -3.0f ? 0.0f : (x >= 3.0f ? 1.0f : (x + 3.0f) / 6.0f); } };
// Not a neural activation, but shares the elementwise shape: binary threshold.
struct Threshold   { static inline float apply(float x, float t) { return x > t ? 1.0f : 0.0f; } };

// 8 elements per thread (two float4s back to back), mirroring activation.cuh's
// activation_kernelx8. `n` = total elements, grid = ceil(n/8) threads.
template <typename ACT>
inline void activation_x8(device const float* A, device float* C, uint n, float alpha, uint gid) {
    uint base = gid * 8u;
    if (base + 7u >= n) {                       // scalar tail for the last partial group
        for (uint j = base; j < n; ++j) C[j] = ACT::apply(A[j], alpha);
        return;
    }
    device const float4* A4 = (device const float4*)(A + base);
    device float4*       C4 = (device float4*)(C + base);
    float4 x0 = A4[0];
    float4 x1 = A4[1];
    x0 = float4(ACT::apply(x0.x, alpha), ACT::apply(x0.y, alpha), ACT::apply(x0.z, alpha), ACT::apply(x0.w, alpha));
    x1 = float4(ACT::apply(x1.x, alpha), ACT::apply(x1.y, alpha), ACT::apply(x1.z, alpha), ACT::apply(x1.w, alpha));
    C4[0] = x0;
    C4[1] = x1;
}
