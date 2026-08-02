// Metal mirror of scalar-ops.cuh: the shared elementwise scalar-op functors.
// Used by the activation shaders (activation.metal.h) and available to fused
// epilogues. Each op exposes both forms it's consumed in:
//   static apply(x, alpha) -- the math (used by activation_x8<ACT>)
//   operator()(x)          -- the functor path (epilogue / Compose); parameterized
//                             ops carry their parameter as a member.
#pragma once
#include <metal_stdlib>
using namespace metal;

struct Relu        { static inline float apply(float x, float)   { return fmax(x, 0.0f); }
                     inline float operator()(float x) const { return apply(x, 0.0f); } };
struct LeakyRelu   { float alpha;
                     static inline float apply(float x, float a) { return x > 0.0f ? x : x * a; }
                     inline float operator()(float x) const { return apply(x, alpha); } };
struct Elu         { float alpha;
                     static inline float apply(float x, float a) { return x > 0.0f ? x : a * (exp(x) - 1.0f); }
                     inline float operator()(float x) const { return apply(x, alpha); } };
struct Sigmoid     { static inline float apply(float x, float)   { return 1.0f / (1.0f + exp(-x)); }
                     inline float operator()(float x) const { return apply(x, 0.0f); } };
struct Swish       { static inline float apply(float x, float)   { return x / (1.0f + exp(-x)); }
                     inline float operator()(float x) const { return apply(x, 0.0f); } };
struct TanhAct     { static inline float apply(float x, float)   { return tanh(x); }
                     inline float operator()(float x) const { return apply(x, 0.0f); } };
struct Gelu        { static inline float apply(float x, float)   { return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
                     inline float operator()(float x) const { return apply(x, 0.0f); } };
struct Selu        { static inline float apply(float x, float)   { return 1.0507f * (fmax(0.0f, x) + fmin(0.0f, 1.67326f * (exp(x) - 1.0f))); }
                     inline float operator()(float x) const { return apply(x, 0.0f); } };
struct Softplus    { static inline float apply(float x, float)   { return log(1.0f + exp(x)); }
                     inline float operator()(float x) const { return apply(x, 0.0f); } };
struct HardSigmoid { static inline float apply(float x, float)   { return x <= -3.0f ? 0.0f : (x >= 3.0f ? 1.0f : (x + 3.0f) / 6.0f); }
                     inline float operator()(float x) const { return apply(x, 0.0f); } };
// Not a neural activation, but shares the elementwise shape: binary threshold.
struct Threshold   { float t;
                     static inline float apply(float x, float a) { return x > a ? 1.0f : 0.0f; }
                     inline float operator()(float x) const { return apply(x, t); } };

// Combinators with no activation-kernel counterpart (epilogue use only).
struct Identity { inline float operator()(float x) const { return x; } };
struct Scale    { float s; inline float operator()(float x) const { return x * s; } };

// Compose two ops: apply F then G, both inlined -> G(F(x)).
template <typename F, typename G>
struct Compose { F f; G g; inline float operator()(float x) const { return g(f(x)); } };
