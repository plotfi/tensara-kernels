// Unified reduction shader header (Metal port of kernel-implementation/
// reduction.cuh). One threadgroup per row: reduce the row to a scalar, finalize
// it, then rewrite each element as output(element, scalar). An op is a functor
// bundling the five hooks; a shader selects one and calls reduce_row<Ops>.
// Included by every reduction .metal via the harness's local-include inliner.
#pragma once
#include <metal_stdlib>
using namespace metal;

// --- op catalog (mirrors reduction.cuh's ReduceOps aliases) ----------------
struct L1NormOps     { static inline float map(float x){return fabs(x);}  static inline float reduce(float a,float b){return a+b;}      static inline float identity(){return 0.0f;}      static inline float finalize(float s,uint){return 1.0f/s;}                 static inline float output(float x,float f){return x*f;} };
struct L2NormOps     { static inline float map(float x){return x*x;}      static inline float reduce(float a,float b){return a+b;}      static inline float identity(){return 0.0f;}      static inline float finalize(float s,uint){return rsqrt(s);}               static inline float output(float x,float f){return x*f;} };
struct RMSNormOps    { static inline float map(float x){return x*x;}      static inline float reduce(float a,float b){return a+b;}      static inline float identity(){return 0.0f;}      static inline float finalize(float s,uint D){return rsqrt(s/float(D)+1e-5f);} static inline float output(float x,float f){return x*f;} };
struct MeanSubOps    { static inline float map(float x){return x;}        static inline float reduce(float a,float b){return a+b;}      static inline float identity(){return 0.0f;}      static inline float finalize(float s,uint D){return s/float(D);}            static inline float output(float x,float f){return x-f;} };
struct AbsMaxNormOps { static inline float map(float x){return fabs(x);}  static inline float reduce(float a,float b){return fmax(a,b);} static inline float identity(){return 0.0f;}      static inline float finalize(float s,uint){return 1.0f/s;}                 static inline float output(float x,float f){return x*f;} };
struct MaxNormOps    { static inline float map(float x){return x;}        static inline float reduce(float a,float b){return fmax(a,b);} static inline float identity(){return -INFINITY;} static inline float finalize(float s,uint){return 1.0f/s;}                 static inline float output(float x,float f){return x*f;} };
struct SumNormOps    { static inline float map(float x){return x;}        static inline float reduce(float a,float b){return a+b;}      static inline float identity(){return 0.0f;}      static inline float finalize(float s,uint){return 1.0f/s;}                 static inline float output(float x,float f){return x*f;} };
struct LogSoftmaxOps { static inline float map(float x){return exp(x);}   static inline float reduce(float a,float b){return a+b;}      static inline float identity(){return 0.0f;}      static inline float finalize(float s,uint){return log(s);}                 static inline float output(float x,float f){return x-f;} };
struct SoftmaxOps    { static inline float map(float x){return exp(x);}   static inline float reduce(float a,float b){return a+b;}      static inline float identity(){return 0.0f;}      static inline float finalize(float s,uint){return s;}                      static inline float output(float x,float f){return exp(x)/f;} };

// --- shared kernel body: one threadgroup per row, tree reduce over the row ---
// `smem` is a threadgroup array of size >= tpg, declared in the kernel and
// passed in (MSL threadgroup memory can't be declared inside a helper).
template <typename Ops>
inline void reduce_row(device const float* X, device float* Y, uint D,
                       uint tid, uint row, uint tpg, threadgroup float* smem) {
    const device float* rx = X + (uint)row * D;
    device float*       ry = Y + (uint)row * D;

    float acc = Ops::identity();
    for (uint i = tid; i < D; i += tpg) acc = Ops::reduce(acc, Ops::map(rx[i]));
    smem[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = Ops::reduce(smem[tid], smem[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float f = Ops::finalize(smem[0], D);
    for (uint i = tid; i < D; i += tpg) ry[i] = Ops::output(rx[i], f);
}
