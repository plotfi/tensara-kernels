// Unified activation shader header (Metal port of kernel-implementation/
// activation.cuh). Selects a scalar-op functor from scalar-ops.metal.h and
// calls activation_x8<Functor>. Included by every activation .metal via the
// harness's local-include inliner (which also inlines scalar-ops.metal.h).
#pragma once
#include <metal_stdlib>
using namespace metal;
#include "scalar-ops.metal.h"   // shared scalar-op functors (Relu, Swish, ...)

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
