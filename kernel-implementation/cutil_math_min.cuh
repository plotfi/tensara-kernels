#pragma once
// Minimal float3 vector math (host+device) — the subset of the old CUDA SDK
// cutil_math.h that CS4803 lab4 (raytracer) needs. CUDA 12 no longer ships it.
#include <cuda_runtime.h>
#include <math.h>

__host__ __device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline float3 operator-(float3 a)           { return make_float3(-a.x, -a.y, -a.z); }
__host__ __device__ inline float3 operator*(float3 a, float s)  { return make_float3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ inline float3 operator*(float s, float3 a)  { return make_float3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ inline float3 operator/(float3 a, float s)  { return make_float3(a.x/s, a.y/s, a.z/s); }
__host__ __device__ inline void   operator+=(float3 &a, float3 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; }
__host__ __device__ inline void   operator*=(float3 &a, float s)  { a.x*=s; a.y*=s; a.z*=s; }
__host__ __device__ inline void   operator/=(float3 &a, float s)  { a.x/=s; a.y/=s; a.z/=s; }

__host__ __device__ inline float dot(float3 a, float3 b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__host__ __device__ inline float  length(float3 v)    { return sqrtf(dot(v, v)); }
__host__ __device__ inline float3 normalize(float3 v) { float inv = 1.0f/sqrtf(dot(v,v)); return v*inv; }
__host__ __device__ inline float3 reflect(float3 i, float3 n) { return i - n*(2.0f*dot(n,i)); }
__host__ __device__ inline float  clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
