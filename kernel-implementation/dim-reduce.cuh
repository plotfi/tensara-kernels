#pragma once
// Generic reduction along one axis `dim` of an N-D tensor described by shape/ndim.
// A tensor factors as [outer, L, inner] around the reduced axis (L = shape[dim]),
// so output[o, i] = reduce over l of input[o*L*inner + l*inner + i]. One thread
// per output element. `shape` is a device pointer, copied to host to compute the
// three extents. Covers sum/mean/max/min/product (value) and argmax/argmin (index).
#include <cuda_runtime.h>
#include <cfloat>

// ---- value ops: init / combine / finalize(acc, L) --------------------------
struct SumOp  { __device__ static float init(){return 0.0f;}      __device__ static float combine(float a,float b){return a+b;}        __device__ static float finalize(float a,long){return a;} };
struct MeanOp { __device__ static float init(){return 0.0f;}      __device__ static float combine(float a,float b){return a+b;}        __device__ static float finalize(float a,long L){return a/(float)L;} };
struct MaxOp  { __device__ static float init(){return -FLT_MAX;}  __device__ static float combine(float a,float b){return fmaxf(a,b);}  __device__ static float finalize(float a,long){return a;} };
struct MinOp  { __device__ static float init(){return  FLT_MAX;}  __device__ static float combine(float a,float b){return fminf(a,b);}  __device__ static float finalize(float a,long){return a;} };
struct ProdOp { __device__ static float init(){return 1.0f;}      __device__ static float combine(float a,float b){return a*b;}        __device__ static float finalize(float a,long){return a;} };

template <class Op>
__global__ void dimreduce_kernel(const float* __restrict__ in, float* __restrict__ out,
                                 long outer, long L, long inner) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;   // output element
    if (idx >= outer * inner) return;
    long o = idx / inner, i = idx % inner;
    const float* base = in + o * L * inner + i;
    float acc = Op::init();
    for (long l = 0; l < L; ++l) acc = Op::combine(acc, base[l * inner]);
    out[idx] = Op::finalize(acc, L);
}

// ---- argmax/argmin: SIGN=+1 -> max (v>best), SIGN=-1 -> min (v<best) --------
template <int SIGN>
__global__ void argreduce_kernel(const float* __restrict__ in, int* __restrict__ out,
                                 long outer, long L, long inner) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;
    long o = idx / inner, i = idx % inner;
    const float* base = in + o * L * inner + i;
    float best = base[0]; int bi = 0;
    for (long l = 1; l < L; ++l) {
        float v = base[l * inner];
        if (SIGN * v > SIGN * best) { best = v; bi = (int)l; }   // strict => first index wins
    }
    out[idx] = bi;
}

// ---- extent helper: outer/L/inner from a device shape (T = size_t or int) --
template <class T>
static inline void dimreduce_extents(const T* shape_dev, int ndim, int dim,
                                     long& outer, long& L, long& inner) {
    T hshape[16];
    cudaMemcpy(hshape, shape_dev, (size_t)ndim * sizeof(T), cudaMemcpyDeviceToHost);
    outer = 1; inner = 1; L = (long)hshape[dim];
    for (int a = 0; a < dim; ++a)       outer *= (long)hshape[a];
    for (int a = dim + 1; a < ndim; ++a) inner *= (long)hshape[a];
}

template <class Op>
static inline void launch_dimreduce(const float* in, float* out,
                                    const size_t* shape_dev, size_t ndim, int dim) {
    long outer, L, inner;
    dimreduce_extents<size_t>(shape_dev, (int)ndim, dim, outer, L, inner);
    long nout = outer * inner;
    int BS = 256; long grid = (nout + BS - 1) / BS;
    dimreduce_kernel<Op><<<grid, BS>>>(in, out, outer, L, inner);
}

template <int SIGN>
static inline void launch_argreduce(const float* in, int* out,
                                    const int* shape_dev, int ndim, int dim) {
    long outer, L, inner;
    dimreduce_extents<int>(shape_dev, ndim, dim, outer, L, inner);
    long nout = outer * inner;
    int BS = 256; long grid = (nout + BS - 1) / BS;
    argreduce_kernel<SIGN><<<grid, BS>>>(in, out, outer, L, inner);
}
