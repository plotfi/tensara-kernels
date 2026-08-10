#include "block-reduce.cuh"

struct HingeLossImpl {
  __device__ __forceinline__ static float apply(float x, float y) {
    return fmaxf(0, 1 - x * y);
  }
};

struct HuberLossImpl {
  __device__ __forceinline__ static float apply(float x, float y) {
    float a = fabsf(x - y);
    return (a < 1) ? (0.5f * a * a) : (a - 0.5f);
  }
};

struct MseLossImpl {
  __device__ __forceinline__ static float apply(float x, float y) {
    float a = x - y;
    return a * a;
  }
};
