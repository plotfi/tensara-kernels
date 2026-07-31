#pragma once

// Metal-backed stubs for the CUDA runtime API used by test_utils.cuh.
//
// test_utils.cuh does #include <cuda_runtime.h> and uses cudaMalloc,
// cudaMemcpy, cudaFree, etc. This file provides those symbols backed by Metal
// shared-storage buffers, registered in tensor::registry() so that solution()
// wrappers can recover the MTL::Buffer via tensor::buf(ptr).
//
// Only the APIs actually used by test_utils.cuh are stubbed here.

#include "../tensor-lib/tensor.cuh"
#include <cstring>

enum cudaError_t { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };

inline const char* cudaGetErrorString(cudaError_t) { return ""; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

template <typename T>
inline cudaError_t cudaMalloc(T** ptr, size_t size) {
    auto* buf = tensor::device()->newBuffer(size, MTL::ResourceStorageModeShared);
    memset(buf->contents(), 0, size);
    *ptr = static_cast<T*>(buf->contents());
    tensor::registry()[buf->contents()] = buf;
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* ptr) {
    auto it = tensor::registry().find(ptr);
    if (it != tensor::registry().end()) {
        it->second->release();
        tensor::registry().erase(it);
    }
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* ptr, int value, size_t count) {
    memset(ptr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind) {
    memcpy(dst, src, count);
    return cudaSuccess;
}
