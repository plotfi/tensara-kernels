#pragma once

// Metal backend for the harness (via metal-cpp).
//
// harness::Buffer<T> wraps a single MTL::Buffer with shared (unified) storage,
// so host and device see the same memory. It converts implicitly to T* (the
// shared-memory pointer), exactly like the CUDA backend converts to a device
// pointer -- so a harness calls solution(a, b, c, n) identically on both.
//
// On Metal, solution() is not a kernel launch but a small wrapper (one per
// kernel, in solutions/<kernel>.cpp) that compiles the shader and dispatches it.
// It recovers each MTL::Buffer from the T* it was handed via harness::buf(),
// backed by a registry that Buffer populates on construction. That keeps the
// harness backend-agnostic: no metal-cpp, no #if, no dispatch code.
//
// metal-cpp needs its implementation compiled in exactly one TU per binary (the
// macros NS_PRIVATE_IMPLEMENTATION / MTL_PRIVATE_IMPLEMENTATION defined before
// its headers). That TU is the per-kernel wrapper solutions/<kernel>.cpp, which
// build_metal.sh compiles with -DHARNESS_METAL_IMPL; every other TU (the harness)
// gets declarations only. Defining the macros here — guarded — keeps this
// boilerplate out of both the harnesses and the wrappers.

#include "tensor_common.cuh"

#if defined(TENSOR_METAL_IMPL)
#  define NS_PRIVATE_IMPLEMENTATION
#  define MTL_PRIVATE_IMPLEMENTATION
#endif

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>

// Placeholder CUDA-only element types so harnesses that declare half / fp8
// buffers still compile on Metal (only the byte size matters to the harness;
// the actual quantized kernels are stubs on both backends).
using half = uint16_t;
struct __nv_fp8_e4m3 { uint8_t bits; };

namespace tensor {

inline void fill_random(half* p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = static_cast<half>(rand() & 0xffff);
}
inline void fill_random(__nv_fp8_e4m3* p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i].bits = 0;
}
inline void preview_value(half v)          { printf("%u ", static_cast<unsigned>(v)); }
inline void preview_value(__nv_fp8_e4m3 v) { printf("%u ", static_cast<unsigned>(v.bits)); }

// Lazily-created system default device / command queue, shared for the process.
inline MTL::Device* device() {
    static MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    if (!dev) {
        fprintf(stderr, "tensor: Metal is not supported on this device.\n");
        std::exit(1);
    }
    return dev;
}
inline MTL::CommandQueue* queue() {
    static MTL::CommandQueue* q = device()->newCommandQueue();
    return q;
}

// Registry mapping a Buffer's shared-memory pointer back to its MTL::Buffer, so
// a solution() wrapper can turn the T* it receives into a bindable buffer. The
// map lives in an inline function-local static, so it is shared across the
// harness TU (which registers) and the wrapper TU (which looks up).
inline std::map<const void*, MTL::Buffer*>& registry() {
    static std::map<const void*, MTL::Buffer*> m;
    return m;
}

// Recover the MTL::Buffer a device pointer belongs to (must be a Buffer base
// pointer, which is what a harness passes to solution()).
inline MTL::Buffer* buf(const void* p) {
    auto it = registry().find(p);
    if (it == registry().end()) {
        fprintf(stderr, "tensor: buf() got a pointer not owned by any Buffer\n");
        std::exit(1);
    }
    return it->second;
}

// RAII wrapper over a single shared-storage MTL::Buffer. Converts to T* (the
// shared pointer) so it can be passed straight to solution(), just like CUDA.
template <typename T>
struct Buffer {
    MTL::Buffer* buf = nullptr;
    size_t count = 0;

    explicit Buffer(size_t n) : count(n) {
        buf = device()->newBuffer(n * sizeof(T), MTL::ResourceStorageModeShared);
        memset(buf->contents(), 0, n * sizeof(T));
        registry()[buf->contents()] = buf;
    }
    ~Buffer() { if (buf) { registry().erase(buf->contents()); buf->release(); } }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    size_t size() const { return count; }

    // Host-accessible pointer. With shared storage this IS the device memory.
    T* data() { return static_cast<T*>(buf->contents()); }

    Buffer& fill_random() { ::tensor::fill_random(data(), count); return *this; }

    Buffer& set(std::initializer_list<T> values) {
        T* p = data(); size_t i = 0;
        for (T v : values) if (i < count) p[i++] = v;
        return *this;
    }

    Buffer& fill(T v) {
        T* p = data();
        for (size_t i = 0; i < count; i++) p[i] = v;
        return *this;
    }

    void preview(const char* label, size_t k = 10) {
        T* p = data();
        printf("Output %s (first 10): ", label);
        for (size_t i = 0; i < k && i < count; i++)
            ::tensor::preview_value(p[i]);
        printf("\n");
    }

    // Implicit conversion to the shared-memory pointer (mirrors CUDA's T* dev).
    operator T*() { return static_cast<T*>(buf->contents()); }
};

// ---- shader compilation + pipeline cache --------------------------------

// Directory to look for <kernel>.metal files in. Taken from $TENSOR_SHADER_DIR
// if set, else the working directory. (The repo's shaders live in solutions/.)
inline std::string& shader_dir() {
    static std::string d = []{
        const char* e = getenv("TENSOR_SHADER_DIR");
        return e ? std::string(e) : std::string(".");
    }();
    return d;
}

// Metal's runtime source compiler has no include path for local headers, so we
// inline local (quoted) #include "..." directives ourselves, resolving them
// against `dir`. This lets shaders share a common .metal header (e.g. the unified
// activation header). System includes (<...>) are left untouched; `seen` gives
// pragma-once behavior.
inline void inline_metal_includes(const std::string& path, const std::string& dir,
                                  std::set<std::string>& seen, std::string& out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "tensor: cannot open shader %s\n", path.c_str());
        std::exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        size_t hp = line.find("#include");
        size_t q1 = (hp == std::string::npos) ? std::string::npos : line.find('"', hp);
        if (hp != std::string::npos && q1 != std::string::npos) {
            size_t q2 = line.find('"', q1 + 1);
            std::string inc = line.substr(q1 + 1, q2 - (q1 + 1));
            if (seen.insert(inc).second)
                inline_metal_includes(dir + "/" + inc, dir, seen, out);
            continue;  // replace the local #include with its contents (once)
        }
        out += line;
        out += '\n';
    }
}

inline MTL::ComputePipelineState* compile_pipeline(const std::string& metal_path,
                                                   const std::string& fn_name) {
    std::set<std::string> seen;
    std::string src;
    inline_metal_includes(metal_path, shader_dir(), seen, src);

    NS::Error* err = nullptr;
    NS::String* s = NS::String::string(src.c_str(), NS::UTF8StringEncoding);
    MTL::Library* lib = device()->newLibrary(s, nullptr, &err);
    if (!lib) {
        fprintf(stderr, "tensor: shader compile failed (%s): %s\n", metal_path.c_str(),
                err->localizedDescription()->utf8String());
        std::exit(1);
    }
    NS::String* fn = NS::String::string(fn_name.c_str(), NS::UTF8StringEncoding);
    MTL::Function* func = lib->newFunction(fn);
    if (!func) {
        fprintf(stderr, "tensor: kernel '%s' not found in %s\n", fn_name.c_str(), metal_path.c_str());
        std::exit(1);
    }
    MTL::ComputePipelineState* pso = device()->newComputePipelineState(func, &err);
    func->release(); lib->release();
    if (!pso) {
        fprintf(stderr, "tensor: pipeline build failed: %s\n",
                err->localizedDescription()->utf8String());
        std::exit(1);
    }
    return pso;
}

// Compile (once, cached) the pipeline for kernel function `solution` in
// <shader_dir>/<name>.metal. Safe to call every solution() invocation.
inline MTL::ComputePipelineState* pipeline(const std::string& name) {
    static std::map<std::string, MTL::ComputePipelineState*> cache;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    MTL::ComputePipelineState* pso =
        compile_pipeline(shader_dir() + "/" + name + ".metal", "solution");
    cache[name] = pso;
    return pso;
}

// Wrap a scalar (int, float, size params, ...) as a (bytes, size) pair to bind
// after the buffers. The referenced value must outlive the dispatch call.
template <typename T>
inline std::pair<const void*, size_t> arg(const T& v) {
    return { static_cast<const void*>(&v), sizeof(T) };
}

// ---- dispatch -----------------------------------------------------------

// Encode + commit + wait: bind `buffers` at indices 0..k-1, then `scalars`
// (setBytes) at indices k.., and dispatch a 1-D grid of `grid` threads.
// threadsPerGroup == 0 picks min(pipeline max, grid).
inline void dispatch(MTL::ComputePipelineState* pso,
                     std::initializer_list<MTL::Buffer*> buffers,
                     std::initializer_list<std::pair<const void*, size_t>> scalars,
                     size_t grid, size_t threadsPerGroup = 0) {
    MTL::CommandBuffer* cmd = queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);

    size_t idx = 0;
    for (MTL::Buffer* b : buffers) enc->setBuffer(b, 0, idx++);
    for (const auto& s : scalars)  enc->setBytes(s.first, s.second, idx++);

    if (threadsPerGroup == 0)
        threadsPerGroup = std::min<size_t>(pso->maxTotalThreadsPerThreadgroup(), grid ? grid : 1);

    enc->dispatchThreads(MTL::Size(grid, 1, 1), MTL::Size(threadsPerGroup, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}

// Convenience overload with no scalar arguments.
inline void dispatch(MTL::ComputePipelineState* pso,
                     std::initializer_list<MTL::Buffer*> buffers,
                     size_t grid, size_t threadsPerGroup = 0) {
    dispatch(pso, buffers, {}, grid, threadsPerGroup);
}

// Parity with the CUDA backend's to_host(): with shared storage the "device"
// pointer is already host-visible, so this is a no-op that returns it unchanged
// (the scratch buffer is unused). Read through the returned pointer.
template <typename T>
inline const T* to_host(T* /*scratch*/, const T* device_src, size_t n) {
    (void)n;
    return device_src;
}

// Run `launch` `warmup` times untimed, then time `iters` launches with a host
// clock and print the average per-launch time. `launch` is expected to commit
// AND wait (dispatch does), so wall time is GPU time.
template <typename Fn>
inline void benchmark(Fn&& launch, int warmup = 3, int iters = 100) {
    for (int i = 0; i < warmup; i++) launch();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) launch();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Avg kernel time: %.4f ms (over %d iters)\n", ms / iters, iters);
}

} // namespace tensor
