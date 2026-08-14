// max-dim — reduce along dim of an N-D tensor (shape/ndim). Uses the shared dim-reduce framework.
#include "../kernel-implementation/dim-reduce.cuh"

extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim) {
    launch_dimreduce<MaxOp>(input, output, shape, ndim, dim);
}
