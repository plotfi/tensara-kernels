// argmax — index of max along dim of an N-D tensor. Uses the shared dim-reduce framework.
#include "../kernel-implementation/dim-reduce.cuh"

extern "C" void solution(const float* input, int dim, int* output, const int* shape, int ndim) {
    launch_argreduce<1>(input, output, shape, ndim, dim);
}
