// Correctness test for argmax / argmin over dim=1 of a 2D [64,64] tensor.
// Selected at compile time: -DARG_MIN -> argmin (default argmax).
// Output[i] = index (0..cols-1) of the max/min in row i. Ties -> first index.
// Inputs are made distinct per row so the argmax/argmin index is unambiguous.

#include "test_utils.cuh"
extern "C" void solution(const float* input, int dim, int* output,
                         const int* shape, int ndim);

#if defined(ARG_MIN)
#  define ARG_NAME "argmin"
#else
#  define ARG_NAME "argmax"
#endif

int main() {
    test::seed();
    int rows = 64, cols = 64;
    int dim = 1, ndim = 2;

    float* h_in  = new float[rows * cols];
    int*   h_out = new int[rows];
    int*   h_ref = new int[rows];
    test::fill_random(h_in, (size_t)rows * cols);
    // Nudge values to be distinct within a row (avoid tie ambiguity).
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            h_in[i * cols + j] += j * 1e-4f;

    test::DBuf<float> d_in((size_t)rows * cols);
    test::DBuf<int>   d_out(rows), d_shape(ndim);
    d_in.upload(h_in);
    int shape[2] = {rows, cols};
    d_shape.upload(shape);

    solution(d_in, dim, d_out, d_shape, ndim);
    test::check_cuda(ARG_NAME);
    d_out.download(h_out);

    for (int i = 0; i < rows; i++) {
        const float* row = h_in + (size_t)i * cols;
        int best = 0;
        for (int j = 1; j < cols; j++) {
#if defined(ARG_MIN)
            if (row[j] < row[best]) best = j;
#else
            if (row[j] > row[best]) best = j;
#endif
        }
        h_ref[i] = best;
    }

    int bad = test::compare_int(ARG_NAME, h_out, h_ref, rows);
    int rc = test::report(ARG_NAME, bad, rows);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
