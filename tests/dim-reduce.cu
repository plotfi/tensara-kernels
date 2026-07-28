// Correctness test for dimension reductions over dim=1 of a 2D [64,64] tensor
// (reduce each row to one value -> 64 outputs). Op selected at compile time:
//   -DOP_SUM -> sum-dim   -DOP_MEAN -> mean-dim   -DOP_MAX -> max-dim
//   -DOP_MIN -> min-dim   -DOP_PROD -> product-dim

#include "test_utils.cuh"
extern "C" void solution(const float* input, int dim, float* output,
                         const size_t* shape, size_t ndim);

#if defined(OP_MEAN)
#  define OP_NAME "mean-dim"
#elif defined(OP_MAX)
#  define OP_NAME "max-dim"
#elif defined(OP_MIN)
#  define OP_NAME "min-dim"
#elif defined(OP_PROD)
#  define OP_NAME "product-dim"
#else
#  define OP_NAME "sum-dim"
#endif

int main() {
    test::seed();
    size_t rows = 64, cols = 64;
    int dim = 1; size_t ndim = 2;

    float* h_in  = new float[rows * cols];
    float* h_out = new float[rows];
    float* h_ref = new float[rows];
    test::fill_random(h_in, rows * cols);

    test::DBuf<float> d_in(rows * cols), d_out(rows);
    test::DBuf<size_t> d_shape(ndim);
    d_in.upload(h_in);
    size_t shape[2] = {rows, cols};
    d_shape.upload(shape);

    solution(d_in, dim, d_out, d_shape, ndim);
    test::check_cuda(OP_NAME);
    d_out.download(h_out);

    for (size_t i = 0; i < rows; i++) {
        const float* row = h_in + i * cols;
#if defined(OP_MAX)
        float acc = row[0];
        for (size_t j = 1; j < cols; j++) acc = fmaxf(acc, row[j]);
        h_ref[i] = acc;
#elif defined(OP_MIN)
        float acc = row[0];
        for (size_t j = 1; j < cols; j++) acc = fminf(acc, row[j]);
        h_ref[i] = acc;
#elif defined(OP_PROD)
        double acc = 1.0;
        for (size_t j = 0; j < cols; j++) acc *= row[j];
        h_ref[i] = static_cast<float>(acc);
#else // sum or mean
        double acc = 0.0;
        for (size_t j = 0; j < cols; j++) acc += row[j];
#  if defined(OP_MEAN)
        h_ref[i] = static_cast<float>(acc / cols);
#  else
        h_ref[i] = static_cast<float>(acc);
#  endif
#endif
    }

    // product-dim of 64 values in [-1,1] can be tiny; use an absolute floor.
    int bad = test::compare(OP_NAME, h_out, h_ref, rows, 1e-3f, 1e-5f);
    int rc = test::report(OP_NAME, bad, rows);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
