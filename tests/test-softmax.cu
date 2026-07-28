// Correctness test for softmax over dim=1 of a 2D [64,64] tensor (per-row).
// out[i,j] = exp(x[i,j] - max_i) / sum_j exp(x[i,j] - max_i).

#include "test_utils.cuh"
extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim);

int main() {
    test::seed();
    size_t rows = 64, cols = 64;
    int dim = 1; size_t ndim = 2;

    float* h_in  = new float[rows * cols];
    float* h_out = new float[rows * cols];
    float* h_ref = new float[rows * cols];
    test::fill_random(h_in, rows * cols);

    test::DBuf<float> d_in(rows * cols), d_out(rows * cols);
    test::DBuf<size_t> d_shape(ndim);
    d_in.upload(h_in);
    size_t shape[2] = {rows, cols};
    d_shape.upload(shape);

    solution(d_in, dim, d_out, d_shape, ndim);
    test::check_cuda("softmax");
    d_out.download(h_out);

    for (size_t i = 0; i < rows; i++) {
        float mx = -INFINITY;
        for (size_t j = 0; j < cols; j++) mx = fmaxf(mx, h_in[i * cols + j]);
        double sum = 0.0;
        for (size_t j = 0; j < cols; j++) sum += exp(static_cast<double>(h_in[i * cols + j] - mx));
        for (size_t j = 0; j < cols; j++)
            h_ref[i * cols + j] = static_cast<float>(exp(static_cast<double>(h_in[i * cols + j] - mx)) / sum);
    }

    int bad = test::compare("softmax", h_out, h_ref, rows * cols, 1e-3f, 1e-4f);
    int rc = test::report("softmax", bad, rows * cols);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
