// Correctness test for all-pairs-shortest-path (Floyd-Warshall).
// The test builds a well-defined dense graph: non-negative edge weights in [0,1]
// and a zero diagonal (so shortest paths are well-defined, no negative cycles).
// output[i*n+j] = shortest distance from i to j.

#include "test_utils.cuh"
extern "C" void solution(const float* adj_matrix, float* output, size_t n);

int main() {
    test::seed();
    size_t n = 64;

    float* h_adj = new float[n * n];
    float* h_out = new float[n * n];
    float* h_ref = new float[n * n];
    test::fill_random(h_adj, n * n, 0.0f, 1.0f);
    for (size_t i = 0; i < n; i++) h_adj[i * n + i] = 0.0f;

    test::DBuf<float> d_adj(n * n), d_out(n * n);
    d_adj.upload(h_adj);
    solution(d_adj, d_out, n);
    test::check_cuda("all-pairs-shortest-path");
    d_out.download(h_out);

    for (size_t i = 0; i < n * n; i++) h_ref[i] = h_adj[i];
    for (size_t k = 0; k < n; k++)
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++) {
                double via = (double)h_ref[i * n + k] + h_ref[k * n + j];
                if (via < h_ref[i * n + j]) h_ref[i * n + j] = static_cast<float>(via);
            }

    int bad = test::compare("all-pairs-shortest-path", h_out, h_ref, n * n, 1e-4f, 1e-5f);
    int rc = test::report("all-pairs-shortest-path", bad, n * n);
    delete[] h_adj; delete[] h_out; delete[] h_ref;
    return rc;
}
