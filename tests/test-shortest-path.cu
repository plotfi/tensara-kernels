// Correctness test for single-source shortest path (Dijkstra on a dense graph).
// Non-negative edge weights in [0,1], zero diagonal. source=0.
// d_distances[j] = shortest distance from source to j.

#include "test_utils.cuh"
extern "C" void solution(const float* d_adj_matrix, int source, float* d_distances, size_t n);

int main() {
    test::seed();
    size_t n = 64;
    int source = 0;

    float* h_adj  = new float[n * n];
    float* h_dist = new float[n];
    float* h_ref  = new float[n];
    test::fill_random(h_adj, n * n, 0.0f, 1.0f);
    for (size_t i = 0; i < n; i++) h_adj[i * n + i] = 0.0f;

    test::DBuf<float> d_adj(n * n), d_dist(n);
    d_adj.upload(h_adj);
    solution(d_adj, source, d_dist, n);
    test::check_cuda("shortest-path");
    d_dist.download(h_dist);

    // Dijkstra (dense, O(n^2)).
    bool* done = new bool[n];
    for (size_t i = 0; i < n; i++) { h_ref[i] = INFINITY; done[i] = false; }
    h_ref[source] = 0.0f;
    for (size_t iter = 0; iter < n; iter++) {
        int u = -1; double best = INFINITY;
        for (size_t v = 0; v < n; v++) if (!done[v] && h_ref[v] < best) { best = h_ref[v]; u = (int)v; }
        if (u < 0) break;
        done[u] = true;
        for (size_t v = 0; v < n; v++) {
            double cand = (double)h_ref[u] + h_adj[u * n + v];
            if (cand < h_ref[v]) h_ref[v] = static_cast<float>(cand);
        }
    }
    delete[] done;

    int bad = test::compare("shortest-path", h_dist, h_ref, n, 1e-4f, 1e-5f);
    int rc = test::report("shortest-path", bad, n);
    delete[] h_adj; delete[] h_dist; delete[] h_ref;
    return rc;
}
