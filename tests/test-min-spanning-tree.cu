// Correctness test for minimum-spanning-tree total weight (Prim's algorithm).
// The test builds a symmetric graph with non-negative weights in [0,1] and a
// zero diagonal. min_weight[0] = total weight of the MST over all n vertices.

#include "test_utils.cuh"
extern "C" void solution(const float* A, float* min_weight, size_t n);

int main() {
    test::seed();
    size_t n = 64;

    float* h_a = new float[n * n];
    float h_o = 0.0f, h_ref = 0.0f;
    test::fill_random(h_a, n * n, 0.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        h_a[i * n + i] = 0.0f;
        for (size_t j = i + 1; j < n; j++) h_a[j * n + i] = h_a[i * n + j]; // symmetric
    }

    test::DBuf<float> d_a(n * n), d_o(1);
    d_a.upload(h_a);
    solution(d_a, d_o, n);
    test::check_cuda("min-spanning-tree");
    d_o.download(&h_o);

    // Prim's algorithm (dense).
    bool* in_tree = new bool[n];
    float* key = new float[n];
    for (size_t i = 0; i < n; i++) { key[i] = INFINITY; in_tree[i] = false; }
    key[0] = 0.0f;
    double total = 0.0;
    for (size_t iter = 0; iter < n; iter++) {
        int u = -1; double best = INFINITY;
        for (size_t v = 0; v < n; v++) if (!in_tree[v] && key[v] < best) { best = key[v]; u = (int)v; }
        if (u < 0) break;
        in_tree[u] = true;
        total += key[u];
        for (size_t v = 0; v < n; v++)
            if (!in_tree[v] && h_a[u * n + v] < key[v]) key[v] = h_a[u * n + v];
    }
    h_ref = static_cast<float>(total);
    delete[] in_tree; delete[] key;

    int bad = test::compare("min-spanning-tree", &h_o, &h_ref, 1, 1e-3f, 1e-4f);
    int rc = test::report("min-spanning-tree", bad, 1);
    delete[] h_a;
    return rc;
}
