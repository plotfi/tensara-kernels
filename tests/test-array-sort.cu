// Correctness test for array-sort: b = sort(a) ascending.

#include "test_utils.cuh"
#include <algorithm>
extern "C" void solution(const int* a, int* b, size_t n);

int main() {
    test::seed();
    size_t n = 1024;

    int* h_a = new int[n];
    int* h_b = new int[n];
    int* h_ref = new int[n];
    test::fill_random_int(h_a, n, -100000, 100000);

    test::DBuf<int> d_a(n), d_b(n);
    d_a.upload(h_a);
    solution(d_a, d_b, n);
    test::check_cuda("array-sort");
    d_b.download(h_b);

    for (size_t i = 0; i < n; i++) h_ref[i] = h_a[i];
    std::sort(h_ref, h_ref + n);

    int bad = test::compare_int("array-sort", h_b, h_ref, n);
    int rc = test::report("array-sort", bad, n);
    delete[] h_a; delete[] h_b; delete[] h_ref;
    return rc;
}
