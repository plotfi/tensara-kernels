// Correctness test for ecc-point-negation over F_p: -(x, y) = (x, -y mod p).
// out_xy is interleaved [x0,y0, x1,y1, ...]:
//   out_xy[2i]   = x_i mod p
//   out_xy[2i+1] = (p - (y_i mod p)) mod p     (0 stays 0)
// The harness fills xs/ys with rand() values (< 2^31, well below p).

#include "test_utils.cuh"
extern "C" void solution(const uint64_t* xs, const uint64_t* ys, uint64_t p,
                         uint64_t* out_xy, size_t n);

int main() {
    test::seed();
    uint64_t p = 18446744073709551557ULL;
    size_t n = 1024;

    uint64_t* h_xs = new uint64_t[n];
    uint64_t* h_ys = new uint64_t[n];
    uint64_t* h_out = new uint64_t[n * 2];
    uint64_t* h_ref = new uint64_t[n * 2];
    test::fill_random_u64(h_xs, n);
    test::fill_random_u64(h_ys, n);

    test::DBuf<uint64_t> d_xs(n), d_ys(n), d_out(n * 2);
    d_xs.upload(h_xs); d_ys.upload(h_ys);
    solution(d_xs, d_ys, p, d_out, n);
    test::check_cuda("ecc-point-negation");
    d_out.download(h_out);

    for (size_t i = 0; i < n; i++) {
        uint64_t x = h_xs[i] % p;
        uint64_t y = h_ys[i] % p;
        h_ref[2 * i]     = x;
        h_ref[2 * i + 1] = (y == 0) ? 0 : (p - y);
    }

    int bad = test::compare_int("ecc-point-negation", h_out, h_ref, n * 2);
    int rc = test::report("ecc-point-negation", bad, n * 2);
    delete[] h_xs; delete[] h_ys; delete[] h_out; delete[] h_ref;
    return rc;
}
