// Solution for "tanh" (elementwise activation).
// Selects the op, then pulls in the shared vectorized activation template.
// Guarded so building it alongside a test compiled with -DACT_TANH (for the test's
// CPU reference) doesn't trigger a macro-redefinition warning.
#ifndef ACT_TANH
#define ACT_TANH
#endif
#include "../kernel-implementation/activation.cuh"
