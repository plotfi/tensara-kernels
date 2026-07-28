// Solution for "sigmoid" (elementwise activation).
// Selects the op, then pulls in the shared vectorized activation template.
// Guarded so building it alongside a test compiled with -DACT_SIGMOID (for the test's
// CPU reference) doesn't trigger a macro-redefinition warning.
#ifndef ACT_SIGMOID
#define ACT_SIGMOID
#endif
#include "../kernel-implementation/activation.cuh"
