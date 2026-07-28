// Solution for "leaky-relu" (elementwise activation).
// Selects the op, then pulls in the shared vectorized activation template.
// Guarded so building it alongside a test compiled with -DACT_LEAKY_RELU (for the test's
// CPU reference) doesn't trigger a macro-redefinition warning.
#ifndef ACT_LEAKY_RELU
#define ACT_LEAKY_RELU
#endif
#include "../kernel-implementation/activation.cuh"
