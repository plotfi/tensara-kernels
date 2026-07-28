// Solution for "soft-plus" (elementwise activation).
// Selects the op, then pulls in the shared vectorized activation template.
// Guarded so building it alongside a test compiled with -DACT_SOFTPLUS (for the test's
// CPU reference) doesn't trigger a macro-redefinition warning.
#ifndef ACT_SOFTPLUS
#define ACT_SOFTPLUS
#endif
#include "../kernel-implementation/activation.cuh"
