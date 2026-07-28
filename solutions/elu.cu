// Solution for "elu" (elementwise activation).
// Selects the op, then pulls in the shared vectorized activation template.
// Guarded so building it alongside a test compiled with -DACT_ELU (for the test's
// CPU reference) doesn't trigger a macro-redefinition warning.
#ifndef ACT_ELU
#define ACT_ELU
#endif
#include "../kernel-implementation/activation.cuh"
