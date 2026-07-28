#!/bin/bash
# Build the reference-suite tests (kernels implemented in solutions/) into
# tests/build/bin/. Each test links its solutions/<kernel>.cu, same as the spec
# suite. The activation test is one source compiled once per -DACT_* flag.
set -e

cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
NVCCFLAGS=${NVCCFLAGS:--O2 -std=c++17}
BINDIR=build/bin
SOL=../solutions
mkdir -p "$BINDIR"

# --- activations: one binary per -DACT_* flag, linking solutions/<act>.cu ----
declare -A ACT_FLAGS=(
    [relu]=-DACT_RELU
    [elu]=-DACT_ELU
    [leaky-relu]=-DACT_LEAKY_RELU
    [swish]=-DACT_SWISH
    [gelu]=-DACT_GELU
    [selu]=-DACT_SELU
    [sigmoid]=-DACT_SIGMOID
    [soft-plus]=-DACT_SOFTPLUS
    [tanh]=-DACT_TANH
    [hard-sigmoid]=-DACT_HARD_SIGMOID
)
for name in "${!ACT_FLAGS[@]}"; do
    echo "  build test-$name.exe"
    $NVCC $NVCCFLAGS ${ACT_FLAGS[$name]} -o "$BINDIR/test-$name.exe" test-activation.cu "$SOL/$name.cu"
done

# --- reductions + standalone kernels: link solutions/<kernel>.cu -------------
for kernel in vector-addition matrix-vector conv-1d \
              rms-norm l1-norm l2-norm max-normalize mean-subtract log-softmax softmax; do
    echo "  build test-$kernel.exe"
    $NVCC $NVCCFLAGS -o "$BINDIR/test-$kernel.exe" "test-$kernel.cu" "$SOL/$kernel.cu"
done

echo "Built reference tests into $BINDIR/"
