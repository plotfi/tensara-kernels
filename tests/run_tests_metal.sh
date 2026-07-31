#!/bin/zsh
# Build and run every correctness test against the Metal solutions.
#
# Mirror of run_tests.sh, but compiles test sources as C++ with a fake
# cuda_runtime.h (tests/metal/cuda_runtime.h) backed by Metal shared memory.
# Each test links against solutions/<kernel>.cpp (the Metal wrapper).
#
# Usage:
#   METAL_CPP=/path/to/metal-cpp ./tests/run_tests_metal.sh           # all
#   METAL_CPP=/path/to/metal-cpp ./tests/run_tests_metal.sh relu      # one
#
# Prerequisites: build shaders first with `make metal`.

set -e
cd "${0:A:h}/.."

METAL_CPP=${METAL_CPP:-metal-cpp}
CXX=${CXX:-clang++}
BINDIR=build/metal-tests
SHADER_DIR=build/metal
mkdir -p "$BINDIR"

# -I tests/metal comes first so the fake cuda_runtime.h is found before any
# system CUDA (there is none on macOS, but this makes intent explicit).
CXXFLAGS=(-std=c++17 -O2 -x c++ -DHARNESS_METAL -I tests/metal -I kernel-implementation -I"$METAL_CPP")
LDFLAGS=(-framework Metal -framework Foundation)

if [[ ! -d "$SHADER_DIR" ]] || [[ -z "$(ls "$SHADER_DIR"/*.metal 2>/dev/null)" ]]; then
    echo "error: shader dir $SHADER_DIR is empty — run 'make metal' first." >&2
    exit 1
fi

typeset -A TESTS
TESTS=(
  relu           "test-activation.cu|-DACT_RELU"
  elu            "test-activation.cu|-DACT_ELU"
  leaky-relu     "test-activation.cu|-DACT_LEAKY_RELU"
  swish          "test-activation.cu|-DACT_SWISH"
  gelu           "test-activation.cu|-DACT_GELU"
  selu           "test-activation.cu|-DACT_SELU"
  sigmoid        "test-activation.cu|-DACT_SIGMOID"
  soft-plus      "test-activation.cu|-DACT_SOFTPLUS"
  tanh           "test-activation.cu|-DACT_TANH"
  hard-sigmoid   "test-activation.cu|-DACT_HARD_SIGMOID"
  vector-addition   "test-vector-addition.cu|"
  matrix-vector     "test-matrix-vector.cu|"
  conv-1d           "test-conv-1d.cu|"
  rms-norm          "test-rms-norm.cu|"
  l1-norm           "test-l1-norm.cu|"
  l2-norm           "test-l2-norm.cu|"
  max-normalize     "test-max-normalize.cu|"
  mean-subtract     "test-mean-subtract.cu|"
  log-softmax       "test-log-softmax.cu|"
  softmax           "test-softmax.cu|"
  matrix-multiplication  "test-matrix-multiplication.cu|"
  square-matmul     "test-square-matmul.cu|"
  matmul-3d         "test-matmul-3d.cu|"
  matmul-4d         "test-matmul-4d.cu|"
  matrix-scalar     "test-matrix-scalar.cu|"
  matrix-power      "test-matrix-power.cu|"
  diagonal-matmul   "test-diagonal-matmul.cu|"
  symmetric-matmul  "test-symmetric-matmul.cu|"
  gemm-relu         "test-gemm-relu.cu|"
  gemm-multiply-leakyrelu  "test-gemm-multiply-leakyrelu.cu|"
  matmul-swish      "test-matmul-swish.cu|"
  matmul-swish-scaling     "test-matmul-swish-scaling.cu|"
  matmul-sigmoid-sum       "test-matmul-sigmoid-sum.cu|"
  conv-2d           "test-conv-2d.cu|"
  conv-square-3d    "test-conv-square-3d.cu|"
  conv2d-relu-hardswish    "test-conv2d-relu-hardswish.cu|"
  grayscale         "test-grayscale.cu|"
  threshold         "test-threshold.cu|"
  box-blur          "test-box-blur.cu|"
  edge-detect       "test-edge-detect.cu|"
  cumsum            "test-cumsum.cu|"
  cumprod           "test-cumprod.cu|"
  running-sum-1d    "test-running-sum-1d.cu|"
  array-sort        "test-array-sort.cu|"
  histogram         "test-histogram.cu|"
  frobenius-norm    "test-frobenius-norm.cu|"
  cosine-similarity "test-cosine-similarity.cu|"
  layer-norm        "test-layer-norm.cu|"
  batch-norm        "test-batch-norm.cu|"
  mse-loss          "test-mse-loss.cu|"
  triplet-margin    "test-triplet-margin.cu|"
  scaled-dot-attention     "test-scaled-dot-attention.cu|"
  all-pairs-shortest-path  "test-all-pairs-shortest-path.cu|"
  shortest-path     "test-shortest-path.cu|"
  min-spanning-tree "test-min-spanning-tree.cu|"
  ecc-point-negation       "test-ecc-point-negation.cu|"
  sum-dim           "dim-reduce.cu|-DOP_SUM"
  mean-dim          "dim-reduce.cu|-DOP_MEAN"
  max-dim           "dim-reduce.cu|-DOP_MAX"
  min-dim           "dim-reduce.cu|-DOP_MIN"
  product-dim       "dim-reduce.cu|-DOP_PROD"
  argmax            "argreduce.cu|"
  argmin            "argreduce.cu|-DARG_MIN"
  huber-loss        "loss-reduce.cu|-DL_HUBER"
  hinge-loss        "loss-reduce.cu|-DL_HINGE"
  kl-loss           "loss-reduce.cu|-DL_KL"
  upper-trig-matmul "trig-matmul.cu|-DTRIG_UPPER"
  lower-trig-matmul "trig-matmul.cu|-DTRIG_LOWER"
  avg-pool-1d       "avg-pool.cu|-DPOOL_DIM=1"
  avg-pool-2d       "avg-pool.cu|-DPOOL_DIM=2"
  avg-pool-3d       "avg-pool.cu|-DPOOL_DIM=3"
  max-pool-1d       "max-pool.cu|-DPOOL_DIM=1"
  max-pool-2d       "max-pool.cu|-DPOOL_DIM=2"
  max-pool-3d       "max-pool.cu|-DPOOL_DIM=3"
)

UNCOVERED=(mxfp4-quantize mxfp4-dequantize mxfp4-gemm
           mxfp8-quantize mxfp8-dequantize mxfp8-gemm
           nvfp4-quantize nvfp4-dequantize nvfp4-gemm nvfp4-gemv
           poly-multiply-ff vector-multiply-ff)

# If specific kernels given on CLI, filter to just those.
typeset -A RUN_TESTS
if (( $# > 0 )); then
    for k in "$@"; do
        if (( ${+TESTS[$k]} )); then
            RUN_TESTS[$k]="${TESTS[$k]}"
        else
            echo "  SKIP  $k (no test defined)"
        fi
    done
else
    for k in ${(k)TESTS}; do RUN_TESTS[$k]="${TESTS[$k]}"; done
fi

pass=0; regression=0; build_fail=0; unimpl=0
regressions=(); build_fails=()

echo "=== Metal tests (each links solutions/<kernel>.cpp) ==="
for k in ${(ok)RUN_TESTS}; do
    entry="${RUN_TESTS[$k]}"
    src="${entry%%|*}"
    flags="${entry#*|}"
    sol="solutions/$k.cpp"

    # Skip kernels whose Metal shader is a stub (empty or contains TODO marker).
    if [[ ! -s "solutions/$k.metal" ]] || grep -q "// TODO: implement" "solutions/$k.metal" 2>/dev/null; then
        echo "  TODO  $k (stub shader)"; unimpl=$((unimpl+1)); continue
    fi

    # Compile test TU (no IMPL) + solution TU (with IMPL) → link.
    if ! $CXX "${CXXFLAGS[@]}" ${=flags} -c "tests/$src" -o "$BINDIR/test-$k.test.o" 2>&1; then
        echo "  BUILD-FAIL  $k (test TU)"; build_fail=$((build_fail+1)); build_fails+=("$k"); continue
    fi
    if ! $CXX "${CXXFLAGS[@]}" -DHARNESS_METAL_IMPL -c "$sol" -o "$BINDIR/test-$k.sol.o" 2>&1; then
        echo "  BUILD-FAIL  $k (solution TU)"; build_fail=$((build_fail+1)); build_fails+=("$k"); continue
    fi
    if ! $CXX "$BINDIR/test-$k.test.o" "$BINDIR/test-$k.sol.o" -o "$BINDIR/test-$k" "${LDFLAGS[@]}" 2>&1; then
        echo "  BUILD-FAIL  $k (link)"; build_fail=$((build_fail+1)); build_fails+=("$k"); continue
    fi

    if HARNESS_SHADER_DIR="$SHADER_DIR" "$BINDIR/test-$k" >/dev/null 2>&1; then
        echo "  PASS  $k"; pass=$((pass+1))
    else
        echo "  FAIL  $k (regression)"
        HARNESS_SHADER_DIR="$SHADER_DIR" "$BINDIR/test-$k" 2>&1 | sed 's/^/    /'
        regression=$((regression+1)); regressions+=("$k")
    fi
    rm -f "$BINDIR/test-$k.test.o" "$BINDIR/test-$k.sol.o"
done

echo ""
if (( $# == 0 )); then
    echo "=== uncovered (no test yet; needs exact spec) ==="
    for k in ${(o)UNCOVERED}; do echo "  ----  $k"; done
    echo ""
fi

total=$(( ${#RUN_TESTS} + ${#UNCOVERED} ))
(( $# > 0 )) && total=${#RUN_TESTS}

echo "=================================================="
echo "Kernels total:     $total"
echo "  passed:          $pass"
echo "  regressions:     $regression${regressions:+  (${regressions[*]})}"
echo "  build errors:    $build_fail${build_fails:+  (${build_fails[*]})}"
echo "  unimplemented:   $unimpl"
(( $# == 0 )) && echo "  uncovered:       ${#UNCOVERED}"
echo "=================================================="

exit $(( regression + build_fail ))
