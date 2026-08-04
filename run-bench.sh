#!/usr/bin/env bash
#
# run-bench.sh — build (via CMake + Ninja) and run the benchmark harnesses,
# either the whole set or a single kernel. Each harness prints its average
# kernel time (3-iteration warmup + 100 timed iterations).
#
# Usage:
#   ./run-bench.sh                 # build + run every implemented harness (timing table)
#   ./run-bench.sh huber-loss      # build + run one harness (full output)
#   ./run-bench.sh -l              # list harnesses (stub = not implemented)
#   ./run-bench.sh -b              # build the harnesses, don't run
#   ./run-bench.sh -b huber-loss   # build just that harness, don't run
#   ./run-bench.sh -B              # BENCHMARK MODE: per-kernel sizes that make each
#                                  #   kernel compute/bandwidth-bound (not launch-bound)
#   ./run-bench.sh -B relu         # benchmark-size a single kernel
#   ./run-bench.sh -A              # in the all-run, also run unimplemented stubs
#   ./run-bench.sh -a 89 ...       # set CMAKE_CUDA_ARCHITECTURES (default: native)
#   ./run-bench.sh -c ...          # clean-reconfigure first (rm -rf build)
#
# Size knobs (read by the harnesses; just pass them in the environment):
#   TENSOR_SCALE=<k>   multiply every input size dimension by k
#   TENSOR_<DIM>=<n>   set one dim absolutely (TENSOR_N, TENSOR_M, TENSOR_K, ...)
#   e.g.  TENSOR_SCALE=16384 ./run-bench.sh vector-addition   # bandwidth-bound
# With -B, run-bench picks a sensible size per kernel automatically; any TENSOR_*
# you set yourself still wins over the profile.
#
# Notes:
#   * Uses the Ninja generator; configures ./build once, reconfigures when
#     CMakeLists.txt changes.
#   * Unimplemented kernels (solutions-cuda/<k>.cu is still `// TODO: implement`)
#     run but do nothing, so the all-run skips them by default (-A includes them).
#   * These actually execute on the GPU, unlike the test build.
set -euo pipefail
cd "$(dirname "$0")"

BUILD=build
ARCH=""
BUILD_ONLY=0
CLEAN=0
INCLUDE_STUBS=0
BIG=0
ACTION=run

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

is_stub() { grep -q "// TODO: implement" "solutions-cuda/$1.cu" 2>/dev/null; }

# ---- benchmark-mode sizes (-B): env assignments per kernel -------------------
# Goal: push each kernel out of the launch-latency floor. Memory-bound kernels
# get tens of millions of elements; matmuls get ~2K dims (O(n^3)); images/conv
# get a few thousand per side. Setting a TENSOR_* a kernel ignores is harmless,
# so groups are generous. Structural params are never touched.
bench_profile() {
    case "$1" in
        # matmul / GEMM (O(n^3)) — keep dims ~2048
        matrix-multiplication|matmul-swish|matmul-swish-scaling|matmul-sigmoid-sum|\
        gemm-relu|gemm-multiply-leakyrelu|int8-weight-gemm|\
        mxfp4-gemm|mxfp8-gemm|nvfp4-gemm)
            echo "TENSOR_M=2048 TENSOR_N=2048 TENSOR_K=2048 TENSOR_B=16 TENSOR_BATCH_SIZE=2048 TENSOR_IN_FEATURES=2048 TENSOR_OUT_FEATURES=2048" ;;
        square-matmul|symmetric-matmul|upper-trig-matmul|lower-trig-matmul)
            echo "TENSOR_N=2048" ;;                          # N is the matrix side
        matmul-3d)  echo "TENSOR_N=32 TENSOR_M=1024 TENSOR_K=1024 TENSOR_L=1024" ;;
        matmul-4d)  echo "TENSOR_B=8 TENSOR_I=32 TENSOR_J=512 TENSOR_K=512 TENSOR_L=512" ;;
        matrix-vector|nvfp4-gemv) echo "TENSOR_M=16384 TENSOR_K=16384" ;;
        diagonal-matmul) echo "TENSOR_N=8192 TENSOR_M=8192" ;;
        matrix-power)   echo "TENSOR_SIZE=1024" ;;           # N is the exponent — leave it
        matrix-scalar)  echo "TENSOR_N=8192" ;;              # n*n elementwise
        # low-precision convert (M×K, memory-bound)
        mxfp4-quantize|mxfp4-dequantize|mxfp8-quantize|mxfp8-dequantize|\
        nvfp4-quantize|nvfp4-dequantize) echo "TENSOR_M=8192 TENSOR_K=8192" ;;
        # norms over rows×cols
        rms-norm)             echo "TENSOR_B=8192 TENSOR_N=8192" ;;
        layer-norm|batch-norm) echo "TENSOR_B=8192 TENSOR_F=8192" ;;
        l1-norm|l2-norm|max-normalize|mean-subtract) echo "TENSOR_B=8192 TENSOR_D=8192" ;;
        cosine-similarity)    echo "TENSOR_N=8192 TENSOR_D=8192" ;;
        frobenius-norm)       echo "TENSOR_SIZE=67108864" ;;
        # 2D elementwise activations (n×m)
        relu|elu|leaky-relu|swish|gelu|selu|sigmoid|soft-plus|tanh|hard-sigmoid)
            echo "TENSOR_N=8192 TENSOR_M=8192" ;;
        # images (keep CHANNELS/NUM_BINS as-is)
        threshold|grayscale|box-blur|edge-detect|histogram)
            echo "TENSOR_HEIGHT=4096 TENSOR_WIDTH=4096" ;;
        # conv
        conv-2d|conv2d-relu-hardswish) echo "TENSOR_H=4096 TENSOR_W=4096" ;;
        conv-1d|conv1d-maxpool1d)      echo "TENSOR_N=16777216" ;;
        conv-square-3d)                echo "TENSOR_SIZE=512" ;;
        # pooling
        avg-pool-1d|max-pool-1d) echo "TENSOR_H=16777216" ;;
        avg-pool-2d|max-pool-2d) echo "TENSOR_H=4096 TENSOR_W=4096" ;;
        avg-pool-3d|max-pool-3d) echo "TENSOR_H=512 TENSOR_W=512 TENSOR_D=512" ;;
        # attention
        scaled-dot-attention) echo "TENSOR_B=8 TENSOR_H=8 TENSOR_S=2048 TENSOR_E=64" ;;
        # softmax family (rows M × reduced dim N). dim-reduce kernels are stubs
        # and still size via a fixed shape array — they'll need the same wiring.
        softmax|log-softmax) echo "TENSOR_M=65536 TENSOR_N=1024" ;;
        # distances / margin losses over rows×cols
        triplet-margin) echo "TENSOR_B=8192 TENSOR_E=8192" ;;
        # graphs (O(n^2)/O(n^3)) — keep moderate
        all-pairs-shortest-path) echo "TENSOR_N=2048" ;;
        shortest-path|min-spanning-tree) echo "TENSOR_N=8192" ;;
        # default: 1-D bandwidth, 64M elements
        *) echo "TENSOR_N=67108864" ;;
    esac
}

# Run a harness, applying the -B profile in a subshell (user's env wins).
run_exe() {
    local k="$1"
    (
        if [[ $BIG -eq 1 ]]; then
            for kv in $(bench_profile "$k"); do
                local var="${kv%%=*}"
                [[ -z "${!var:-}" ]] && export "$kv"
            done
        fi
        exec "./$BUILD/bin/$k.exe"
    )
}

# ---- args ------------------------------------------------------------------
KERNEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)  usage 0 ;;
        -l|--list)  ACTION=list ;;
        -b|--build-only) BUILD_ONLY=1 ;;
        -c|--clean) CLEAN=1 ;;
        -A|--all-stubs) INCLUDE_STUBS=1 ;;
        -B|--big)   BIG=1 ;;
        -a|--arch)  ARCH="$2"; shift ;;
        -*)         echo "unknown option: $1" >&2; usage 1 ;;
        *)          KERNEL="$1" ;;
    esac
    shift
done

command -v ninja >/dev/null || { echo "error: ninja not found (install ninja-build)" >&2; exit 1; }

# ---- configure (Ninja) -----------------------------------------------------
[[ $CLEAN -eq 1 ]] && rm -rf "$BUILD"
if [[ -d "$BUILD" && ! -f "$BUILD/build.ninja" ]]; then
    echo ">> existing non-Ninja build/ — reconfiguring clean"
    rm -rf "$BUILD"
fi
if [[ ! -f "$BUILD/build.ninja" ]]; then
    cfg=(-S . -B "$BUILD" -G Ninja)
    [[ -n "$ARCH" ]] && cfg+=("-DCMAKE_CUDA_ARCHITECTURES=$ARCH")
    echo ">> cmake ${cfg[*]}"
    cmake "${cfg[@]}"
fi

# All harness names, sorted.
mapfile -t KERNELS < <(for f in kernel-harnesses/*.cu; do basename "$f" .cu; done | sort)

# ---- list mode -------------------------------------------------------------
if [[ "$ACTION" == list ]]; then
    echo ">> harnesses (stub = solutions-cuda/<k>.cu not implemented):"
    for k in "${KERNELS[@]}"; do
        if is_stub "$k"; then printf '  %-28s (stub)\n' "$k"; else printf '  %s\n' "$k"; fi
    done
    exit 0
fi

# ---- single kernel ---------------------------------------------------------
if [[ -n "$KERNEL" ]]; then
    echo ">> building $KERNEL"
    cmake --build "$BUILD" --target "$KERNEL"
    [[ $BUILD_ONLY -eq 1 ]] && { echo ">> build-only; not running."; exit 0; }
    is_stub "$KERNEL" && echo ">> note: $KERNEL is a stub (unimplemented) — timing is not meaningful"
    [[ $BIG -eq 1 ]] && echo ">> benchmark sizes: $(bench_profile "$KERNEL")"
    echo ">> running $KERNEL"
    run_exe "$KERNEL"
    exit 0
fi

# ---- all harnesses ---------------------------------------------------------
echo ">> building all harnesses"
cmake --build "$BUILD" --target harnesses
[[ $BUILD_ONLY -eq 1 ]] && { echo ">> build-only; not running."; exit 0; }

echo ">> running harnesses (this executes on the GPU)"
rows=""; ran=0; skipped=0; failed=0
for k in "${KERNELS[@]}"; do
    if [[ $INCLUDE_STUBS -eq 0 ]] && is_stub "$k"; then skipped=$((skipped+1)); continue; fi
    exe="./$BUILD/bin/$k.exe"
    [[ -x "$exe" ]] || { printf '  %-28s (no binary)\n' "$k"; failed=$((failed+1)); continue; }
    out="$(run_exe "$k" 2>/dev/null || true)"
    ms="$(printf '%s' "$out" | grep -oiE 'Avg kernel time:[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+' | head -1)"
    if [[ -n "$ms" ]]; then
        rows+="$ms|$k"$'\n'; ran=$((ran+1))
    else
        printf '  %-28s (no timing / crashed)\n' "$k"; failed=$((failed+1))
    fi
done

echo
printf '  %-28s %12s\n' "KERNEL" "AVG (ms)"
printf '  %-28s %12s\n' "----------------------------" "------------"
# Sort slowest-first.
printf '%s' "$rows" | sort -t'|' -k1 -rn | while IFS='|' read -r ms k; do
    [[ -z "$k" ]] && continue
    printf '  %-28s %12s\n' "$k" "$ms"
done
echo
echo ">> ran $ran, skipped $skipped stub(s), $failed failed/no-timing"
