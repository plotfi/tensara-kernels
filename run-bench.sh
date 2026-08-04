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
#   ./run-bench.sh -A              # in the all-run, also run unimplemented stubs
#   ./run-bench.sh -a 89 ...       # set CMAKE_CUDA_ARCHITECTURES (default: native)
#   ./run-bench.sh -c ...          # clean-reconfigure first (rm -rf build)
#
# Size knobs (read by the harnesses; just pass them in the environment):
#   TENSOR_SCALE=<k>   multiply every input size dimension by k
#   TENSOR_<DIM>=<n>   set one dim absolutely (TENSOR_N, TENSOR_M, TENSOR_K, ...)
#   e.g.  TENSOR_SCALE=16384 ./run-bench.sh vector-addition   # bandwidth-bound
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
ACTION=run

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

is_stub() { grep -q "// TODO: implement" "solutions-cuda/$1.cu" 2>/dev/null; }

# ---- args ------------------------------------------------------------------
KERNEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)  usage 0 ;;
        -l|--list)  ACTION=list ;;
        -b|--build-only) BUILD_ONLY=1 ;;
        -c|--clean) CLEAN=1 ;;
        -A|--all-stubs) INCLUDE_STUBS=1 ;;
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
    echo ">> running $KERNEL"
    "./$BUILD/bin/$KERNEL.exe"
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
    out="$("$exe" 2>/dev/null || true)"
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
