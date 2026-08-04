#!/usr/bin/env bash
#
# run-tests.sh — build (via CMake + Ninja) and run the CUDA correctness tests,
# either the whole suite or a single kernel.
#
# Usage:
#   ./run-tests.sh                 # build + run the whole suite
#   ./run-tests.sh huber-loss      # build + run just one kernel's test
#   ./run-tests.sh -l              # list available test names
#   ./run-tests.sh -b              # build the tests, don't run them
#   ./run-tests.sh -b huber-loss   # build just that test, don't run it
#   ./run-tests.sh -a 89 ...       # set CMAKE_CUDA_ARCHITECTURES (default: native)
#   ./run-tests.sh -c ...          # clean-reconfigure first (rm -rf build)
#
# Notes:
#   * Uses the Ninja generator. Configures once into ./build; reconfigures
#     automatically when CMakeLists.txt changes.
#   * Stub-backed tests (// TODO: implement) are DISABLED, so a green run can
#     still have unimplemented kernels — that's expected.
set -euo pipefail
cd "$(dirname "$0")"

BUILD=build
ARCH=""
BUILD_ONLY=0
CLEAN=0

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ---- args ------------------------------------------------------------------
KERNEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)  usage 0 ;;
        -l|--list)  ACTION=list ;;
        -b|--build-only) BUILD_ONLY=1 ;;
        -c|--clean) CLEAN=1 ;;
        -a|--arch)  ARCH="$2"; shift ;;
        -*)         echo "unknown option: $1" >&2; usage 1 ;;
        *)          KERNEL="$1" ;;
    esac
    shift
done

command -v ninja >/dev/null || { echo "error: ninja not found (install ninja-build)" >&2; exit 1; }

# ---- configure (Ninja) -----------------------------------------------------
[[ $CLEAN -eq 1 ]] && rm -rf "$BUILD"
# If build/ exists but isn't a Ninja build (e.g. a Make generator was used),
# wipe it so we don't fight generators.
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

# ---- list mode -------------------------------------------------------------
if [[ "${ACTION:-}" == list ]]; then
    echo ">> registered tests (Disabled = stub solution):"
    ctest --test-dir "$BUILD" -N 2>/dev/null | grep -E '^[[:space:]]+Test #' | sed 's/^[[:space:]]*/  /'
    exit 0
fi

# ---- build -----------------------------------------------------------------
if [[ -n "$KERNEL" ]]; then
    target="test-$KERNEL"
    echo ">> building $target"
    cmake --build "$BUILD" --target "$target"
else
    echo ">> building all tests"
    cmake --build "$BUILD" --target tests
fi

[[ $BUILD_ONLY -eq 1 ]] && { echo ">> build-only; not running."; exit 0; }

# ---- run -------------------------------------------------------------------
if [[ -n "$KERNEL" ]]; then
    echo ">> running $KERNEL"
    ctest --test-dir "$BUILD" -R "^${KERNEL}\$" --output-on-failure
else
    echo ">> running suite"
    ctest --test-dir "$BUILD" -j --output-on-failure
fi
