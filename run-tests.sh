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
#   ./run-tests.sh -T [kernel]     # TRITON: check solutions-triton/*.py (--check vs torch ref)
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
TRITON=0
ACTION=run
PYVENV=.venv/bin/python

usage() { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ---- args ------------------------------------------------------------------
KERNEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)  usage 0 ;;
        -l|--list)  ACTION=list ;;
        -b|--build-only) BUILD_ONLY=1 ;;
        -c|--clean) CLEAN=1 ;;
        -T|--triton) TRITON=1 ;;
        -a|--arch)  ARCH="$2"; shift ;;
        -*)         echo "unknown option: $1" >&2; usage 1 ;;
        *)          KERNEL="$1" ;;
    esac
    shift
done

# ---- Triton backend: check solutions-triton/*.py against a torch reference ---
if [[ $TRITON -eq 1 ]]; then
    [[ -x "$PYVENV" ]] || { echo "error: $PYVENV not found — create the venv (see README Triton section)" >&2; exit 1; }
    mapfile -t TKS < <(for f in solutions-triton/*.py; do [[ -e "$f" ]] && basename "$f" .py; done | sort)
    if [[ "$ACTION" == list ]]; then
        echo ">> Triton solutions:"; printf '  %s\n' "${TKS[@]}"; exit 0
    fi
    [[ -n "$KERNEL" ]] && TKS=("$KERNEL")
    pass=0; fail=0
    for k in "${TKS[@]}"; do
        [[ -f "solutions-triton/$k.py" ]] || { echo "  no Triton solution: $k"; fail=$((fail+1)); continue; }
        if "$PYVENV" "solutions-triton/$k.py" --check >/tmp/tt.$$ 2>/dev/null && grep -q "^PASS:" /tmp/tt.$$; then
            echo "  PASS  $k"; pass=$((pass+1))
        else
            echo "  FAIL  $k"; fail=$((fail+1))
        fi
    done
    rm -f /tmp/tt.$$
    echo ">> triton: $pass passed, $fail failed"
    exit $(( fail > 0 ? 1 : 0 ))
fi

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
    echo ">> building test-$KERNEL"
    cmake --build "$BUILD" --target "test-$KERNEL"
    # Also build the benchmark harness so a single-kernel run shows its timing.
    if [[ -f "kernel-harnesses/$KERNEL.cu" ]]; then
        echo ">> building harness $KERNEL"
        cmake --build "$BUILD" --target "$KERNEL"
    fi
else
    echo ">> building all tests"
    cmake --build "$BUILD" --target tests
fi

[[ $BUILD_ONLY -eq 1 ]] && { echo ">> build-only; not running."; exit 0; }

# ---- run -------------------------------------------------------------------
if [[ -n "$KERNEL" ]]; then
    echo ">> running test $KERNEL"
    rc=0
    ctest --test-dir "$BUILD" -R "^${KERNEL}\$" --output-on-failure || rc=$?
    # Then the benchmark harness output (Avg kernel time + preview).
    if [[ -x "$BUILD/bin/$KERNEL.exe" ]]; then
        echo
        echo ">> benchmark:"
        "./$BUILD/bin/$KERNEL.exe" || true
    fi
    exit $rc
else
    echo ">> running suite"
    ctest --test-dir "$BUILD" -j --output-on-failure
fi
