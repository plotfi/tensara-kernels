#!/bin/bash
# Build every harness and every test, in parallel.
#
# Each kernel builds the same way: the harness links solutions-cuda/<name>.cu, which
# the Makefile picks up automatically. Implemented kernels have real code there;
# the rest link a fill-in stub (they compile and run, producing zeroed output,
# until you implement them).
#
# Override the job count with JOBS=, e.g. `JOBS=4 ./build_all.sh`.
set -e

JOBS=${JOBS:-$(nproc)}
mkdir -p build/bin

# Generate the target list: one build/bin/<name>.exe per harness.
targets=()
for h in kernel-harnesses/*.cu; do
    targets+=("build/bin/$(basename "$h" .cu).exe")
done

echo "Building ${#targets[@]} harnesses in parallel (-j$JOBS)..."
make -j"$JOBS" "${targets[@]}"
echo "Built all harnesses into build/bin/"

echo "Building tests in parallel (-j$JOBS)..."
make -C tests --no-print-directory -j"$JOBS" all-tests
echo "Built all tests into tests/build/bin/"
