#!/bin/bash
# Build the Metal harnesses (macOS only; requires metal-cpp headers).
#
#   METAL_CPP=/path/to/metal-cpp ./build_metal.sh            # all kernels
#   METAL_CPP=/path/to/metal-cpp ./build_metal.sh relu ...   # specific kernels
#
# Each binary links the (backend-agnostic) harness .cu with its Metal solution
# wrapper solutions/<kernel>.cpp; the shader solutions/<kernel>.metal is copied
# next to the binary and loaded at runtime. Run with the shader dir on the env:
#   HARNESS_SHADER_DIR=build/metal ./build/metal/relu
set -e

METAL_CPP=${METAL_CPP:-../metal-kernels/metal-cpp/metal-cpp}
CXX=${CXX:-clang++}
OUT=build/metal
mkdir -p "$OUT"

CXXFLAGS="-std=c++17 -O2 -x c++ -DHARNESS_METAL -I kernel-implementation -I$METAL_CPP"
LDFLAGS="-framework Metal -framework Foundation"

kernels=("$@")
if [ ${#kernels[@]} -eq 0 ]; then
    for h in tensara-harnesses/*.cu; do kernels+=("$(basename "$h" .cu)"); done
fi

for k in "${kernels[@]}"; do
    if [ ! -f "solutions/$k.cpp" ] || [ ! -f "solutions/$k.metal" ]; then
        echo "skip $k (missing solutions/$k.cpp or .metal)"; continue
    fi
    echo "  clang++ $k"
    # The wrapper is the one TU that carries the metal-cpp implementation.
    $CXX $CXXFLAGS -c "tensara-harnesses/$k.cu" -o "$OUT/$k.harness.o"
    $CXX $CXXFLAGS -DHARNESS_METAL_IMPL -c "solutions/$k.cpp" -o "$OUT/$k.wrapper.o"
    $CXX "$OUT/$k.harness.o" "$OUT/$k.wrapper.o" -o "$OUT/$k" $LDFLAGS
    rm -f "$OUT/$k.harness.o" "$OUT/$k.wrapper.o"
    cp "solutions/$k.metal" "$OUT/"
done

echo "Built Metal harnesses into $OUT/"
echo "Run e.g.:  HARNESS_SHADER_DIR=$OUT ./$OUT/vector-addition"
