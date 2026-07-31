#!/bin/bash
# Build the Metal harnesses (macOS only; requires metal-cpp headers).
#
#   METAL_CPP=/path/to/metal-cpp ./build_metal.sh            # all kernels
#   METAL_CPP=/path/to/metal-cpp ./build_metal.sh relu ...   # specific kernels
#
# Each binary links the (backend-agnostic) harness .cu with its Metal solution
# wrapper solutions-metal/<kernel>.cpp; the shader solutions-metal/<kernel>.metal is copied
# next to the binary and loaded at runtime. Run with the shader dir on the env:
#   TENSOR_SHADER_DIR=build/metal ./build/metal/relu
set -e

METAL_CPP=${METAL_CPP:-metal-cpp}
CXX=${CXX:-clang++}
OUT=build/metal
mkdir -p "$OUT"

# Shared .metal headers (e.g. the unified activation header) live in
# kernel-implementation/ but must be staged next to the shaders so the harness's
# local-include inliner can resolve them at runtime.
cp kernel-implementation/*.metal.h "$OUT/" 2>/dev/null || true

CXXFLAGS="-std=c++17 -O2 -x c++ -DTENSOR_METAL -I tensor-lib -I kernel-implementation -I$METAL_CPP"
LDFLAGS="-framework Metal -framework Foundation"

kernels=("$@")
if [ ${#kernels[@]} -eq 0 ]; then
    for h in kernel-harnesses/*.cu; do kernels+=("$(basename "$h" .cu)"); done
fi

for k in "${kernels[@]}"; do
    if [ ! -f "solutions-metal/$k.cpp" ] || [ ! -f "solutions-metal/$k.metal" ]; then
        echo "skip $k (missing solutions-metal/$k.cpp or .metal)"; continue
    fi
    echo "  clang++ $k"
    # The wrapper is the one TU that carries the metal-cpp implementation.
    $CXX $CXXFLAGS -c "kernel-harnesses/$k.cu" -o "$OUT/$k.tensor.o"
    $CXX $CXXFLAGS -DTENSOR_METAL_IMPL -c "solutions-metal/$k.cpp" -o "$OUT/$k.wrapper.o"
    $CXX "$OUT/$k.tensor.o" "$OUT/$k.wrapper.o" -o "$OUT/$k" $LDFLAGS
    rm -f "$OUT/$k.tensor.o" "$OUT/$k.wrapper.o"
    cp "solutions-metal/$k.metal" "$OUT/"
done

echo "Built Metal harnesses into $OUT/"
echo "Run e.g.:  TENSOR_SHADER_DIR=$OUT ./$OUT/vector-addition"
