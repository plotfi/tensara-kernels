NVCC        := nvcc
NVCCFLAGS   := -O2 -std=c++17
HARNESS_DIR := kernel-harnesses
BINDIR      := build/bin

# Every kernel is named by its harness; its implementation lives in solutions-cuda/ (CUDA) or solutions-metal/ (Metal).
KERNELS := $(patsubst $(HARNESS_DIR)/%.cu,%,$(wildcard $(HARNESS_DIR)/*.cu))

.PHONY: all clean help
all: $(KERNELS:%=$(BINDIR)/%.exe)

# Pattern rule — works for every harness.
#
#   Build a kernel:          make build/bin/matrix-multiplication.exe
#     (auto-links solutions-cuda/matrix-multiplication.cu — the stub or your code)
#   Point at another file:   make build/bin/matrix-multiplication.exe SOLUTION=my-solution.cu
#
# If SOLUTION isn't given, link solutions-cuda/<name>.cu when it exists.
SOLUTION ?=
$(BINDIR)/%.exe: $(HARNESS_DIR)/%.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(if $(SOLUTION),$(SOLUTION),$(wildcard solutions-cuda/$*.cu))

$(BINDIR):
	mkdir -p $(BINDIR)

# ---------------------------------------------------------------------------
# Metal backend (macOS only, requires metal-cpp headers)
#
#   make metal -j$(sysctl -n hw.ncpu)        # build all Metal kernels in parallel
#   make metal/vector-addition               # build one
#   TENSOR_SHADER_DIR=build/metal ./build/metal/vector-addition   # run
# ---------------------------------------------------------------------------

METAL_CPP   ?= metal-cpp
CXX         ?= clang++
METAL_DIR   := build/metal
METAL_CXX   := $(CXX) -std=c++17 -O2 -x c++ -DTENSOR_METAL -I tensor-lib -I kernel-implementation -I$(METAL_CPP)
METAL_LD    := -framework Metal -framework Foundation

# Kernels that have both a .cpp wrapper and a .metal shader.
METAL_KERNELS := $(foreach k,$(KERNELS),$(if $(and $(wildcard solutions-metal/$(k).cpp),$(wildcard solutions-metal/$(k).metal)),$(k)))

# Shared .metal.h headers that shaders #include at runtime.
METAL_HEADERS_SRC := $(wildcard kernel-implementation/*.metal.h)
METAL_HEADERS_DST := $(METAL_HEADERS_SRC:kernel-implementation/%.metal.h=$(METAL_DIR)/%.metal.h)

# Keep intermediate files — Make would delete them as intermediates otherwise.
.PRECIOUS: $(METAL_DIR)/%.metal $(METAL_DIR)/%.metal.h $(METAL_DIR)/%.tensor.o $(METAL_DIR)/%.wrapper.o

.PHONY: metal metal-clean
metal: $(METAL_KERNELS:%=$(METAL_DIR)/%)

# Stage shared shader headers next to the binaries.
$(METAL_DIR)/%.metal.h: kernel-implementation/%.metal.h | $(METAL_DIR)
	cp $< $@

# Copy shader source next to the binary for runtime compilation.
$(METAL_DIR)/%.metal: solutions-metal/%.metal | $(METAL_DIR)
	cp $< $@

# Compile the harness object (no metal-cpp implementation symbols).
$(METAL_DIR)/%.tensor.o: $(HARNESS_DIR)/%.cu tensor-lib/tensor.cuh tensor-lib/detail/tensor_metal.cuh tensor-lib/detail/tensor_common.cuh | $(METAL_DIR)
	$(METAL_CXX) -c $< -o $@

# Compile the wrapper object (carries the metal-cpp implementation).
$(METAL_DIR)/%.wrapper.o: solutions-metal/%.cpp tensor-lib/tensor.cuh tensor-lib/detail/tensor_metal.cuh tensor-lib/detail/tensor_common.cuh | $(METAL_DIR)
	$(METAL_CXX) -DTENSOR_METAL_IMPL -c $< -o $@

# Link (object files are kept for incremental rebuilds).
$(METAL_DIR)/%: $(METAL_DIR)/%.tensor.o $(METAL_DIR)/%.wrapper.o $(METAL_DIR)/%.metal $(METAL_HEADERS_DST)
	$(CXX) $< $(word 2,$^) -o $@ $(METAL_LD)

$(METAL_DIR):
	mkdir -p $(METAL_DIR)

# Shorthand: "make metal/relu" instead of "make build/metal/relu"
metal/%:
	$(MAKE) $(METAL_DIR)/$*

.PHONY: metal-test
metal-test: metal
	METAL_CPP=$(METAL_CPP) ./tests/run_tests_metal.sh

metal-clean:
	rm -rf $(METAL_DIR) build/metal-tests

clean:
	rm -rf build/

help:
	@echo "=== CUDA ==="
	@echo "Build every harness (implementation comes from solutions-cuda/<name>.cu):"
	@echo "  make            # or ./build_all.sh (also builds the tests)"
	@echo ""
	@echo "Build one kernel:"
	@echo "  make build/bin/matrix-multiplication.exe"
	@echo "  make build/bin/softmax.exe"
	@echo ""
	@echo "Use a different solution file for a kernel:"
	@echo "  make build/bin/matrix-multiplication.exe SOLUTION=my-solution.cu"
	@echo ""
	@echo "=== Metal (macOS) ==="
	@echo "Build all Metal kernels (parallel):"
	@echo "  make metal -j\$$(sysctl -n hw.ncpu)"
	@echo ""
	@echo "Build one Metal kernel:"
	@echo "  make metal/softmax"
	@echo ""
	@echo "Run a Metal kernel:"
	@echo "  TENSOR_SHADER_DIR=build/metal ./build/metal/softmax"
	@echo ""
	@echo "Set metal-cpp path (default: ../metal-kernels/metal-cpp/metal-cpp):"
	@echo "  make metal METAL_CPP=/path/to/metal-cpp"
	@echo ""
	@echo "=== General ==="
	@echo "  make clean      # remove all build artifacts"
