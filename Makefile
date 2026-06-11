NVCC        := nvcc
NVCCFLAGS   := -O2 -std=c++17
HARNESS_DIR := tensara-launch/harnesses
BINDIR      := build/bin

# Harnesses that include their own solution — compile with no extra files
SELF_CONTAINED := relu elu leaky-relu swish

.PHONY: all clean help $(SELF_CONTAINED)

all: $(SELF_CONTAINED:%=$(BINDIR)/%.exe)

# Pattern rule — works for every harness.
#
#   Self-contained:       make build/bin/relu.exe
#   With solution file:   make build/bin/matrix-multiplication.exe SOLUTION=my-solution.cu
#   With activation flag: make build/bin/relu.exe NVCCFLAGS='-O2 -std=c++17 -DACT_RELU'
#
$(BINDIR)/%.exe: $(HARNESS_DIR)/%.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SOLUTION)

$(BINDIR):
	mkdir -p $(BINDIR)

# Short aliases: `make relu` instead of `make build/bin/relu.exe`
$(SELF_CONTAINED): %: $(BINDIR)/%.exe

clean:
	rm -rf build/

help:
	@echo "Build all self-contained harnesses:"
	@echo "  make"
	@echo ""
	@echo "Build a specific self-contained harness:"
	@echo "  make build/bin/relu.exe"
	@echo "  make build/bin/elu.exe"
	@echo "  make build/bin/leaky-relu.exe"
	@echo "  make build/bin/swish.exe"
	@echo ""
	@echo "Select activation in activations.cu at compile time:"
	@echo "  make build/bin/relu.exe     NVCCFLAGS='-O2 -std=c++17 -DACT_RELU'"
	@echo "  make build/bin/gelu.exe     NVCCFLAGS='-O2 -std=c++17 -DACT_GELU'"
	@echo "  make build/bin/selu.exe     NVCCFLAGS='-O2 -std=c++17 -DACT_SELU'"
	@echo "  make build/bin/sigmoid.exe  NVCCFLAGS='-O2 -std=c++17 -DACT_SIGMOID'"
	@echo "  make build/bin/tanh.exe     NVCCFLAGS='-O2 -std=c++17 -DACT_TANH'"
	@echo "  make build/bin/softplus.exe NVCCFLAGS='-O2 -std=c++17 -DACT_SOFTPLUS'"
	@echo "  make build/bin/swish.exe    NVCCFLAGS='-O2 -std=c++17 -DACT_SWISH'"
	@echo ""
	@echo "Build any harness with your solution file:"
	@echo "  make build/bin/matrix-multiplication.exe SOLUTION=my-solution.cu"
	@echo "  make build/bin/softmax.exe               SOLUTION=my-solution.cu"
	@echo ""
	@echo "Clean all built binaries:"
	@echo "  make clean"
