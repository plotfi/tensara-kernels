NVCC        := nvcc
NVCCFLAGS   := -O2 -std=c++17
HARNESS_DIR := tensara-launch/harnesses

# Harnesses that include their own solution — compile with no extra files
SELF_CONTAINED := relu elu leaky-relu swish

.PHONY: all clean help $(SELF_CONTAINED)

all: $(SELF_CONTAINED:%=%.exe)

# Pattern rule — works for every harness.
#
#   Self-contained:       make relu.exe
#   With solution file:   make matrix-multiplication.exe SOLUTION=my-solution.cu
#
%.exe: $(HARNESS_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SOLUTION)

# Short aliases: `make relu` instead of `make relu.exe`
$(SELF_CONTAINED): %: %.exe

clean:
	rm -f *.exe

help:
	@echo "Build all self-contained harnesses:"
	@echo "  make"
	@echo ""
	@echo "Build a specific self-contained harness:"
	@echo "  make relu.exe"
	@echo "  make elu.exe"
	@echo "  make leaky-relu.exe"
	@echo "  make swish.exe"
	@echo ""
	@echo "Override the activation in activations.cu at compile time:"
	@echo "  make swish.exe NVCCFLAGS='-O2 -std=c++17 -DACT_RELU'"
	@echo "  Flags: -DACT_RELU  -DACT_LEAKY_RELU  -DACT_ELU   -DACT_GELU"
	@echo "         -DACT_SELU  -DACT_SIGMOID      -DACT_TANH  -DACT_SOFTPLUS"
	@echo "         -DACT_SWISH (default)"
	@echo ""
	@echo "Build any harness with your solution file:"
	@echo "  make matrix-multiplication.exe SOLUTION=my-solution.cu"
	@echo "  make softmax.exe              SOLUTION=my-solution.cu"
	@echo ""
	@echo "Clean all built binaries:"
	@echo "  make clean"
