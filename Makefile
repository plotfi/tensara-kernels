NVCC        := nvcc
NVCCFLAGS   := -O2 -std=c++17
HARNESS_DIR := tensara-harnesses
BINDIR      := build/bin

ACTIVATIONS := relu elu leaky-relu swish gelu selu sigmoid soft-plus tanh

.PHONY: all clean help $(ACTIVATIONS)

all: $(ACTIVATIONS:%=$(BINDIR)/%.exe)

# Per-target activation flags
$(BINDIR)/relu.exe:       NVCCFLAGS += -DACT_RELU
$(BINDIR)/elu.exe:        NVCCFLAGS += -DACT_ELU
$(BINDIR)/leaky-relu.exe: NVCCFLAGS += -DACT_LEAKY_RELU
$(BINDIR)/swish.exe:      NVCCFLAGS += -DACT_SWISH
$(BINDIR)/gelu.exe:       NVCCFLAGS += -DACT_GELU
$(BINDIR)/selu.exe:       NVCCFLAGS += -DACT_SELU
$(BINDIR)/sigmoid.exe:    NVCCFLAGS += -DACT_SIGMOID
$(BINDIR)/soft-plus.exe:  NVCCFLAGS += -DACT_SOFTPLUS
$(BINDIR)/tanh.exe:       NVCCFLAGS += -DACT_TANH

# Pattern rule — works for every harness.
#
#   Activation:           make build/bin/relu.exe
#   With solution file:   make build/bin/matrix-multiplication.exe SOLUTION=my-solution.cu
#
$(BINDIR)/%.exe: $(HARNESS_DIR)/%.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SOLUTION)

$(BINDIR):
	mkdir -p $(BINDIR)

# Short aliases: `make relu` instead of `make build/bin/relu.exe`
$(ACTIVATIONS): %: $(BINDIR)/%.exe

clean:
	rm -rf build/

help:
	@echo "Build all activation harnesses:"
	@echo "  make"
	@echo ""
	@echo "Build a specific activation harness:"
	@echo "  make build/bin/relu.exe"
	@echo "  make build/bin/elu.exe"
	@echo "  make build/bin/leaky-relu.exe"
	@echo "  make build/bin/gelu.exe"
	@echo "  make build/bin/selu.exe"
	@echo "  make build/bin/sigmoid.exe"
	@echo "  make build/bin/soft-plus.exe"
	@echo "  make build/bin/swish.exe"
	@echo "  make build/bin/tanh.exe"
	@echo ""
	@echo "Build any harness with your solution file:"
	@echo "  make build/bin/matrix-multiplication.exe SOLUTION=my-solution.cu"
	@echo "  make build/bin/softmax.exe               SOLUTION=my-solution.cu"
	@echo ""
	@echo "Clean all built binaries:"
	@echo "  make clean"
