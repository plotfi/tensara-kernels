NVCC        := nvcc
NVCCFLAGS   := -O2 -std=c++17
HARNESS_DIR := tensara-harnesses
BINDIR      := build/bin

# Every kernel is named by its harness; its implementation lives in solutions/.
KERNELS := $(patsubst $(HARNESS_DIR)/%.cu,%,$(wildcard $(HARNESS_DIR)/*.cu))

.PHONY: all clean help
all: $(KERNELS:%=$(BINDIR)/%.exe)

# Pattern rule — works for every harness.
#
#   Build a kernel:          make build/bin/matrix-multiplication.exe
#     (auto-links solutions/matrix-multiplication.cu — the stub or your code)
#   Point at another file:   make build/bin/matrix-multiplication.exe SOLUTION=my-solution.cu
#
# If SOLUTION isn't given, link solutions/<name>.cu when it exists.
SOLUTION ?=
$(BINDIR)/%.exe: $(HARNESS_DIR)/%.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(if $(SOLUTION),$(SOLUTION),$(wildcard solutions/$*.cu))

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf build/

help:
	@echo "Build every harness (implementation comes from solutions/<name>.cu):"
	@echo "  make            # or ./build_all.sh (also builds the tests)"
	@echo ""
	@echo "Build one kernel:"
	@echo "  make build/bin/matrix-multiplication.exe"
	@echo "  make build/bin/softmax.exe"
	@echo ""
	@echo "Use a different solution file for a kernel:"
	@echo "  make build/bin/matrix-multiplication.exe SOLUTION=my-solution.cu"
	@echo ""
	@echo "Fill in a kernel by editing its solution:"
	@echo "  \$$EDITOR solutions/matrix-multiplication.cu"
	@echo ""
	@echo "Clean all built binaries:"
	@echo "  make clean"
