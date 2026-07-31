NVCC      := nvcc
NVCCFLAGS := -O2 -std=c++17
BINDIR    := build/bin
OBJDIR    := build/obj
SOLUTION  ?=

# Solution linked into a spec-test binary: an explicit SOLUTION= if given, else
# the fill-in stub solutions-cuda/<kernel>.cu, with <kernel> derived from the target
# name (test-<kernel>.exe). Empty when neither exists.
sol = $(if $(SOLUTION),$(SOLUTION),$(wildcard ../solutions-cuda/$(patsubst test-%,%,$(notdir $(basename $@))).cu))

# Every spec-test binary (kernels whose harness builds against an external
# solution). `make spec` builds them all against the solutions-cuda/ stubs.
SPEC_TESTS := \
  test-matrix-multiplication test-square-matmul test-matmul-3d test-matmul-4d \
  test-matrix-scalar test-matrix-power test-diagonal-matmul test-symmetric-matmul \
  test-gemm-relu test-gemm-multiply-leakyrelu test-matmul-swish \
  test-matmul-swish-scaling test-matmul-sigmoid-sum test-conv-2d test-conv-square-3d \
  test-conv2d-relu-hardswish test-grayscale test-threshold test-box-blur \
  test-edge-detect test-cumsum test-cumprod test-running-sum-1d test-array-sort \
  test-histogram test-frobenius-norm test-cosine-similarity test-layer-norm \
  test-batch-norm test-mse-loss test-triplet-margin test-scaled-dot-attention \
  test-all-pairs-shortest-path test-shortest-path test-min-spanning-tree \
  test-ecc-point-negation \
  test-sum-dim test-mean-dim test-max-dim test-min-dim test-product-dim \
  test-argmax test-argmin test-huber-loss test-hinge-loss test-kl-loss \
  test-upper-trig-matmul test-lower-trig-matmul \
  test-avg-pool-1d test-avg-pool-2d test-avg-pool-3d \
  test-max-pool-1d test-max-pool-2d test-max-pool-3d

# ----------------------------------------------------------------------------
# Spec tests build against YOUR implementation, passed via SOLUTION=<file>, e.g.
#   make build/bin/test-softmax.exe SOLUTION=../my-softmax.cu
#   ./build/bin/test-softmax.exe
# Grouped sources (dim-reduce, argreduce, loss-reduce, trig-matmul, avg-pool,
# max-pool) are compiled once per kernel with a -D flag; explicit rules below.
# ----------------------------------------------------------------------------

# Simple 1-source-per-kernel spec tests.
$(BINDIR)/test-%.exe: test-%.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(sol)

# Grouped: dimension reductions
$(BINDIR)/test-sum-dim.exe:     dim-reduce.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DOP_SUM  -o $@ $< $(sol)
$(BINDIR)/test-mean-dim.exe:    dim-reduce.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DOP_MEAN -o $@ $< $(sol)
$(BINDIR)/test-max-dim.exe:     dim-reduce.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DOP_MAX  -o $@ $< $(sol)
$(BINDIR)/test-min-dim.exe:     dim-reduce.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DOP_MIN  -o $@ $< $(sol)
$(BINDIR)/test-product-dim.exe: dim-reduce.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DOP_PROD -o $@ $< $(sol)

# Grouped: argmax / argmin
$(BINDIR)/test-argmax.exe: argreduce.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS)          -o $@ $< $(sol)
$(BINDIR)/test-argmin.exe: argreduce.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DARG_MIN -o $@ $< $(sol)

# Grouped: scalar 1D losses
$(BINDIR)/test-huber-loss.exe: loss-reduce.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DL_HUBER -o $@ $< $(sol)
$(BINDIR)/test-hinge-loss.exe: loss-reduce.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DL_HINGE -o $@ $< $(sol)
$(BINDIR)/test-kl-loss.exe:    loss-reduce.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DL_KL    -o $@ $< $(sol)

# Grouped: triangular matmul
$(BINDIR)/test-upper-trig-matmul.exe: trig-matmul.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DTRIG_UPPER -o $@ $< $(sol)
$(BINDIR)/test-lower-trig-matmul.exe: trig-matmul.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DTRIG_LOWER -o $@ $< $(sol)

# Grouped: average pooling
$(BINDIR)/test-avg-pool-1d.exe: avg-pool.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DPOOL_DIM=1 -o $@ $< $(sol)
$(BINDIR)/test-avg-pool-2d.exe: avg-pool.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DPOOL_DIM=2 -o $@ $< $(sol)
$(BINDIR)/test-avg-pool-3d.exe: avg-pool.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DPOOL_DIM=3 -o $@ $< $(sol)

# Grouped: max pooling
$(BINDIR)/test-max-pool-1d.exe: max-pool.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DPOOL_DIM=1 -o $@ $< $(sol)
$(BINDIR)/test-max-pool-2d.exe: max-pool.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DPOOL_DIM=2 -o $@ $< $(sol)
$(BINDIR)/test-max-pool-3d.exe: max-pool.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DPOOL_DIM=3 -o $@ $< $(sol)

# Activation tests: one source (test-activation.cu) per -DACT_* flag, each linked
# against its solutions-cuda/<activation>.cu (via $(sol), derived from the target).
$(BINDIR)/test-relu.exe:         test-activation.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DACT_RELU        -o $@ $< $(sol)
$(BINDIR)/test-elu.exe:          test-activation.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DACT_ELU         -o $@ $< $(sol)
$(BINDIR)/test-leaky-relu.exe:   test-activation.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DACT_LEAKY_RELU  -o $@ $< $(sol)
$(BINDIR)/test-swish.exe:        test-activation.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DACT_SWISH       -o $@ $< $(sol)
$(BINDIR)/test-gelu.exe:         test-activation.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DACT_GELU        -o $@ $< $(sol)
$(BINDIR)/test-selu.exe:         test-activation.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DACT_SELU        -o $@ $< $(sol)
$(BINDIR)/test-sigmoid.exe:      test-activation.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DACT_SIGMOID     -o $@ $< $(sol)
$(BINDIR)/test-soft-plus.exe:    test-activation.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DACT_SOFTPLUS    -o $@ $< $(sol)
$(BINDIR)/test-tanh.exe:         test-activation.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DACT_TANH        -o $@ $< $(sol)
$(BINDIR)/test-hard-sigmoid.exe: test-activation.cu FORCE | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -DACT_HARD_SIGMOID -o $@ $< $(sol)

$(BINDIR) $(OBJDIR):
	mkdir -p $@

# ----------------------------------------------------------------------------
# compile-check: compile every spec test translation unit WITHOUT linking (so no
# SOLUTION is needed). Verifies signatures and reference code compile. Requires
# nvcc; does not require a GPU.
# ----------------------------------------------------------------------------
# Force spec-test binaries to relink every invocation, since SOLUTION is a
# variable (not a file prerequisite) and switching solutions must take effect.
.PHONY: FORCE
FORCE:

# Tests for the kernels implemented in solutions-cuda/ (activations, reductions, ...).
REF_TESTS := \
  test-relu test-elu test-leaky-relu test-swish test-gelu test-selu \
  test-sigmoid test-soft-plus test-tanh test-hard-sigmoid \
  test-vector-addition test-matrix-vector test-conv-1d \
  test-rms-norm test-l1-norm test-l2-norm test-max-normalize test-mean-subtract \
  test-log-softmax test-softmax

# Build every spec test, each linked against its solutions-cuda/<kernel>.cu stub
# (or your own file via SOLUTION=). Does not implement anything.
.PHONY: spec
spec: $(SPEC_TESTS:%=$(BINDIR)/%.exe)
	@echo "Built $(words $(SPEC_TESTS)) spec tests into $(BINDIR)/"

# Build every test (reference + spec). Run in parallel with `make all-tests -jN`.
.PHONY: all-tests
all-tests: $(REF_TESTS:%=$(BINDIR)/%.exe) $(SPEC_TESTS:%=$(BINDIR)/%.exe)
	@echo "Built $(words $(REF_TESTS) $(SPEC_TESTS)) tests into $(BINDIR)/"

.PHONY: compile-check clean
compile-check: | $(OBJDIR)
	@set -e; \
	for f in $(wildcard test-*.cu); do \
		echo "  cc $$f"; $(NVCC) $(NVCCFLAGS) -c $$f -o $(OBJDIR)/$${f%.cu}.o; \
	done; \
	echo "  cc dim-reduce.cu (x5)";  for d in OP_SUM OP_MEAN OP_MAX OP_MIN OP_PROD; do $(NVCC) $(NVCCFLAGS) -D$$d -c dim-reduce.cu -o $(OBJDIR)/dim-$$d.o; done; \
	echo "  cc argreduce.cu (x2)";   $(NVCC) $(NVCCFLAGS) -c argreduce.cu -o $(OBJDIR)/arg-max.o; $(NVCC) $(NVCCFLAGS) -DARG_MIN -c argreduce.cu -o $(OBJDIR)/arg-min.o; \
	echo "  cc loss-reduce.cu (x3)"; for d in L_HUBER L_HINGE L_KL; do $(NVCC) $(NVCCFLAGS) -D$$d -c loss-reduce.cu -o $(OBJDIR)/loss-$$d.o; done; \
	echo "  cc trig-matmul.cu (x2)"; for d in TRIG_UPPER TRIG_LOWER; do $(NVCC) $(NVCCFLAGS) -D$$d -c trig-matmul.cu -o $(OBJDIR)/trig-$$d.o; done; \
	echo "  cc avg-pool.cu (x3)";    for d in 1 2 3; do $(NVCC) $(NVCCFLAGS) -DPOOL_DIM=$$d -c avg-pool.cu -o $(OBJDIR)/avg-$$d.o; done; \
	echo "  cc max-pool.cu (x3)";    for d in 1 2 3; do $(NVCC) $(NVCCFLAGS) -DPOOL_DIM=$$d -c max-pool.cu -o $(OBJDIR)/max-$$d.o; done; \
	echo "All spec tests compile."

clean:
	rm -rf build/
