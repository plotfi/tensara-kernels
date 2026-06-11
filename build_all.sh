#!/bin/bash

mkdir -p build/bin

make build/bin/elu.exe           NVCCFLAGS='-O2 -std=c++17 -DACT_ELU'
make build/bin/gelu.exe          NVCCFLAGS='-O2 -std=c++17 -DACT_GELU'
make build/bin/leaky-relu.exe    NVCCFLAGS='-O2 -std=c++17 -DACT_LEAKY_RELU'
make build/bin/relu.exe          NVCCFLAGS='-O2 -std=c++17 -DACT_RELU'
make build/bin/selu.exe          NVCCFLAGS='-O2 -std=c++17 -DACT_SELU'
make build/bin/sigmoid.exe       NVCCFLAGS='-O2 -std=c++17 -DACT_SIGMOID'
make build/bin/soft-plus.exe     NVCCFLAGS='-O2 -std=c++17 -DACT_SOFTPLUS'
make build/bin/swish.exe         NVCCFLAGS='-O2 -std=c++17 -DACT_SWISH'
make build/bin/tanh.exe          NVCCFLAGS='-O2 -std=c++17 -DACT_TANH'

# make build/bin/conv-1d.exe
# make build/bin/conv-2d.exe
# make build/bin/softmax.exe
# make build/bin/vector-addition.exe
# make build/bin/vector-multiply-ff.exe
# make build/bin/matrix-scalar.exe
# make build/bin/matrix-vector.exe

# make build/bin/all-pairs-shortest-path.exe
# make build/bin/argmax.exe
# make build/bin/argmin.exe
# make build/bin/array-sort.exe
# make build/bin/avg-pool-1d.exe
# make build/bin/avg-pool-2d.exe
# make build/bin/avg-pool-3d.exe
# make build/bin/batch-norm.exe
# make build/bin/box-blur.exe
# make build/bin/conv2d-relu-hardswish.exe
# make build/bin/conv-square-3d.exe
# make build/bin/cosine-similarity.exe
# make build/bin/cumprod.exe
# make build/bin/cumsum.exe
# make build/bin/diagonal-matmul.exe
# make build/bin/ecc-point-negation.exe
# make build/bin/edge-detect.exe
# make build/bin/frobenius-norm.exe
# make build/bin/gemm-multiply-leakyrelu.exe
# make build/bin/gemm-relu.exe
# make build/bin/grayscale.exe
# make build/bin/hard-sigmoid.exe
# make build/bin/hinge-loss.exe
# make build/bin/histogram.exe
# make build/bin/huber-loss.exe
# make build/bin/kl-loss.exe
# make build/bin/l1-norm.exe
# make build/bin/l2-norm.exe
# make build/bin/layer-norm.exe
# make build/bin/log-softmax.exe
# make build/bin/lower-trig-matmul.exe
# make build/bin/matmul-3d.exe
# make build/bin/matmul-4d.exe
# make build/bin/matmul-sigmoid-sum.exe
# make build/bin/matmul-swish.exe
# make build/bin/matmul-swish-scaling.exe
# make build/bin/matrix-multiplication.exe
# make build/bin/matrix-power.exe
# make build/bin/max-dim.exe
# make build/bin/max-pool-1d.exe
# make build/bin/max-pool-2d.exe
# make build/bin/max-pool-3d.exe
# make build/bin/mean-dim.exe
# make build/bin/min-dim.exe
# make build/bin/min-spanning-tree.exe
# make build/bin/mse-loss.exe
# make build/bin/mxfp4-dequantize.exe
# make build/bin/mxfp4-gemm.exe
# make build/bin/mxfp4-quantize.exe
# make build/bin/mxfp8-dequantize.exe
# make build/bin/mxfp8-gemm.exe
# make build/bin/mxfp8-quantize.exe
# make build/bin/nvfp4-dequantize.exe
# make build/bin/nvfp4-gemm.exe
# make build/bin/nvfp4-gemv.exe
# make build/bin/nvfp4-quantize.exe
# make build/bin/poly-multiply-ff.exe
# make build/bin/product-dim.exe
# make build/bin/rms-norm.exe
# make build/bin/running-sum-1d.exe
# make build/bin/scaled-dot-attention.exe
# make build/bin/shortest-path.exe
# make build/bin/square-matmul.exe
# make build/bin/sum-dim.exe
# make build/bin/symmetric-matmul.exe
# make build/bin/threshold.exe
# make build/bin/triplet-margin.exe
# make build/bin/upper-trig-matmul.exe
