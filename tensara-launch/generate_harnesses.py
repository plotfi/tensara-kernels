#!/usr/bin/env python3
"""Generate host-side CUDA harness (main.cu) for every Tensara problem."""

import os, ast, json

PROBLEMS_DIR = "tensara-problems/problems"
OUTPUT_DIR = "harnesses"

TYPE_MAP = {
    "float": "float",
    "int": "int",
    "size_t": "size_t",
    "uint8_t": "uint8_t",
    "uint32_t": "uint32_t",
    "uint64_t": "uint64_t",
    "float8": "__nv_fp8_e4m3",
    "float16": "half",
}

INCLUDES_FOR_TYPE = {
    "float8": "#include <cuda_fp8.h>",
    "float16": "#include <cuda_fp16.h>",
}

RAND_EXPR = {
    "float": "static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f",
    "int": "rand() % 201 - 100",
    "uint8_t": "static_cast<uint8_t>(rand() % 256)",
    "uint32_t": "static_cast<uint32_t>(rand())",
    "uint64_t": "static_cast<uint64_t>(rand())",
    "float8": "static_cast<__nv_fp8_e4m3>(0)",
    "float16": "__float2half(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f)",
}

PRINT_FMT = {
    "float": '"%f "',
    "int": '"%d "',
    "uint8_t": '"%u "',
    "uint32_t": '"%u "',
    "uint64_t": '"%llu "',
    "float8": '"%u "',
    "float16": '"%f "',
}

PRINT_CAST = {
    "float16": "__half2float(h_{}[i])",
    "float8": "static_cast<unsigned>(reinterpret_cast<uint8_t*>(h_{})[i])",
}

# Default dimensions for scalar params per problem
# Keys are problem names, values are dicts of param_name -> default_value
DEFAULTS = {
    "vector-addition": {"n": 1024},
    "relu": {"n": 64, "m": 64},
    "leaky-relu": {"n": 64, "m": 64, "alpha": 0.01},
    "gelu": {"n": 64, "m": 64},
    "sigmoid": {"n": 64, "m": 64},
    "tanh": {"n": 64, "m": 64},
    "elu": {"n": 64, "m": 64, "alpha": 1.0},
    "selu": {"n": 64, "m": 64},
    "swish": {"n": 64, "m": 64},
    "hard-sigmoid": {"n": 64, "m": 64},
    "soft-plus": {"n": 64, "m": 64},
    "matrix-multiplication": {"m": 64, "n": 64, "k": 64},
    "square-matmul": {"n": 64},
    "symmetric-matmul": {"n": 64},
    "lower-trig-matmul": {"n": 64},
    "upper-trig-matmul": {"n": 64},
    "diagonal-matmul": {"n": 64, "m": 64},
    "matrix-vector": {"m": 64, "k": 64},
    "matrix-scalar": {"n": 64, "scalar": 2.5},
    "matrix-power": {"n": 3, "size": 8},
    "conv-1d": {"N": 1024, "K": 5},
    "conv-2d": {"H": 64, "W": 64, "Kh": 3, "Kw": 3},
    "conv-square-3d": {"size": 16, "K": 3},
    "conv2d-relu-hardswish": {"H": 64, "W": 64, "Kh": 3, "Kw": 3},
    "softmax": {"dim": 1, "ndim": 2},
    "log-softmax": {"M": 64, "N": 64},
    "rms-norm": {"B": 8, "N": 64},
    "layer-norm": {"B": 2, "F": 4, "D1": 8, "D2": 8},
    "batch-norm": {"B": 2, "F": 4, "D1": 8, "D2": 8},
    "l1-norm": {"B": 8, "D": 64},
    "l2-norm": {"B": 8, "D": 64},
    "frobenius-norm": {"size": 4096},
    "sum-dim": {"dim": 1, "ndim": 2},
    "max-dim": {"dim": 1, "ndim": 2},
    "min-dim": {"dim": 1, "ndim": 2},
    "mean-dim": {"dim": 1, "ndim": 2},
    "argmax": {"dim": 1, "ndim": 2},
    "argmin": {"dim": 1, "ndim": 2},
    "product-dim": {"dim": 1, "ndim": 2},
    "cumsum": {"N": 1024},
    "cumprod": {"N": 1024},
    "running-sum-1d": {"W": 5, "N": 1024},
    "array-sort": {"n": 1024},
    "avg-pool-1d": {"kernel_size": 3, "stride": 1, "padding": 1, "H": 1024},
    "avg-pool-2d": {"kernel_size": 3, "stride": 1, "padding": 1, "H": 32, "W": 32},
    "avg-pool-3d": {"kernel_size": 3, "stride": 1, "padding": 1, "H": 16, "W": 16, "D": 16},
    "max-pool-1d": {"kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "H": 1024},
    "max-pool-2d": {"kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "H": 32, "W": 32},
    "max-pool-3d": {"kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "H": 16, "W": 16, "D": 16},
    "gemm-relu": {"B": 8, "N": 64, "M": 32},
    "gemm-multiply-leakyrelu": {"alpha": 0.01, "M": 64, "N": 64, "K": 64},
    "matmul-swish": {"scaling_factor": 1.0, "batch_size": 8, "in_features": 64, "out_features": 32},
    "matmul-swish-scaling": {"scale": 1.0, "M": 64, "N": 64, "K": 64},
    "matmul-sigmoid-sum": {"M": 64, "N": 64, "K": 64},
    "matmul-3d": {"n": 4, "m": 64, "k": 64, "l": 32},
    "matmul-4d": {"b": 2, "i": 4, "j": 32, "l": 32, "k": 16},
    "cosine-similarity": {"n": 64, "d": 128},
    "mse-loss": {"ndim": 1},
    "huber-loss": {"n": 1024},
    "hinge-loss": {"n": 1024},
    "kl-loss": {"n": 1024},
    "triplet-margin": {"B": 8, "E": 128, "margin": 1.0},
    "histogram": {"num_bins": 256, "height": 64, "width": 64},
    "grayscale": {"height": 64, "width": 64, "channels": 3},
    "box-blur": {"kernel_size": 3, "height": 64, "width": 64},
    "edge-detect": {"height": 64, "width": 64},
    "threshold": {"threshold_value": 0.5, "height": 64, "width": 64},
    "all-pairs-shortest-path": {"n": 64},
    "shortest-path": {"source": 0, "n": 64},
    "min-spanning-tree": {"n": 64},
    "ecc-point-negation": {"p": 0xFFFFFFFFFFFFFFC5, "n": 1024},
    "poly-multiply-ff": {"n": 256},
    "vector-multiply-ff": {"n": 1024},
    "mxfp4-dequantize": {"m": 64, "k": 64},
    "mxfp4-quantize": {"m": 64, "k": 64},
    "mxfp4-gemm": {"m": 64, "n": 64, "k": 64},
    "mxfp8-dequantize": {"m": 64, "k": 64},
    "mxfp8-quantize": {"m": 64, "k": 64},
    "mxfp8-gemm": {"m": 64, "n": 64, "k": 64},
    "nvfp4-dequantize": {"sf_g": 1.0, "m": 64, "k": 64},
    "nvfp4-quantize": {"sf_g": 1.0, "m": 64, "k": 64},
    "nvfp4-gemm": {"sf_g_a": 1.0, "sf_g_b": 1.0, "m": 64, "n": 64, "k": 64},
    "nvfp4-gemv": {"sf_g_a": 1.0, "sf_g_x": 1.0, "m": 64, "k": 64},
    "scaled-dot-attention": {"B": 2, "H": 4, "S": 32, "E": 64},
}


def get_all_params():
    results = {}
    for prob_name in sorted(os.listdir(PROBLEMS_DIR)):
        def_path = os.path.join(PROBLEMS_DIR, prob_name, "def.py")
        if not os.path.isfile(def_path):
            continue
        with open(def_path) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id == "parameters":
                                params_str = ast.get_source_segment(source, item.value)
                                try:
                                    params = eval(params_str)
                                    results[prob_name] = params
                                except:
                                    pass
    return results


def compute_buffer_size(prob_name, params):
    """Figure out the total element count for each pointer parameter."""
    scalars = {p["name"]: p for p in params if not p["pointer"]}
    pointers = [p for p in params if p["pointer"]]

    sizes = {}
    snames = [s["name"] for s in scalars.values()]
    defaults = DEFAULTS.get(prob_name, {})

    for ptr in pointers:
        pname = ptr["name"]

        # Heuristic: multiply all size_t/int scalar params that look like dimensions
        dim_params = []
        for s in scalars.values():
            if s["type"] in ("size_t", "int") and s["name"] not in (
                "dim", "ndim", "kernel_size", "stride", "padding", "dilation",
                "num_bins", "source", "channels",
            ):
                dim_params.append(s["name"])

        if not dim_params:
            dim_params = [s["name"] for s in scalars.values() if s["type"] in ("size_t",)]

        sizes[pname] = dim_params if dim_params else ["1024"]

    return sizes


def get_cpp_type(ptype):
    return TYPE_MAP.get(ptype, ptype)


def needs_extra_include(ptype):
    return INCLUDES_FOR_TYPE.get(ptype, None)


def get_scalar_default(prob_name, param):
    defaults = DEFAULTS.get(prob_name, {})
    if param["name"] in defaults:
        v = defaults[param["name"]]
        if param["type"] == "uint64_t":
            return f"{v}ULL"
        if param["type"] == "float":
            return f"{v}f"
        return str(v)
    if param["type"] == "float":
        return "1.0f"
    if param["type"] in ("int", "size_t"):
        return "64"
    if param["type"] == "uint64_t":
        return "1024ULL"
    return "0"


def generate_harness(prob_name, params):
    """Generate a complete main.cu harness for a single problem."""

    extra_includes = set()
    for p in params:
        inc = needs_extra_include(p["type"])
        if inc:
            extra_includes.add(inc)

    pointers = [p for p in params if p["pointer"]]
    scalars = [p for p in params if not p["pointer"]]
    const_ptrs = [p for p in pointers if p["const"]]
    mut_ptrs = [p for p in pointers if not p["const"]]

    # Build the solution function signature
    sig_parts = []
    for p in params:
        ctype = get_cpp_type(p["type"])
        if p["pointer"]:
            if p["const"]:
                sig_parts.append(f"const {ctype}* {p['name']}")
            else:
                sig_parts.append(f"{ctype}* {p['name']}")
        else:
            sig_parts.append(f"{ctype} {p['name']}")

    sig = ", ".join(sig_parts)

    # Figure out buffer sizes for each pointer
    # For dimension-based problems we multiply appropriate scalar params
    defaults = DEFAULTS.get(prob_name, {})

    # Special size computations per problem
    buf_sizes = {}
    for ptr in pointers:
        buf_sizes[ptr["name"]] = get_buffer_size_expr(prob_name, ptr, params, defaults)

    lines = []
    lines.append("#include <cstdio>")
    lines.append("#include <cstdlib>")
    lines.append("#include <cstring>")
    lines.append("#include <cstdint>")
    lines.append("#include <cuda_runtime.h>")
    for inc in sorted(extra_includes):
        lines.append(inc)
    lines.append("")
    lines.append(f'extern "C" void solution({sig});')
    lines.append("")
    lines.append("int main(int argc, char** argv) {")
    lines.append(f'    printf("=== {prob_name} ===\\n");')
    lines.append("    srand(42);")
    lines.append("")

    # Declare scalar variables
    for s in scalars:
        ctype = get_cpp_type(s["type"])
        val = get_scalar_default(prob_name, s)
        lines.append(f"    {ctype} {s['name']} = {val};")

    # Special: for shape-array problems (softmax, sum-dim, etc.)
    shape_problems = {
        "softmax", "sum-dim", "max-dim", "min-dim", "mean-dim",
        "argmax", "argmin", "product-dim", "mse-loss",
    }
    if prob_name in shape_problems:
        lines.append("")
        lines.append("    // Default shape: 2D tensor 64x64")
        has_size_t_shape = any(p["name"] == "shape" and p["type"] == "size_t" for p in pointers)
        has_int_shape = any(p["name"] == "shape" and p["type"] == "int" for p in pointers)
        if has_size_t_shape:
            lines.append("    size_t h_shape[] = {64, 64};")
        elif has_int_shape:
            lines.append("    int h_shape[] = {64, 64};")
        lines.append("    ndim = 2;")
        lines.append("    dim = 1;")

    lines.append("")

    # Compute buffer sizes
    for ptr in pointers:
        pname = ptr["name"]
        size_expr = buf_sizes[pname]
        lines.append(f"    size_t {pname}_count = {size_expr};")

    lines.append("")

    # Allocate host arrays
    for ptr in pointers:
        pname = ptr["name"]
        ctype = get_cpp_type(ptr["type"])
        if pname == "shape":
            continue  # handled specially
        lines.append(f"    {ctype}* h_{pname} = new {ctype}[{pname}_count];")

    lines.append("")

    # Fill input arrays with random data
    for ptr in const_ptrs:
        pname = ptr["name"]
        if pname == "shape":
            continue
        ctype = ptr["type"]
        rand_expr = RAND_EXPR.get(ctype, "0")
        lines.append(f"    for (size_t i = 0; i < {pname}_count; i++)")
        lines.append(f"        h_{pname}[i] = {rand_expr};")

    # Zero-fill output arrays
    for ptr in mut_ptrs:
        pname = ptr["name"]
        ctype = get_cpp_type(ptr["type"])
        lines.append(f"    memset(h_{pname}, 0, {pname}_count * sizeof({ctype}));")

    lines.append("")

    # Allocate device arrays
    for ptr in pointers:
        pname = ptr["name"]
        ctype = get_cpp_type(ptr["type"])
        if pname == "shape":
            shape_type = ctype
            lines.append(f"    {ctype}* d_shape;")
            lines.append(f"    cudaMalloc(&d_shape, ndim * sizeof({ctype}));")
            lines.append(f"    cudaMemcpy(d_shape, h_shape, ndim * sizeof({ctype}), cudaMemcpyHostToDevice);")
        else:
            lines.append(f"    {ctype}* d_{pname};")
            lines.append(f"    cudaMalloc(&d_{pname}, {pname}_count * sizeof({ctype}));")

    lines.append("")

    # Copy inputs to device
    for ptr in const_ptrs:
        pname = ptr["name"]
        ctype = get_cpp_type(ptr["type"])
        if pname == "shape":
            continue
        lines.append(f"    cudaMemcpy(d_{pname}, h_{pname}, {pname}_count * sizeof({ctype}), cudaMemcpyHostToDevice);")

    # Zero device outputs
    for ptr in mut_ptrs:
        pname = ptr["name"]
        ctype = get_cpp_type(ptr["type"])
        lines.append(f"    cudaMemset(d_{pname}, 0, {pname}_count * sizeof({ctype}));")

    lines.append("")

    # Call solution
    call_args = []
    for p in params:
        if p["pointer"]:
            call_args.append(f"d_{p['name']}")
        else:
            call_args.append(p["name"])
    lines.append(f'    solution({", ".join(call_args)});')
    lines.append("    cudaDeviceSynchronize();")
    lines.append("")

    # Copy outputs back
    for ptr in mut_ptrs:
        pname = ptr["name"]
        ctype = get_cpp_type(ptr["type"])
        lines.append(f"    cudaMemcpy(h_{pname}, d_{pname}, {pname}_count * sizeof({ctype}), cudaMemcpyDeviceToHost);")

    lines.append("")

    # Print first few output values
    for ptr in mut_ptrs:
        pname = ptr["name"]
        ctype = ptr["type"]
        fmt = PRINT_FMT.get(ctype, '"%f "')
        lines.append(f'    printf("Output {pname} (first 10): ");')
        lines.append(f"    for (size_t i = 0; i < 10 && i < {pname}_count; i++)")
        if ctype in PRINT_CAST:
            cast_expr = PRINT_CAST[ctype].format(pname)
            lines.append(f"        printf({fmt}, {cast_expr});")
        else:
            lines.append(f"        printf({fmt}, h_{pname}[i]);")
        lines.append('    printf("\\n");')

    lines.append("")

    # Cleanup
    for ptr in pointers:
        pname = ptr["name"]
        lines.append(f"    cudaFree(d_{pname});")
    for ptr in pointers:
        pname = ptr["name"]
        if pname == "shape":
            continue
        lines.append(f"    delete[] h_{pname};")

    lines.append("")
    lines.append('    printf("Done.\\n");')
    lines.append("    return 0;")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def get_buffer_size_expr(prob_name, ptr, params, defaults):
    """Return a C++ expression for the number of elements in a pointer buffer."""
    pname = ptr["name"]
    scalars = {p["name"]: p for p in params if not p["pointer"]}

    # Special cases by problem
    specials = {
        # shape arrays
        ("softmax", "shape"): "ndim",
        ("sum-dim", "shape"): "ndim",
        ("max-dim", "shape"): "ndim",
        ("min-dim", "shape"): "ndim",
        ("mean-dim", "shape"): "ndim",
        ("argmax", "shape"): "ndim",
        ("argmin", "shape"): "ndim",
        ("product-dim", "shape"): "ndim",
        ("mse-loss", "shape"): "ndim",
        # dim-reduce output: total / dim_size (for 2D: just one dim)
        ("softmax", "output"): "64 * 64",
        ("softmax", "input"): "64 * 64",
        ("sum-dim", "output"): "64",
        ("sum-dim", "input"): "64 * 64",
        ("max-dim", "output"): "64",
        ("max-dim", "input"): "64 * 64",
        ("min-dim", "output"): "64",
        ("min-dim", "input"): "64 * 64",
        ("mean-dim", "output"): "64",
        ("mean-dim", "input"): "64 * 64",
        ("argmax", "output"): "64",
        ("argmax", "input"): "64 * 64",
        ("argmin", "output"): "64",
        ("argmin", "input"): "64 * 64",
        ("product-dim", "output"): "64",
        ("product-dim", "input"): "64 * 64",
        ("mse-loss", "output"): "1",
        ("mse-loss", "predictions"): "64 * 64",
        ("mse-loss", "targets"): "64 * 64",

        # vector-addition
        ("vector-addition", "d_input1"): "n",
        ("vector-addition", "d_input2"): "n",
        ("vector-addition", "d_output"): "n",

        # array-sort
        ("array-sort", "a"): "n",
        ("array-sort", "b"): "n",

        # cumsum / cumprod
        ("cumsum", "input"): "N",
        ("cumsum", "output"): "N",
        ("cumprod", "input"): "N",
        ("cumprod", "output"): "N",

        # running-sum-1d
        ("running-sum-1d", "input"): "N",
        ("running-sum-1d", "output"): "N",

        # conv-1d
        ("conv-1d", "A"): "N",
        ("conv-1d", "B"): "K",
        ("conv-1d", "C"): "N",

        # conv-2d
        ("conv-2d", "A"): "H * W",
        ("conv-2d", "B"): "Kh * Kw",
        ("conv-2d", "C"): "H * W",

        # conv-square-3d
        ("conv-square-3d", "A"): "size * size * size",
        ("conv-square-3d", "B"): "K * K * K",
        ("conv-square-3d", "C"): "size * size * size",

        # conv2d-relu-hardswish
        ("conv2d-relu-hardswish", "image"): "H * W",
        ("conv2d-relu-hardswish", "kernel"): "Kh * Kw",
        ("conv2d-relu-hardswish", "output"): "H * W",

        # matrix-multiplication
        ("matrix-multiplication", "input_a"): "m * k",
        ("matrix-multiplication", "input_b"): "k * n",
        ("matrix-multiplication", "output_c"): "m * n",

        # square-matmul
        ("square-matmul", "input_a"): "n * n",
        ("square-matmul", "input_b"): "n * n",
        ("square-matmul", "output_c"): "n * n",

        # symmetric-matmul
        ("symmetric-matmul", "input_a"): "n * n",
        ("symmetric-matmul", "input_b"): "n * n",
        ("symmetric-matmul", "output_c"): "n * n",

        # lower/upper-trig-matmul
        ("lower-trig-matmul", "input_a"): "n * n",
        ("lower-trig-matmul", "input_b"): "n * n",
        ("lower-trig-matmul", "output_c"): "n * n",
        ("upper-trig-matmul", "input_a"): "n * n",
        ("upper-trig-matmul", "input_b"): "n * n",
        ("upper-trig-matmul", "output_c"): "n * n",

        # diagonal-matmul
        ("diagonal-matmul", "diagonal_a"): "n",
        ("diagonal-matmul", "input_b"): "n * m",
        ("diagonal-matmul", "output_c"): "n * m",

        # matrix-vector
        ("matrix-vector", "input_a"): "m * k",
        ("matrix-vector", "input_b"): "k",
        ("matrix-vector", "output_c"): "m",

        # matrix-scalar
        ("matrix-scalar", "input_matrix"): "n * n",
        ("matrix-scalar", "output_matrix"): "n * n",

        # matrix-power
        ("matrix-power", "input_matrix"): "size * size",
        ("matrix-power", "output_matrix"): "size * size",

        # gemm-relu: A(B,N), W(M,N), b(M), C(B,M)
        ("gemm-relu", "A"): "B * N",
        ("gemm-relu", "W"): "M * N",
        ("gemm-relu", "b"): "M",
        ("gemm-relu", "C"): "B * M",

        # gemm-multiply-leakyrelu
        ("gemm-multiply-leakyrelu", "A"): "M * K",
        ("gemm-multiply-leakyrelu", "B"): "K * N",
        ("gemm-multiply-leakyrelu", "C"): "M * N",
        ("gemm-multiply-leakyrelu", "output"): "M * N",

        # matmul-swish
        ("matmul-swish", "input_matrix"): "batch_size * in_features",
        ("matmul-swish", "weight_matrix"): "out_features * in_features",
        ("matmul-swish", "bias"): "out_features",
        ("matmul-swish", "output"): "batch_size * out_features",

        # matmul-swish-scaling
        ("matmul-swish-scaling", "A"): "M * K",
        ("matmul-swish-scaling", "B"): "K * N",
        ("matmul-swish-scaling", "output"): "M * N",

        # matmul-sigmoid-sum
        ("matmul-sigmoid-sum", "A"): "M * K",
        ("matmul-sigmoid-sum", "B"): "K * N",
        ("matmul-sigmoid-sum", "output"): "1",

        # matmul-3d
        ("matmul-3d", "A"): "n * m * k",
        ("matmul-3d", "B"): "k * l",
        ("matmul-3d", "C"): "n * m * l",

        # matmul-4d
        ("matmul-4d", "A"): "b * i * j * k",
        ("matmul-4d", "B"): "k * l",
        ("matmul-4d", "C"): "b * i * j * l",

        # 1d/2d/3d element-wise activations
        ("relu", "input"): "n * m",
        ("relu", "output"): "n * m",
        ("leaky-relu", "input"): "n * m",
        ("leaky-relu", "output"): "n * m",
        ("gelu", "input"): "n * m",
        ("gelu", "output"): "n * m",
        ("sigmoid", "input"): "n * m",
        ("sigmoid", "output"): "n * m",
        ("tanh", "input"): "n * m",
        ("tanh", "output"): "n * m",
        ("elu", "input"): "n * m",
        ("elu", "output"): "n * m",
        ("selu", "input"): "n * m",
        ("selu", "output"): "n * m",
        ("swish", "input"): "n * m",
        ("swish", "output"): "n * m",
        ("hard-sigmoid", "input"): "n * m",
        ("hard-sigmoid", "output"): "n * m",
        ("soft-plus", "input"): "n * m",
        ("soft-plus", "output"): "n * m",

        # rms-norm
        ("rms-norm", "X"): "B * N",
        ("rms-norm", "Y"): "B * N",

        # layer-norm
        ("layer-norm", "X"): "B * F * D1 * D2",
        ("layer-norm", "gamma"): "F * D1 * D2",
        ("layer-norm", "beta"): "F * D1 * D2",
        ("layer-norm", "Y"): "B * F * D1 * D2",

        # batch-norm
        ("batch-norm", "X"): "B * F * D1 * D2",
        ("batch-norm", "Y"): "B * F * D1 * D2",

        # l1/l2-norm
        ("l1-norm", "X"): "B * D",
        ("l1-norm", "Y"): "B * D",
        ("l2-norm", "X"): "B * D",
        ("l2-norm", "Y"): "B * D",

        # frobenius-norm
        ("frobenius-norm", "X"): "size",
        ("frobenius-norm", "Y"): "size",

        # log-softmax
        ("log-softmax", "input"): "M * N",
        ("log-softmax", "output"): "M * N",

        # cosine-similarity
        ("cosine-similarity", "predictions"): "n * d",
        ("cosine-similarity", "targets"): "n * d",
        ("cosine-similarity", "output"): "n",

        # loss functions
        ("huber-loss", "predictions"): "n",
        ("huber-loss", "targets"): "n",
        ("huber-loss", "output"): "1",
        ("hinge-loss", "predictions"): "n",
        ("hinge-loss", "targets"): "n",
        ("hinge-loss", "output"): "1",
        ("kl-loss", "predictions"): "n",
        ("kl-loss", "targets"): "n",
        ("kl-loss", "output"): "1",
        ("triplet-margin", "anchor"): "B * E",
        ("triplet-margin", "positive"): "B * E",
        ("triplet-margin", "negative"): "B * E",
        ("triplet-margin", "loss"): "1",

        # histogram
        ("histogram", "image"): "height * width",
        ("histogram", "histogram"): "num_bins",

        # image ops
        ("grayscale", "rgb_image"): "height * width * channels",
        ("grayscale", "grayscale_output"): "height * width",
        ("box-blur", "input_image"): "height * width",
        ("box-blur", "output_image"): "height * width",
        ("edge-detect", "input_image"): "height * width",
        ("edge-detect", "output_image"): "height * width",
        ("threshold", "input_image"): "height * width",
        ("threshold", "output_image"): "height * width",

        # pooling
        ("avg-pool-1d", "input"): "H",
        ("avg-pool-1d", "output"): "H",
        ("avg-pool-2d", "input"): "H * W",
        ("avg-pool-2d", "output"): "H * W",
        ("avg-pool-3d", "input"): "H * W * D",
        ("avg-pool-3d", "output"): "H * W * D",
        ("max-pool-1d", "input"): "H",
        ("max-pool-1d", "output"): "H",
        ("max-pool-2d", "input"): "H * W",
        ("max-pool-2d", "output"): "H * W",
        ("max-pool-3d", "input"): "H * W * D",
        ("max-pool-3d", "output"): "H * W * D",

        # graph
        ("all-pairs-shortest-path", "adj_matrix"): "n * n",
        ("all-pairs-shortest-path", "output"): "n * n",
        ("shortest-path", "d_adj_matrix"): "n * n",
        ("shortest-path", "d_distances"): "n",
        ("min-spanning-tree", "A"): "n * n",
        ("min-spanning-tree", "min_weight"): "1",

        # ecc
        ("ecc-point-negation", "xs"): "n",
        ("ecc-point-negation", "ys"): "n",
        ("ecc-point-negation", "out_xy"): "n * 2",

        # finite field
        ("poly-multiply-ff", "d_input1"): "n",
        ("poly-multiply-ff", "d_input2"): "n",
        ("poly-multiply-ff", "d_output"): "2 * n - 1",
        ("vector-multiply-ff", "d_input1"): "n",
        ("vector-multiply-ff", "d_input2"): "n",
        ("vector-multiply-ff", "d_output"): "n",

        # quantization
        ("mxfp4-dequantize", "q"): "m * k / 2",
        ("mxfp4-dequantize", "scale"): "m * (k / 32)",
        ("mxfp4-dequantize", "out"): "m * k",
        ("mxfp4-quantize", "a"): "m * k",
        ("mxfp4-quantize", "q"): "m * k / 2",
        ("mxfp4-quantize", "scale"): "m * (k / 32)",
        ("mxfp8-dequantize", "q"): "m * k",
        ("mxfp8-dequantize", "scale"): "m * (k / 32)",
        ("mxfp8-dequantize", "out"): "m * k",
        ("mxfp8-quantize", "a"): "m * k",
        ("mxfp8-quantize", "q"): "m * k",
        ("mxfp8-quantize", "scale"): "m * (k / 32)",

        ("mxfp4-gemm", "q_a"): "m * k / 2",
        ("mxfp4-gemm", "scale_a"): "m * (k / 32)",
        ("mxfp4-gemm", "q_b"): "k * n / 2",
        ("mxfp4-gemm", "scale_b"): "k * (n / 32)",
        ("mxfp4-gemm", "c"): "m * n",
        ("mxfp8-gemm", "q_a"): "m * k",
        ("mxfp8-gemm", "scale_a"): "m * (k / 32)",
        ("mxfp8-gemm", "q_b"): "k * n",
        ("mxfp8-gemm", "scale_b"): "k * (n / 32)",
        ("mxfp8-gemm", "c"): "m * n",

        ("nvfp4-dequantize", "q"): "m * k / 2",
        ("nvfp4-dequantize", "scale"): "m * (k / 16)",
        ("nvfp4-dequantize", "out"): "m * k",
        ("nvfp4-quantize", "a"): "m * k",
        ("nvfp4-quantize", "q"): "m * k / 2",
        ("nvfp4-quantize", "scale"): "m * (k / 16)",
        ("nvfp4-gemm", "q_a"): "m * k / 2",
        ("nvfp4-gemm", "scale_a"): "m * (k / 16)",
        ("nvfp4-gemm", "q_b"): "k * n / 2",
        ("nvfp4-gemm", "scale_b"): "k * (n / 16)",
        ("nvfp4-gemm", "c"): "m * n",
        ("nvfp4-gemv", "q_a"): "m * k / 2",
        ("nvfp4-gemv", "scale_a"): "m * (k / 16)",
        ("nvfp4-gemv", "q_x"): "k / 2",
        ("nvfp4-gemv", "scale_x"): "k / 16",
        ("nvfp4-gemv", "y"): "m",

        # scaled-dot-attention
        ("scaled-dot-attention", "Q"): "B * H * S * E",
        ("scaled-dot-attention", "K"): "B * H * S * E",
        ("scaled-dot-attention", "V"): "B * H * S * E",
        ("scaled-dot-attention", "output"): "B * H * S * E",
    }

    key = (prob_name, pname)
    if key in specials:
        return specials[key]

    # Fallback: multiply all size_t scalar params
    size_params = [p["name"] for p in params
                   if not p["pointer"] and p["type"] in ("size_t",)
                   and p["name"] not in ("ndim",)]
    if size_params:
        return " * ".join(size_params)

    return "1024"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_params = get_all_params()

    for prob_name, params in all_params.items():
        code = generate_harness(prob_name, params)
        out_path = os.path.join(OUTPUT_DIR, f"{prob_name}.cu")
        with open(out_path, "w") as f:
            f.write(code)
        print(f"Generated {out_path}")

    print(f"\nGenerated {len(all_params)} harness files in {OUTPUT_DIR}/")
    print(f"\nUsage: nvcc -o test harnesses/<problem>.cu <your-solution>.cu")


if __name__ == "__main__":
    main()
