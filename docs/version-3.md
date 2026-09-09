# Version 3 — My first Triton kernel

This version implements RMSNorm three ways: PyTorch, compiled PyTorch, and Triton. It has its own benchmark so I can study one operation before changing a whole model.

The code is built. I verified Triton correctness on a Runpod RTX 4090. GPU performance timing still needs a repeated benchmark run before I claim a speedup.

## What is RMSNorm?

RMSNorm rescales a row of numbers, then multiplies it by a learned weight vector:

```text
mean_square = mean(x * x)
output = x / sqrt(mean_square + eps) * weight
```

`eps` is a small positive number that keeps the denominator from becoming zero. The operation works across the last dimension. For an input shaped `[batch, tokens, hidden_size]`, each token's hidden vector is one row.

For example, with `x = [3, 4]`, the mean square is 12.5. With weights `[1, 1]` and a tiny epsilon, the output is about `[0.849, 1.131]`.

RMSNorm doesn't subtract the row's mean. That is one difference from LayerNorm. GPT-2 uses LayerNorm, so I haven't replaced its normalization with this operation. This version doesn't change model generation.

## The three implementations

### PyTorch

The reference follows the formula directly. It converts values and weights to FP32, calculates the mean square, normalizes, applies weights, then converts the output back to the input dtype.

Using FP32 internally matters for FP16: a value such as 1000 fits in FP16, but its square does not.

### Compiled PyTorch

The same math is passed to `torch.compile` with Inductor and `fullgraph=True`. The compiler must capture the whole function; graph breaks are errors here.

Each shape and dtype gets a fresh compiler setup. First-call timing is recorded separately, followed by warmup. If the compiler captures another graph during measurement, the benchmark rejects the run.

### Triton

The custom kernel assigns one program to each row:

1. Load the row and its weights into FP32 values.
2. Square the values and reduce them to a sum.
3. Divide by the real row width and calculate the inverse RMS.
4. Normalize, apply the weights, and store the output.

Triton uses a power-of-two block size. For a row with 127 values, the block has 128 slots. A mask prevents reading or writing past the row, and the extra slot contributes zero to the sum.

The kernel combines these steps in one launch. The hypothesis is that this can reduce intermediate memory traffic and launch overhead. Measurements and later profiling will tell me whether it helps on a particular shape.

## Limits for this first kernel

- Forward inference only; no backward pass.
- Contiguous inputs and weights on the same device, with matching dtypes.
- FP32, FP16, and BF16 inputs; FP32 arithmetic and reduction in every implementation.
- One weight vector matching the last dimension.
- Nonempty inputs with a last dimension from 1 to 16,384.
- Finite input values whose squared reduction fits in FP32.
- Triton execution targets NVIDIA CUDA. CPU can run the two PyTorch versions.

These limits keep the first implementation small enough to understand. Unsupported layouts are rejected instead of silently copying inputs during timing.

## Run a local check

Use the existing Python 3.11–3.13 environment:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q

miniinfer-kernel --device cpu --implementations eager \
  --shapes 3x127 32x1024 --dtypes fp32 fp16 bf16 \
  --runs 5 --iterations 10 --warmup 2 \
  --output results/rmsnorm-cpu.json
```

To compare eager and compiled PyTorch, change the option to `--implementations eager compiled` and choose a new output filename. CPU compilation needs a C++ toolchain. The macOS library-path workaround from the [Version 2 guide](version-2.md) may apply.

The command prints a small timing table and saves the full report. CPU numbers describe CPU execution; they don't establish Triton performance.

## Run on an NVIDIA GPU

Use Linux with the project's CUDA PyTorch 2.8.0 environment. Install the kernel extra, which pins Triton 3.4.0:

```bash
python -m pip install -e '.[dev,kernels]'

python -m pytest -q tests/test_rmsnorm.py -k triton_against_reference_on_cuda

miniinfer-kernel --device cuda \
  --implementations eager compiled triton \
  --shapes 1x128 32x1024 128x4096 \
  --dtypes fp32 fp16 bf16 \
  --runs 30 --iterations 100 --warmup 10 \
  --output results/rmsnorm-cuda.json
```

On a GPU without native BF16, use `--dtypes fp32 fp16`. Requested implementations aren't silently skipped by the benchmark. The CUDA tests skip when their hardware or Triton dependency isn't available, so check the test summary before treating them as verified.

Run with a mostly idle GPU. Save `nvidia-smi` output and `python -m pip freeze` with published results, and repeat whole experiments to check variability.

## What gets checked?

Before timing, each implementation is compared with an independent FP64 calculation made from the actual input values. The output must have the same shape, dtype, and device, contain finite values, and pass these tolerances:

| Output dtype | Relative tolerance | Absolute tolerance |
| --- | ---: | ---: |
| FP32 | 0.00001 | 0.00001 |
| FP16 | 0.002 | 0.002 |
| BF16 | 0.02 | 0.02 |

The report saves the tolerances and maximum/mean absolute error. These are starting correctness bounds, not a claim that the errors are harmless in a model.

Tests cover irregular widths, multiple leading dimensions, zero values, small values, and large values. GPU tests compare Triton with both the FP64 calculation and the PyTorch reference. The existing `MINIINFER_TEST_COMPILE=1` switch also enables a real compiled RMSNorm sweep in all three dtypes; Linux CPU CI runs it.

## What gets timed?

Every sample calls an implementation `--iterations` times and divides the elapsed time by that count. There are `--runs` samples. The report keeps every sample and gives the mean, median, and p95 of these per-call averages.

- **CPU:** wall-clock time, including Python dispatch and allocation.
- **CUDA:** CUDA events around the repeated calls. Event intervals can include gaps while the CPU dispatches work; this isn't an isolated device-instruction measurement.
- **Startup:** a separately timed first call, which can include compilation or JIT work. Compiler disk caches may be reused.
- **Allocation:** each call allocates an output. Eager PyTorch also creates intermediates.
- **Cache:** the same input and weight are reused, without flushing caches.
- **Order:** implementation order is shuffled with a fixed seed for every measured round.

Inputs, transfers, the FP64 oracle, correctness checks, first calls, and warmup are outside the steady-state timings. A failed correctness check prevents the report from being saved.

This measures an RMSNorm operation. It doesn't measure tokens/sec, model memory savings, or end-to-end generation latency.

## Results

CUDA validation:

| Check | Result |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| PyTorch / Triton | PyTorch 2.8.0+cu128, Triton 3.4.0 |
| Native BF16 | Yes |
| CUDA RMSNorm tests | 21 passed |

This proves the Triton kernel matched the reference on the tested shapes and dtypes. It is not a speed benchmark.

| Implementation | GPU median operation time | Speedup vs eager |
| --- | ---: | ---: |
| PyTorch | — | — |
| Compiled PyTorch | — | — |
| Triton | — | — |

I'll fill the timing table from saved repeated runs on a named GPU, for a stated shape and dtype. Speedup is eager median time divided by candidate median time. There isn't one speedup that describes every input size.

## Code to read

1. [rmsnorm.py](../src/miniinfer/kernels/rmsnorm.py): the formula and input checks.
2. [triton_rmsnorm.py](../src/miniinfer/kernels/triton_rmsnorm.py): row indexing, masks, reduction, and stores.
3. [kernel_benchmark.py](../src/miniinfer/kernel_benchmark.py): correctness checks, warmup, and timing.
4. [test_rmsnorm.py](../tests/test_rmsnorm.py): supported shapes and edge cases.

The [Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) is useful background for row reductions and masked loads. The RMSNorm formula here is simpler because it doesn't subtract a mean or add a bias.
