# MiniInfer

### Optimizing transformer inference from PyTorch → Triton → CUDA


I'm building this project to understand what happens when a transformer generates tokens, where the time and memory go, and which changes actually make it faster. I started with a simple PyTorch baseline and then added one systems idea at a time.

The repo now has six versions built: baseline inference, precision/compile/quantization options, an RMSNorm kernel experiment, KV cache decoding, a small inference server, and profiling tools. I also verified the CUDA-specific tests on a Runpod RTX 4090. The main thing still missing is a full pretrained-model GPU benchmark table with repeated performance runs.

| Implementation | TTFT (ms) | Output tokens/sec | Peak CUDA allocated (GiB) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| PyTorch eager FP32, no KV cache | — | — | — | — |
| torch.compile | — | — | — | — |
| FP16 / BF16 | — | — | — | — |
| INT8 / INT4 | — | — | — | — |
| Triton, integrated into a model | — | — | — | — |

This table is only for repeated pretrained GPU performance results. I leave cells blank until I run that exact experiment. CUDA correctness is already verified, but correctness tests are not the same thing as speedup numbers.

## 1. Motivation

I want to learn inference by building and measuring it. For each change, I'll ask:

- What work does this remove or make cheaper?
- How much faster is it, and how much memory does it use?
- Does it change the model's output or quality?
- Can I explain the result with measurements?

## 2. Architecture

The current system is a Python command-line benchmark:

```text
Choose model and workload
          ↓
Load model and prepare tokens
          ↓
Run warmup
          ↓
Generate tokens and measure
          ↓
Save timings, metrics, and settings to JSON
```

| File | What it does |
| --- | --- |
| [benchmark.py](src/miniinfer/benchmark.py) | Loads the model, runs generation, and writes results |
| [optimization.py](src/miniinfer/optimization.py) | Loads precision and quantization variants |
| [quality.py](src/miniinfer/quality.py) | Compares predictions with eager FP32 |
| [compare.py](src/miniinfer/compare.py) | Compares matching experiment reports |
| [generation.py](src/miniinfer/generation.py) | Runs cached and uncached decoding |
| [cache_sweep.py](src/miniinfer/cache_sweep.py) | Compares caching across prompt lengths |
| [metrics.py](src/miniinfer/metrics.py) | Calculates latency and throughput summaries |
| [telemetry.py](src/miniinfer/telemetry.py) | Samples NVIDIA GPU utilization and memory when available |
| [tests/](tests/) | Checks generation, metric calculations, and the CLI |
| [kernels/](src/miniinfer/kernels/) | PyTorch and Triton RMSNorm implementations |
| [kernel_benchmark.py](src/miniinfer/kernel_benchmark.py) | Checks and times standalone RMSNorm |
| [results/](results/) | Stores experiment reports |

## 3. Baseline

The default model is GPT-2 from Hugging Face. The CLI also accepts another compatible model, such as TinyLlama.

The baseline uses FP32 (32-bit floating point), eager execution, and no KV cache. Eager execution runs operations as they are called. Each generation step runs the model on the whole sequence, picks the most likely next token, and appends it.

```text
prompt                 → next token
prompt + token         → next token
prompt + token + token → next token
```

I use a fixed input length and output length to keep the amount of work consistent. The prompt is repeated or cut to fit, and each item in a batch gets the same input. Generation continues for the requested token count even if the model emits an end token.

CPU, Apple Silicon MPS, and NVIDIA CUDA are supported. The offline smoke mode uses a tiny model with random weights to check that the benchmark runs.

## 4. Optimization techniques

Version 2 now supports three experiments:

| Experiment | What I want to learn |
| --- | --- |
| `torch.compile` | Can compiling the model reduce generation time? |
| FP16 / BF16 | How do 16-bit formats affect speed, memory, and numerical results? |
| INT8 / INT4 | How much memory can quantization save, and what does it cost in quality and speed? |

I'll keep the workload fixed. The first generation, which includes compilation when enabled, is recorded separately from later runs. Quality checks use held-out text, perplexity (how well the model predicts that text), differences in model scores, and generated-token agreement with FP32.

Read the [Version 2 walkthrough](docs/version-2.md) for commands and how to interpret the results. The implementation is in place; pretrained GPU experiments are still pending.

## 5. Triton kernels

Version 3 implements RMSNorm, which rescales each row using its root mean square and a learned weight vector.

There are three paths: a PyTorch reference, the same math with `torch.compile`, and a custom Triton kernel. They use FP32 arithmetic internally and return the input dtype. A separate `miniinfer-kernel` command checks correctness before timing each shape and dtype.

CPU tests and a compiled CPU sweep can run locally. Triton execution and GPU results still need CUDA verification. The kernel isn't integrated into a model, so its timings don't represent generation speedups. CUDA C++ remains a possible later extension.

Read the [Version 3 walkthrough](docs/version-3.md) for the formula, kernel design, limits, and commands.

## 6. KV cache

Version 4 adds an explicit decoding loop that reuses attention keys and values through Hugging Face's `DynamicCache`.

```text
Process the prompt once → save K and V
Process one new token   → reuse earlier K and V
```

The `--kv-cache` option checks generated-token agreement against the uncached path. A separate `miniinfer-cache-sweep` command compares prompt lengths and saves latency and memory plots. CPU smoke results are saved; pretrained GPU measurements are pending.

Read the [Version 4 walkthrough](docs/version-4.md) for the cache logic, memory formula, and commands. The full 128–2048 prompt-length sweep needs a model with enough room for both prompt and output.

## 7. Dynamic batching

Version 5 adds a FastAPI server with one model worker, a bounded request queue, and dynamic batching. The baseline route runs uncached singleton requests; the optimized route groups compatible requests and uses the KV cache.

`miniinfer-load` tests concurrent clients and saves HTTP latency, throughput, failures, queue timings, and telemetry. Read the [Version 5 walkthrough](docs/version-5.md) for the API and local commands.

## 8. Benchmarks

The benchmark records:

- **TTFT:** time until the first generated token.
- **Latency:** time to generate the full requested output.
- **Throughput:** output tokens/sec, decode tokens/sec, and completed sequences/sec.
- **p50 / p95 / p99 latency:** the middle and slower ends of the measured batch timings.
- **Memory:** peak CUDA allocated and reserved memory, plus device-wide memory when available.
- **GPU utilization:** optional NVIDIA samples during measured runs.

Timing starts with the model and input tokens already on the device. Downloads, model loading, tokenization, warmup, and saving the report are outside the timer. These are model-side measurements; server measurements will come later.

Each JSON report includes raw trials, generated token IDs, settings, model revision, input hash, software versions, and device details. Missing metrics are `null`. Existing result files are never overwritten.

See [how measurements work](docs/methodology.md) for the exact definitions.

## 9. Profiling

Version 6 adds `miniinfer-profile` for separate generation and RMSNorm traces, operator summaries, and observation tables. CPU profiler checks are verified. NVIDIA Nsight capture commands are documented but still need a GPU run.

Read the [Version 6 walkthrough](docs/version-6.md) for how to investigate a bottleneck without confusing profiler overhead with benchmark results.

## 10. Results

The saved [CPU smoke run](results/smoke-cpu.json) used a tiny randomly initialized GPT-2 on an Apple Silicon CPU, with 16 input tokens and 4 generated tokens across 3 measured runs.

That checks the benchmark workflow. It doesn't establish pretrained-model quality or GPU performance.

The CUDA validation was run on a Runpod RTX 4090:

| Check | Result |
| --- | --- |
| GPU-specific tests | 23 passed |
| Full test suite | 116 passed, 2 dependency warnings |
| PyTorch / Triton | PyTorch 2.8.0+cu128, Triton 3.4.0 |
| BF16 | Native BF16 supported |

Those tests prove the CUDA paths work: Triton RMSNorm correctness and cached CUDA generation parity both passed. They don't claim an end-to-end inference speedup yet. That comes from the next real benchmark run.

## 11. Lessons learned

The baseline makes a few things clear:

- Timing boundaries matter: loading a model and generating a token are different measurements.
- Tokens/sec needs the total token count divided by total generation time.
- CUDA tensor allocations and whole-device memory use measure different things.
- A tiny smoke test is useful for checking code, but performance comparisons need a fixed, realistic workload.
- A passing GPU test tells me the CUDA code is correct on that machine, but it is not a benchmark by itself.
- Dynamic batching helped the CPU smoke server under higher concurrency, but the result is still a workflow check because it used a tiny random model.

I'll add what I learn from each experiment as I go.

## 12. Reproduce experiments

Use Python 3.11–3.13. For NVIDIA runs, install a PyTorch 2.8.0 CUDA build that matches your GPU and driver before installing this project.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Run a small offline check:

```bash
miniinfer --smoke --device cpu \
  --prompt-tokens 16 --new-tokens 4 --runs 3
```

For optional NVIDIA telemetry, install the GPU extra:

```bash
python -m pip install -e '.[gpu]'
```

Run the pretrained baseline on an NVIDIA GPU. The first run downloads the model:

```bash
miniinfer --device cuda --model openai-community/gpt2 \
  --prompt-tokens 128 --new-tokens 32 --batch-size 1 \
  --warmup 3 --runs 30 --output results/gpt2-fp32-cuda.json
```

Use `--device mps` for Apple Silicon or `--device cpu` for CPU runs. Those runs won't have CUDA memory or NVIDIA utilization readings.

For repeat experiments, pass `--revision` with the model commit saved in the first report. Keep the GPU and workload the same, use a new output filename, and save the installed dependencies alongside the results:

```bash
python -m pip freeze > results/requirements.txt
```

The [roadmap](docs/roadmap.md) lists the next experiments and what I'll check before calling each version complete.
