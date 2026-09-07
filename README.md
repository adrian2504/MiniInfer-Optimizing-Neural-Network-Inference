# MiniInfer

### Optimizing transformer inference from PyTorch → Triton → CUDA

**How fast can I make transformer inference?**

I'm building this project to understand what happens when a transformer generates tokens, where the time and memory go, and which changes actually make it faster. I'll start with a simple PyTorch baseline and work through each optimization one at a time.

The Version 1 baseline code is built, with tests and a saved CPU smoke run. The next part is Version 2: compilation, lower precision, and quantization. A real pretrained GPU baseline still needs to be measured before I can report speedups.

| Implementation | TTFT (ms) | Output tokens/sec | Peak CUDA allocated (GiB) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| PyTorch eager FP32, no KV cache | — | — | — | — |
| torch.compile | — | — | — | — |
| FP16 / BF16 | — | — | — | — |
| INT8 / INT4 | — | — | — | — |
| Triton, integrated into a model | — | — | — | — |

The table is for measured GPU results. Empty cells mean the experiment hasn't been recorded yet. Speedup will compare output tokens/sec against the FP32 baseline on the same workload and GPU. Individual kernel timings will have their own table.

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
| [metrics.py](src/miniinfer/metrics.py) | Calculates latency and throughput summaries |
| [telemetry.py](src/miniinfer/telemetry.py) | Samples NVIDIA GPU utilization and memory when available |
| [tests/](tests/) | Checks generation, metric calculations, and the CLI |
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

Version 2 will compare three changes:

| Experiment | What I want to learn |
| --- | --- |
| `torch.compile` | Can compiling the model reduce generation time? |
| FP16 / BF16 | How do 16-bit formats affect speed, memory, and numerical results? |
| INT8 / INT4 | How much memory can quantization save, and what does it cost in quality and speed? |

I'll keep the workload fixed and record compilation time separately from later runs. Quality checks will use held-out text, perplexity (how well the model predicts that text), differences in model scores, and generated-token agreement with FP32.

## 5. Triton kernels

Version 3 will start with RMSNorm, an operation that rescales a vector using its root mean square.

I'll write a PyTorch reference, try `torch.compile`, and then implement it in Triton. First I'll check that the outputs agree within a chosen tolerance across input sizes and data types. Then I'll measure kernel latency.

A faster RMSNorm kernel only improves generation if it is integrated into a model that uses it. I'll measure those two results separately. CUDA C++ can be a later extension.

## 6. KV cache

Version 4 will save attention keys and values from earlier tokens so generation can reuse them.

```text
Process the prompt once → save K and V
Generate the next token → reuse earlier K and V
```

I'll compare cached and uncached output tokens, then plot latency and memory against prompt length. The planned lengths are 128, 256, 512, 1024, and 2048 tokens, using a model with enough context space for both the prompt and output. GPT-2 cannot cover the full sweep.

## 7. Dynamic batching

Version 5 will add an inference server:

```text
Clients → FastAPI → request queue → dynamic batcher → model → GPU
```

The planned routes are `/baseline`, `/optimized`, `/benchmark`, and `/metrics`. The batcher will combine waiting requests so the GPU can process them together.

I'll test 1, 10, 50, and 100 concurrent clients, measure queueing time, and check cancellation and overload behavior.

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

Version 6 will use PyTorch Profiler, Nsight Systems, and Nsight Compute to explain the timing results.

I'll look at where execution time goes, how many kernels run, and how much memory traffic they create. Profiling runs will be separate from timing runs, since profiling itself can add overhead.

## 10. Results

The saved [CPU smoke run](results/smoke-cpu.json) used a tiny randomly initialized GPT-2 on an Apple Silicon CPU, with 16 input tokens and 4 generated tokens across 3 measured runs.

That checks the benchmark workflow. It doesn't establish pretrained-model quality or GPU performance. No NVIDIA benchmark results are recorded yet.

## 11. Lessons learned

The baseline makes a few things clear:

- Timing boundaries matter: loading a model and generating a token are different measurements.
- Tokens/sec needs the total token count divided by total generation time.
- CUDA tensor allocations and whole-device memory use measure different things.
- A tiny smoke test is useful for checking code, but performance comparisons need a fixed, realistic workload.

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
