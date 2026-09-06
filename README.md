# MiniInfer

**How fast can I make transformer inference?**

A sequence of reproducible experiments taking transformer inference from PyTorch eager execution toward Triton kernels and GPU serving. Each milestone gets its own implementation, correctness checks, measurements, and Git history.

**Current milestone: Version 1 — eager FP32 baseline.** CUDA performance results are pending. The offline CPU smoke model validates the harness and is not evidence of model quality or GPU performance.

| Implementation | TTFT (ms) | Output tok/s | Peak CUDA allocated (GiB) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| PyTorch eager FP32, no KV cache | — | — | — | — |
| PyTorch FP16 / BF16 | — | — | — | — |
| torch.compile | — | — | — | — |
| INT8 / INT4 | — | — | — | — |
| Triton kernel | — | — | — | — |

No illustrative numbers are presented as measurements. Kernel-only speedups will have a separate table from end-to-end generation. A Triton kernel result does not imply an equivalent model speedup.

## 1. Motivation

Study latency, throughput, numerical quality, and memory together. The aim is to explain measured changes in execution, not to build a chatbot or wrap an external inference API.

## 2. Architecture

```text
CLI configuration → model + fixed token workload → warmup
    → eager greedy generation → synchronized timings + optional NVML
    → raw JSON trials + aggregate metrics + environment metadata
```

## 3. Baseline

The default pretrained model is `openai-community/gpt2`. TinyLlama can be selected with `--model TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Explicit FP32, eager attention, evaluation mode, inference mode, and `use_cache=False` make the starting point clear. TF32 is disabled on CUDA. No compilation, quantization, serving, or custom kernel is enabled yet.

Generation recomputes the full sequence at each step. Greedy argmax emits exactly the requested number of tokens, ignoring EOS to keep work comparable. Input text is repeated and truncated to an exact token length; a batch repeats that same input without padding. This is a controlled workload, not a representative production traffic distribution.

## 4. Optimization techniques

Version 2 will compare compilation, FP32/FP16/BF16, and INT8/INT4 on identical workloads. Report compilation cost separately from steady state. Alongside performance, compare held-out perplexity, logit error, and greedy token agreement to FP32; token agreement alone is not a quality assessment.

## 5. Triton kernels

Version 3 will start with RMSNorm: PyTorch reference, compiled reference, and Triton. Check numerical tolerances across shapes before timing. Measure kernel latency separately; claim end-to-end improvement only after integrating a kernel into a model that uses the operation.

## 6. KV cache

Version 4 will add an explicit cached decode path and token-equivalence tests against the uncached baseline. Sweep prompt lengths 128, 256, 512, 1024, and 2048 using a model whose context window accommodates prompt plus generated tokens. GPT-2's context window cannot accommodate the entire sweep.

## 7. Dynamic batching

Version 5 will add FastAPI, a bounded request queue, dynamic batching, cancellation, and `/baseline`, `/optimized`, `/benchmark`, `/metrics`. Compare 1, 10, 50, and 100 concurrent clients with p50/p95/p99, requests/sec, tokens/sec, GPU utilization, and explicit queueing time.

## 8. Benchmarks

See [measurement definitions](docs/methodology.md). JSON reports contain every measured trial, generated token IDs, model revision, input hash, software versions, hardware, peak CUDA allocations/reservations, and optional NVML samples. Unavailable readings are `null`, never zero. Outputs cannot overwrite existing files.

## 9. Profiling

Version 6 will use PyTorch Profiler, Nsight Systems, and Nsight Compute. Capture profiler runs separately from timing runs. Attribute improvements to measured kernel launches, memory traffic, occupancy, or compute utilization; do not infer memory-bandwidth limits from latency alone.

## 10. Results

No NVIDIA benchmark has been run yet. Populate the table only from saved JSON on a named GPU, with identical model revision, input length, batch size, and output length. The committed `results/smoke-cpu.json`, when present, is a local functionality check using random weights.

## 11. Lessons learned

The first baseline establishes measurement boundaries: tokenization, model loading, downloads, and output serialization are outside inference timing. TTFT here is model-side first-token latency, not HTTP or user-perceived TTFT. Device-wide telemetry is distinct from this process's tensor memory.

## 12. Reproduce experiments

Use Python 3.11–3.13. Create and activate a virtual environment. For CUDA, install an appropriate PyTorch 2.8.0 build for your GPU/driver inside it using the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/) before installing MiniInfer.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,gpu]'
python -m pytest -q

# Offline harness check: no model download, random weights, CPU only.
miniinfer --smoke --device cpu --prompt-tokens 16 --new-tokens 4 --runs 3

# Real pretrained baseline; the first invocation downloads model artifacts.
miniinfer --device cuda --model openai-community/gpt2 \
  --prompt-tokens 128 --new-tokens 32 --batch-size 1 \
  --warmup 3 --runs 30 --output results/gpt2-fp32-cuda.json
```

For repeat experiments, pass `--revision` with the resolved model commit SHA saved in the initial JSON. Save `python -m pip freeze` alongside published results to capture transitive dependencies. `--device mps` supports local Apple Silicon checks but cannot produce NVIDIA utilization or CUDA VRAM readings. `--threads` controls CPU execution threads.

See [the sequential milestone plan](docs/roadmap.md). Each version is a separate reviewable commit/tag; future versions remain planned until their correctness and experiment gates pass.
