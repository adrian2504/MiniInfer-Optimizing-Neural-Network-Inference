# Version 2 — Trying optimizations

The code for these experiments is in place. I can check the workflow on CPU, but GPU speedups and pretrained quality results still need to be measured.

I'll change one thing at a time and keep the model, input, and GPU fixed.

## What changed?

| Option | What it does |
| --- | --- |
| `--compile` | Runs the model through `torch.compile` with the Inductor backend |
| `--dtype fp32` | Uses the original 32-bit baseline |
| `--dtype fp16` | Uses 16-bit floating point |
| `--dtype bf16` | Uses a different 16-bit format with a wider numerical range |
| `--quantization int8` | Loads supported layers using bitsandbytes LLM.int8 |
| `--quantization int4` | Loads supported layers using bitsandbytes NF4, a 4-bit format |
| `--eval-text FILE` | Checks quality against the same model in eager FP32 |
| `--check-quality` | Enables synthetic quality checks in smoke mode; real models also need `--eval-text` |

The `int4` CLI name means the 4-bit experiment. It uses NF4, not uniform integer quantization. Quantized runs still use FP16 or BF16 for unquantized layers and, for NF4, computation. The JSON records the full quantization settings and parameter data types.

Generation still has no KV cache. That experiment comes in Version 4.

## 1. Check it locally

Activate your Python 3.11–3.13 environment and update the editable installation:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Run a baseline and compiled smoke experiment in separate processes:

```bash
miniinfer --smoke --device cpu --check-quality \
  --prompt-tokens 8 --new-tokens 3 --runs 5 --warmup 2 \
  --eval-tokens 16 --output results/v2-smoke-fp32.json

miniinfer --smoke --device cpu --compile --check-quality \
  --prompt-tokens 8 --new-tokens 3 --runs 5 --warmup 2 \
  --eval-tokens 16 --output results/v2-smoke-compiled.json

miniinfer-compare results/v2-smoke-fp32.json results/v2-smoke-compiled.json
```

The first compiled run can take a while. It needs a working C++ toolchain for CPU execution. CUDA compilation also needs a supported GPU and compiler environment. This version doesn't enable compilation on MPS.

On my macOS setup, Inductor built a library but couldn't find `libc++.1.dylib` when loading it. Setting `DYLD_LIBRARY_PATH=/usr/lib` for that command let the CPU check run. This is a local environment workaround, not a benchmark default.

To include the real Inductor integration test:

```bash
MINIINFER_TEST_COMPILE=1 python -m pytest -q
```

CI enables that test on Linux. The regular local test run skips it because it requires a compiler and takes longer.

## 2. Measure the GPU baseline

Use the same NVIDIA GPU for all comparisons. Start with GPT-2 or another small compatible pretrained model that fits in FP32.

Prepare a UTF-8 file with one held-out text document per line. Keep it separate from the repeated benchmark prompt. `data/held-out.txt` below is a path to your own dataset; it isn't bundled with the project.

```bash
miniinfer --device cuda --model openai-community/gpt2 \
  --dtype fp32 --prompt-tokens 128 --new-tokens 32 \
  --batch-size 1 --runs 30 --warmup 3 \
  --eval-text data/held-out.txt --eval-samples 32 --eval-tokens 128 \
  --output results/v2-fp32.json
```

Copy `model.resolved_revision` from the report and use it for every later run. In the commands below, replace `MODEL_COMMIT` with that value.

## 3. Try compilation

```bash
miniinfer --device cuda --model openai-community/gpt2 --revision MODEL_COMMIT \
  --dtype fp32 --compile --prompt-tokens 128 --new-tokens 32 \
  --batch-size 1 --runs 30 --warmup 3 \
  --eval-text data/held-out.txt --eval-samples 32 --eval-tokens 128 \
  --output results/v2-compiled.json
```

Compilation uses dynamic shapes because the sequence grows as tokens are appended. The model can be split into several compiled graphs with Python between them. The report records captured graphs and graph breaks; enabling compilation doesn't mean every operation became one kernel.

There is always one separately timed first generation, followed by `--warmup` additional generations, then the measured trials. If new graphs are compiled during the measured trials, the benchmark asks for more warmup instead of saving a steady-state result.

`startup.first_generation_ms` includes compilation when needed **and** generation. It isn't pure compilation time. Compiler disk caches can make later process startups faster, so record whether caches were reused when comparing startup costs.

## 4. Try FP16 and BF16

Repeat the baseline command with `--dtype fp16` and a new output filename, then with `--dtype bf16`. Leave `--compile` off for these first comparisons.

Native BF16 support is checked on CUDA. A GPU that doesn't support it should use FP16 or FP32. Lower precision isn't guaranteed to be faster on every device or workload.

## 5. Try quantization

Install the optional dependencies in the CUDA environment:

```bash
python -m pip install -e '.[quantization,gpu]'
```

Repeat the FP16 experiment with `--quantization int8`, then with `--quantization int4`, saving a new file each time. Keep `--dtype fp16` for both so unquantized layers use the same format.

Version 2 restricts quantization to pretrained CUDA models. It doesn't combine quantization and compilation. The whole model goes on one GPU; automatic CPU offloading is disabled. The optional dependencies are pinned in `pyproject.toml`.

## 6. Read the comparison

```bash
miniinfer-compare results/v2-fp32.json results/v2-compiled.json
```

The first file must be eager FP32 without quantization. The tool rejects different inputs, model revisions, run settings, or recorded environments. Old Version 1 reports need to be rerun because their startup protocol differs.

- `output_throughput_speedup`: candidate tokens/sec divided by baseline tokens/sec. Above 1 means faster.
- `latency_speedup` and `ttft_speedup`: baseline time divided by candidate time. Above 1 means faster.
- `cuda_allocated_memory_ratio`: candidate peak tensor memory divided by baseline peak tensor memory. Below 1 means less memory.
- `quality`: the candidate's separate comparison against its eager FP32 reference, or `null` if no quality check was requested.

These ratios describe the saved runs. Repeat the experiments before drawing conclusions. The tool checks recorded metadata, but can't detect changes in GPU load or power settings.

## 7. Understand quality

Evaluation runs after the timing and memory readings are saved in memory. Candidate logits are temporarily written to disk, then the candidate is unloaded before loading the FP32 reference. This avoids keeping two models on the GPU at once. The GPU must still be able to hold the FP32 model by itself.

For each document, evaluation predicts each token using the preceding text. It uses at most `--eval-tokens` tokens per document and the first `--eval-samples` usable documents. Empty lines and documents shorter than two tokens are skipped. Documents are evaluated independently, without padding or overlapping windows.

The report contains:

- **Perplexity and average negative log-likelihood:** how well each model predicts the held-out text. Loss is weighted by token count, not document count. Perplexity is `null` if exponentiating the loss would overflow.
- **Mean and maximum logit error:** how much the candidate's scores differ from FP32 on identical text.
- **Teacher-forced token agreement:** how often both models choose the same next token given identical preceding text.
- **Generated-token agreement:** how often their free-running output tokens match on the benchmark prompt. After a difference, their later contexts can differ too.
- **Dataset hashes and token count:** which data the evaluation actually used.

A smoke check uses random weights and synthetic tokens. Its quality numbers check the implementation, not language ability. Four short documents, the default, are only a quick check; meaningful quality conclusions need a larger, appropriate held-out dataset.

## Code to read

Start with `benchmark.py` for the overall flow. Then read `optimization.py` for model loading, `quality.py` for the FP32 comparison, and `compare.py` for comparing saved runs.

References used for the implementation: [PyTorch 2.8 compilation](https://docs.pytorch.org/docs/2.8/generated/torch.compile.html) and [Transformers 4.56.2 bitsandbytes integration](https://huggingface.co/docs/transformers/v4.56.2/quantization/bitsandbytes).
