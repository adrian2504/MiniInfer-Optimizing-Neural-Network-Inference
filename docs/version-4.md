# Version 4 — Reusing attention with a KV cache

This version adds a cached generation loop and a prompt-length sweep. I want to see how avoiding repeated work changes latency, and how much memory the cache needs.

The implementation and CPU checks are in place. Cached CUDA parity also passed on a Runpod RTX 4090. The saved smoke sweep uses random weights; repeated pretrained GPU results are still pending.

## What changes during generation?

Without a cache, every step processes the whole sequence:

```text
[prompt]                 → token 1
[prompt, token 1]        → token 2
[prompt, token 1, token 2] → token 3
```

With a cache, the prompt is processed once. Each later step sends only the newest token:

```text
[prompt] → save its attention keys and values → token 1
[token 1] + saved K/V → token 2
[token 2] + saved K/V → token 3
```

The new token still attends to earlier keys and values. Caching avoids recalculating those earlier representations; it doesn't make attention independent of context length.

## What I implemented

`generation.py` contains both loops. The cached path creates a new Hugging Face `DynamicCache` for every trial and explicitly passes it between forward calls. The model's attention layers calculate and append K/V tensors through that cache. This is a custom decoding loop using the library's cache storage, not a new attention implementation.

The loop also maintains:

- **The attention mask:** one entry for every past and current token.
- **Cache positions:** the absolute positions of the tokens being processed.
- **The generated sequence:** used to collect the output and select the next input token.

This first version uses equal-length inputs without padding. It has no cache sharing between requests, paging, offloading, or static cache. Dynamic caching and `--compile` are kept separate for now.

## Try one cached run

Update the existing Python 3.11–3.13 environment:

```bash
python -m pip install -e '.[dev,plots]'
python -m pytest -q

miniinfer --smoke --device cpu --kv-cache \
  --prompt-tokens 16 --new-tokens 4 --runs 5 \
  --output results/cached-smoke.json
```

Leave off `--kv-cache` to use the uncached baseline. Cached runs automatically compare every measured trial's generated tokens with an untimed uncached run of the same model and dtype. A mismatch prevents the result from being saved.

This strict check is useful for this experiment. Different floating-point operation shapes can sometimes change a close argmax decision; a mismatch needs investigation rather than an automatic speedup claim. Exact agreement on one prompt also doesn't establish model quality. `--eval-text` remains available for the separate FP32 quality checks from Version 2.

## Sweep prompt lengths locally

```bash
miniinfer-cache-sweep --smoke --device cpu \
  --lengths 8 32 128 --new-tokens 4 --runs 5 --warmup 2 \
  --output-dir results/my-cache-sweep
```

The sweep keeps FP32, batch size, prompt text, output length, and seed fixed. It resolves the pretrained model revision once, then uses that revision for all child runs. Each cached and uncached experiment runs in a fresh process so allocator state from one can't carry into the other. Their order is shuffled with a fixed seed at each length.

The output directory contains:

- A raw JSON report for each length and cache mode.
- `sweep.json`, containing matched results, settings, and speedup ratios.
- `latency.png`, showing generation latency and time to first token.
- `memory.png`, showing final K/V tensor bytes and peak CUDA tensor allocation.

Use a new output directory for every sweep. If a child run fails, earlier raw reports stay available, but no completed sweep is claimed. `--no-plots` runs without Matplotlib. You can plot the saved summary later:

```bash
miniinfer-cache-plot results/my-cache-sweep/sweep.json \
  --output-dir results/my-cache-plots
```

## Run the full GPU experiment

The planned lengths are 128, 256, 512, 1024, and 2048 tokens. The model must have room for the longest prompt **plus** the output. GPT-2's context window is too small for the full sweep.

For example, [SmolLM2-135M's configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json) specifies an 8192-token context window:

```bash
miniinfer-cache-sweep --device cuda --model HuggingFaceTB/SmolLM2-135M \
  --lengths 128 256 512 1024 2048 --new-tokens 32 \
  --batch-size 1 --runs 30 --warmup 3 \
  --output-dir results/cache-gpu
```

This command downloads the pretrained model on first use. The default sweep model is GPT-2, so explicitly select a model with enough context for the full range, or use shorter lengths. The smoke model has a 256-token context limit.

Use an otherwise idle GPU. Repeat whole sweeps and save the model revision, dependency versions, GPU/driver details, and power settings before drawing conclusions.

## Read the memory numbers carefully

Each generation trial reports `cache.sequence_length` and `cache.tensor_bytes` after timing ends.

For an ordinary full-attention cache:

```text
K/V bytes = 2 × layers × batch × KV heads × cached tokens × head size × bytes per value
```

The factor of two is for keys and values. Grouped-query attention can have fewer KV heads than query heads, so the test suite checks that case using a small Llama model.

At the end of generation, cached length is `prompt_tokens + new_tokens - 1`. The final emitted token hasn't been fed back into the model yet, so its K/V values haven't been computed.

`cache.tensor_bytes` counts live K/V tensor elements. It excludes the model, attention temporaries, Python objects, and allocator overhead. Dynamic cache growth can briefly need extra memory while concatenating old and new tensors.

Peak CUDA allocated/reserved memory still measures the whole benchmark process. Those peaks are captured before the uncached parity check or any FP32 quality reference is loaded. CPU/MPS runs leave CUDA measurements unavailable; the plot labels them as unavailable rather than drawing a zero-memory result.

## Timing and correctness

The timing boundaries match the earlier benchmark. Cache construction and initial mask/position creation are inside cached TTFT. Updating masks, positions, and the cache is included in total generation latency. First generation and warmup remain outside the steady-state summary.

Tests compare cached output with both uncached output and Hugging Face generation for small GPT-2 and Llama models. They also check positions, mask lengths, one-token decode inputs, cache isolation, context limits, exact tensor sizes, parity failures, and the sweep/plot workflow.

The initial prompt still needs a full forward pass, so caching mainly targets later tokens. The latency plot separates TTFT from total generation time to make that distinction visible.

## Saved local check

The [CPU smoke sweep](../results/cache-smoke-cpu/sweep.json) covers prompt lengths 8, 32, and 128 with four output tokens. Its [latency plot](../results/cache-smoke-cpu/latency.png) and [memory plot](../results/cache-smoke-cpu/memory.png) demonstrate the workflow. They use a tiny random GPT-2 and aren't pretrained GPU performance results.

## Saved CUDA check

On the Runpod RTX 4090, the cached CUDA parity tests passed for both small GPT-2 and small Llama-style models. This check is important because it verifies that cached decoding returns the same greedy tokens as the uncached path on CUDA.

This is still a correctness result, not a performance result. The real Version 4 benchmark is the prompt-length sweep on a pretrained model.

## Code to read

Start with [generation.py](../src/miniinfer/generation.py), then [cache_sweep.py](../src/miniinfer/cache_sweep.py) and [test_cache.py](../tests/test_cache.py). The plot code is in [cache_plot.py](../src/miniinfer/cache_plot.py).

The [Transformers 4.56.2 caching guide](https://huggingface.co/docs/transformers/v4.56.2/cache_explanation) explains the cache API used here.
