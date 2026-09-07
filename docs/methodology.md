# How I measure performance

The goal is to compare the same work before and after each change.

## What is timed?

The model is loaded and the input tokens are already on the selected device before timing starts. Warmup runs happen first and aren't included in the results.

The timer includes model execution, picking the next token, appending it, and the Python work in the generation loop. It waits for GPU work to finish before recording the first-token and final times.

Downloads, tokenization, model loading, input transfer, copying the final output to CPU, and writing JSON are outside the timer. Networking and queueing aren't part of this benchmark yet.

## Metrics

| Metric | Meaning |
| --- | --- |
| TTFT | Time through the first model pass, token selection, and append |
| Generation latency | Time to generate all requested tokens in a batch |
| Output tokens/sec | Total generated tokens divided by total generation time |
| Decode tokens/sec | Tokens after the first token divided by total time after TTFT |
| Requests/sec | Completed sequences divided by total batch generation time |
| p50 / p95 / p99 | Latency values at the 50th, 95th, and 99th percentiles of measured batches |

Rates use totals across trials, rather than averaging each trial's rate. Decode tokens/sec is `null` when only one token is generated.

Percentiles use linear interpolation between sorted samples. A few runs aren't enough to make reliable claims about p95 or p99. Requests/sec here comes from serial batches; it doesn't measure concurrent server capacity.

## Memory and GPU utilization

CUDA reports two process-level peaks after warmup:

- **Allocated memory:** memory used by PyTorch tensors, including the loaded model.
- **Reserved memory:** memory held by PyTorch's allocator, including space it can reuse.

Optional NVIDIA monitoring uses NVML to poll GPU utilization and whole-device memory every 50 ms. These readings can include other processes and gaps between trials. NVIDIA also has its own sampling window, so very short runs may not give useful utilization data. Polling can add some overhead.

If a metric isn't available, the report saves `null` and a reason. CPU and MPS runs don't produce CUDA or NVIDIA readings.

## Keeping comparisons fair

- Use the same model revision, prompt length, output length, batch size, and seed.
- Use the same GPU with other work stopped where possible.
- Save the GPU model, driver, power and clock settings, software versions, and any competing load.
- Keep warmup and compilation separate from measured runs.
- Repeat whole experiments to see how much results vary. Keep all trials.
- Save the JSON report and dependency versions so the experiment can be repeated.

The default model revision is `main`, which can change. Use the resolved commit from a report with `--revision` for later comparisons.

## What the baseline tells me

The baseline explicitly uses eager attention, FP32, and no KV cache. CUDA TF32 is disabled. This gives me a controlled starting point for later changes.

Inputs are repeated or trimmed to a fixed length, every batch item uses the same input, and generation runs for a fixed token count. That makes workloads comparable, but it doesn't represent varied real-world requests.

The random-weight smoke model checks that the code runs. Version 2 quality comparisons will need a pretrained model and a separate held-out natural-text dataset.
