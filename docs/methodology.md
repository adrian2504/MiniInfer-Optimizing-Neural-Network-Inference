# Measurement protocol

- **TTFT:** wall-clock time from preloaded token IDs on the selected device through the first forward pass, argmax, and append, with a device synchronization. Excludes tokenization, transfer, model loading, warmup, networking, and queueing.
- **Generation latency:** synchronized wall-clock time for all requested new tokens, including Python dispatch, the TTFT synchronization, argmax, and concatenation. Token IDs are copied to CPU after timing ends.
- **Output tokens/sec:** total emitted tokens across measured batches divided by summed batch generation time, including prefill. Not the arithmetic mean of per-trial rates.
- **Decode tokens/sec:** tokens after the first token divided by summed latency minus TTFT. Null for one-token output.
- **Requests/sec:** completed sequences divided by summed batch latency. This serial benchmark does not measure a concurrent server's capacity.
- **Percentiles:** linear interpolation of trial batch latencies. p95/p99 from a small sample are unstable; use larger run counts for published claims.
- **CUDA memory:** peak allocated tensor memory and peak reserved allocator memory after warmup and peak-stat reset, including resident model weights. These differ from device-wide VRAM use.
- **GPU utilization:** optional NVML polling every 50 ms over measured trials. NVIDIA's own underlying sampling window applies. Samples include other processes and inter-trial gaps; short runs may give few or unrepresentative readings. Missing NVML, unsupported device handles, or non-CUDA execution produces nulls and a reason. Telemetry polling may add overhead.

Use a dedicated, otherwise idle GPU. Record power/clock settings, driver (`nvidia-smi`), model commit, dependency freeze, GPU model, and ambient competing load with published experiments. Keep model, prompt token count, seed, batch size, and generated token count fixed across comparisons. Separate warmup/compilation from measured trials. Repeat entire runs to characterize variability; do not pick the fastest trial.

The baseline uses explicit eager attention and no cache, so it should be described as this controlled baseline, not as the fastest default Hugging Face configuration. Synthetic repeated prompts and forced output length isolate performance; use a separate held-out natural-text corpus for quality evaluation in Version 2.

Reference APIs: [PyTorch synchronization](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize), [CUDA memory metrics](https://docs.pytorch.org/docs/stable/cuda), and [Hugging Face caching](https://huggingface.co/docs/transformers/v4.50.0/en/cache_explanation).
