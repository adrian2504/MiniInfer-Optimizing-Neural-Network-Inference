# Roadmap

## Version 1 — Baseline

The code is built: eager FP32 generation without a KV cache, timing and memory reports, an offline smoke mode, tests, and CPU CI.

A CPU smoke result is saved. The remaining measurement is a pretrained CUDA run on a named GPU. That will be the reference for later speedups.

## Version 2 — Optimization (code built; GPU experiments pending)

The CLI now has compilation, precision, and quantization options, plus quality checks and a report comparison command. Follow the [Version 2 walkthrough](version-2.md).

The experiment sequence is:

1. Record the pretrained GPU baseline with a fixed model revision and workload.
2. Compare eager execution with `torch.compile`. Record compilation cost separately.
3. Compare FP32, FP16, and BF16 where the hardware supports them.
4. Try INT8 and INT4 quantization.
5. Compare latency, tokens/sec, memory, and quality on the same inputs.

Before calling this complete, save repeatable results and evaluate held-out text. Check perplexity, model-score differences, and greedy-token agreement with FP32.

## Version 3 — A Triton kernel (code built; CUDA verification pending)

RMSNorm is implemented in PyTorch, compiled PyTorch, and Triton. The standalone benchmark checks each shape and dtype against an FP64 reference before timing it. Tests cover numerical edge cases, irregular widths, and input validation.

Follow the [Version 3 walkthrough](version-3.md). Remaining work is to run the CUDA correctness tests and save repeated GPU measurements.

Report kernel timings separately. Measure generation speed only after integrating the kernel into a model that uses RMSNorm.

## Version 4 — KV cache (code built; GPU experiments pending)

Cached decoding, token-parity checks, and the prompt-length sweep are built. A CPU smoke sweep with plots is saved. See the [Version 4 walkthrough](version-4.md).

Remaining work is to run repeated pretrained GPU experiments across prompt lengths of 128, 256, 512, 1024, and 2048 tokens.

Use a model whose context window fits the prompt plus output, and test the context limit.

## Version 5 — Inference server (code built; GPU load experiments pending)

FastAPI, a bounded queue, compatible-request batching, and all four routes are implemented. The load tester supports 1, 10, 50, and 100 clients. CPU tests cover lifecycle, overload, cancellation, and generation. See [Version 5](version-5.md).

Test 1, 10, 50, and 100 concurrent clients. Measure queueing time, latency percentiles, throughput, and GPU utilization. Check cancellation and behavior when the queue fills up.

## Version 6 — Profiling (tools built; NVIDIA captures pending)

PyTorch traces, operator summaries, and observation tables are implemented and checked on CPU. The external capture mode and Nsight commands are ready for NVIDIA verification. See [Version 6](version-6.md).

Run profiling separately from benchmarks, then repeat the comparisons to check that the conclusions hold.

Each version should have its own code, checks, saved measurements, and notes on what I learned. I'll tag milestones once those pieces are complete.
