# Version 6 — Finding where the time goes

Benchmarks tell me whether a change helped. Profiles help me investigate why.

This version adds a separate profiler command for generation and RMSNorm. It saves a trace, an operator summary, and a short table of observations. It doesn't automatically label an operation memory-bound or claim a speedup from profiler timings.

## Start with a CPU profile

```bash
miniinfer-profile --smoke --workload generation \
  --output-dir results/profile-uncached

miniinfer-profile --smoke --workload generation --kv-cache \
  --output-dir results/profile-cached
```

Each output directory contains:

- `trace.json`: a Chrome-format execution trace for a compatible trace viewer.
- `profile.json`: settings, environment, operator call counts, self times, and memory deltas.
- `observations.md`: the top operators by self time, with a reminder of the measurement limits.

The repository also contains a [CPU smoke profile](../results/profile-smoke-cpu/observations.md). Its random model is a workflow check.

## Profile a kernel

```bash
miniinfer-profile --workload rmsnorm --rows 32 --width 1024 \
  --implementation eager --output-dir results/profile-rmsnorm-eager

miniinfer-profile --workload rmsnorm --rows 32 --width 1024 \
  --implementation compiled --output-dir results/profile-rmsnorm-compiled
```

Use `--device cuda --implementation triton` for the custom kernel on an NVIDIA machine. Compilation and warmup happen before tracing. The profile itself has overhead, so run `miniinfer-kernel` separately for timing comparisons.

Generation profiling currently supports eager execution with or without the KV cache. Its prompts are seeded synthetic token IDs. Match model, revision, dtype, seed, and dimensions when comparing traces.

## What to look for

1. Start with an unprofiled benchmark showing the change you want to explain.
2. Find expensive operators in `observations.md` and inspect them in the trace.
3. Look at matrix sizes, allocation patterns, and the number of launches.
4. Form a hypothesis, such as fewer intermediate tensors or less repeated sequence processing.
5. Check it against the trace or GPU counters, then repeat the unprofiled benchmark.

The generation profile wraps the entire generation helper, including output copying and cache accounting that are outside the benchmark timer. Self time excludes nested operator time. Memory deltas can be negative when tensors are freed; they aren't peak VRAM. Summing operator times also doesn't necessarily give wall-clock time when work overlaps.

For cached decoding, inspect the matrix shapes after prefill: later passes should process one new token. For RMSNorm, investigate whether compiled or Triton execution reduces separate operations and intermediate allocations. A lower latency alone doesn't prove a bandwidth bottleneck.

## NVIDIA Nsight Systems

On a Linux NVIDIA machine with Nsight Systems installed:

```bash
nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output=results/generation-nsys \
  miniinfer-profile --device cuda --smoke --kv-cache --external \
  --output-dir results/generation-nsys-metadata
```

The `--external` path warms up first, then brackets the workload with CUDA profiler start/stop calls and NVTX ranges. It disables PyTorch tracing so both tools aren't competing for the same CUDA activity.

Open the resulting `.nsys-rep` in Nsight Systems. Look at launch gaps, CPU dispatch, CUDA kernels, transfers, and synchronization. Metadata and the external-tool reminder are saved separately; no PyTorch trace is generated in this mode.

## NVIDIA Nsight Compute

For a small Triton kernel capture:

```bash
ncu --profile-from-start off --set full --launch-count 1 \
  --export results/rmsnorm-ncu \
  miniinfer-profile --workload rmsnorm --device cuda --implementation triton \
  --rows 32 --width 1024 --steps 1 --external \
  --output-dir results/rmsnorm-ncu-metadata
```

Nsight Compute provides hardware counters for an individual kernel. Inspect memory throughput, compute throughput, occupancy, and resource usage. Its collection and replay costs make it unsuitable as the source of end-to-end latency claims. GPU counter access may depend on the cloud machine's configuration.

The NVIDIA capture paths require NVIDIA hardware and installed Nsight tools. The CUDA test suite has been verified on Runpod, but these Nsight captures still need to be run separately because Runpod does not automatically include the Nsight tools in every template.

## Record a conclusion

For each optimization, write down the workload and hardware, the unprofiled result, the trace/counter evidence, and what remains uncertain. Keep the original reports. A useful conclusion connects a specific change to a measured execution difference and a separately measured performance result.

References: [PyTorch 2.8 profiler](https://docs.pytorch.org/docs/2.8/profiler.html), [Nsight Systems capture example](https://docs.nvidia.com/jax-toolbox/performance-profiling/profiling), and [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/).
