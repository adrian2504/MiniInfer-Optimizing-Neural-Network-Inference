# Sequential delivery

| Version | Deliverable | Gate before publishing results |
| --- | --- | --- |
| 1 | Eager FP32 harness, raw metrics, CPU correctness CI | Validate against HF greedy generation; run real pretrained CUDA baseline |
| 2 | Compile, FP16/BF16, INT8/INT4 comparisons | Reuse workloads; measure held-out quality and compilation cost |
| 3 | Reference/compiled/Triton RMSNorm | Shape/dtype correctness; kernel timing; separate model integration |
| 4 | Explicit KV cache and length plots | Cached/uncached token parity; context bounds; memory growth |
| 5 | Queued FastAPI inference and dynamic batching | Concurrency, cancellation, overload tests; repeatable load generator |
| 6 | Profiling and causal analysis | Saved traces; evidence for bottleneck explanation; reproduce comparisons |

Use one milestone at a time. Push the Version 1 scaffold first; add the measured CUDA baseline in a follow-up commit before moving to optimization. Suggested tags are `v0.1.0` through `v0.6.0`, created only when each milestone's gate is satisfied. No tag implies completed GPU experiments before they run.
