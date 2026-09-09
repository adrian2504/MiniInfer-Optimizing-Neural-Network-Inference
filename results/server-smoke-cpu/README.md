# Local server smoke checks

These reports used the tiny random GPT-2 on CPU, with 100 requests each at 1, 10, 50, and 100 concurrent clients. All 800 requests across the two routes returned HTTP 200.

They check the HTTP and batching workflow. These short local runs were captured during development, at different times, and do not establish a controlled performance comparison or GPU capacity. No model quality claims follow from random weights.

The raw files retain client timings, server timings, status counts, and telemetry availability. See the [Version 5 guide](../../docs/version-5.md) before running repeatable performance experiments.
