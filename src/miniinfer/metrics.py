"""Aggregation kept independent of accelerator and model libraries."""

import math
import statistics


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("Need samples and a quantile in [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(samples, batch_size, new_tokens):
    if not samples or batch_size < 1 or new_tokens < 1:
        raise ValueError("Need samples, positive batch size and token count")
    latencies = [s["latency_ms"] for s in samples]
    total_seconds = sum(latencies) / 1000
    decode_seconds = sum(s["latency_ms"] - s["ttft_ms"] for s in samples) / 1000
    return {
        "ttft_median_ms": statistics.median(s["ttft_ms"] for s in samples),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "requests_per_second": len(samples) * batch_size / total_seconds,
        "output_tokens_per_second": len(samples) * batch_size * new_tokens / total_seconds,
        "decode_tokens_per_second": (
            len(samples) * batch_size * (new_tokens - 1) / decode_seconds
            if new_tokens > 1 and decode_seconds > 0 else None
        ),
    }
