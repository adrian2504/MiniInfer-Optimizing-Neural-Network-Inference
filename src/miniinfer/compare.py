"""Compare saved reports only when their workloads and environments match."""

import argparse
import json
import math
from pathlib import Path


def compare_reports(baseline, candidate):
    for report in (baseline, candidate):
        if report.get("schema_version") != 2:
            raise ValueError("Use Version 2 reports; rerun the baseline with the current benchmark")
    optimization = baseline["optimization"]
    if (optimization["execution"], optimization["dtype"], optimization["quantization"]) != ("eager", "fp32", "none"):
        raise ValueError("The baseline must use eager FP32 without quantization")
    if optimization.get("kv_cache", False):
        raise ValueError("The baseline must have KV caching disabled")
    if candidate["optimization"].get("kv_cache", False) and not (candidate.get("cache_validation") or {}).get("passed"):
        raise ValueError("The cached candidate must pass token parity validation")
    fields = {
        "config": ("prompt_tokens", "new_tokens", "batch_size", "seed", "threads", "runs", "warmup"),
        "environment": ("device", "device_name", "platform", "python", "torch", "transformers", "cuda_runtime", "torch_threads", "git_commit"),
        "model": ("id", "resolved_revision"),
    }
    for section, keys in fields.items():
        for key in keys:
            if baseline[section][key] != candidate[section][key]:
                raise ValueError(f"Reports differ in {section}.{key}; rerun with matching settings")
    for key in ("smoke_only", "input_ids_sha256"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Reports differ in {key}")
    if not baseline["smoke_only"] and not baseline["model"]["resolved_revision"]:
        raise ValueError("A resolved model revision is required for pretrained comparisons")
    def positive(value):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Timing and throughput values must be finite and positive")
        return value
    def ratio(section, key, inverse=False):
        left, right = baseline[section][key], candidate[section][key]
        if left is None or right is None:
            return None
        left, right = positive(left), positive(right)
        return left / right if inverse else right / left
    return {
        "smoke_only": baseline["smoke_only"],
        "candidate": candidate["optimization"],
        "output_throughput_speedup": ratio("summary", "output_tokens_per_second"),
        "latency_speedup": ratio("summary", "latency_mean_ms", inverse=True),
        "ttft_speedup": ratio("summary", "ttft_median_ms", inverse=True),
        "cuda_allocated_memory_ratio": ratio("memory", "peak_cuda_allocated_bytes"),
        "baseline_first_generation_ms": baseline["startup"]["first_generation_ms"],
        "candidate_first_generation_ms": candidate["startup"]["first_generation_ms"],
        "quality": candidate["quality"],
        "note": "Ratios describe these runs, not statistical significance. Smoke runs do not establish GPU speedups. Quality is null unless requested.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        result = compare_reports(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
