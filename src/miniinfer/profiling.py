"""Separate profiler runs for generation or RMSNorm; never speedup measurements."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import torch

from miniinfer.benchmark import build_parser, git_revision
from miniinfer.generation import generate_trial, synchronize
from miniinfer.kernels.rmsnorm import rmsnorm_math, triton_rmsnorm
from miniinfer.optimization import DTYPES, load_model


def make_workload(args, device):
    torch.manual_seed(args.seed)
    if args.workload == "generation":
        config = build_parser().parse_args([])
        for key in ("model", "revision", "device", "smoke", "seed", "dtype"):
            setattr(config, key, getattr(args, key))
        model = load_model(config, device)
        if args.prompt_tokens + args.new_tokens > model.config.max_position_embeddings:
            raise ValueError("Prompt plus output exceeds the model context limit")
        ids = torch.randint(2, model.config.vocab_size, (args.batch_size, args.prompt_tokens), device=device)
        def workload():
            return generate_trial(model, ids, args.new_tokens, kv_cache=args.kv_cache)
        return workload, {"model": "random-tiny-gpt2" if args.smoke else args.model,
                          "revision": getattr(model.config, "_commit_hash", None),
                          "inputs": "seeded synthetic token IDs"}
    if args.width > 16384:
        raise ValueError("RMSNorm width must be <= 16384")
    x = torch.randn(args.rows, args.width, device=device, dtype=DTYPES[args.dtype])
    weight = torch.randn(args.width, device=device, dtype=x.dtype)
    operation = rmsnorm_math
    if args.implementation == "compiled":
        operation = torch.compile(operation, fullgraph=True)
    elif args.implementation == "triton":
        operation = triton_rmsnorm
    return lambda: operation(x, weight, 1e-6), {"shape": [args.rows, args.width]}


@torch.inference_mode()
def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {args.output_dir}")
    if min(args.steps, args.warmup, args.threads, args.rows, args.width, args.batch_size, args.prompt_tokens, args.new_tokens) < 1:
        raise ValueError("All counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu")
    if args.workload == "rmsnorm" and args.implementation == "triton" and device.type != "cuda":
        raise ValueError("Triton requires CUDA")
    if args.workload == "generation" and args.implementation != "eager":
        raise ValueError("Generation profiling supports eager execution with or without --kv-cache")
    if args.workload == "rmsnorm" and args.kv_cache:
        raise ValueError("KV caching applies to generation only")
    if args.external and device.type != "cuda":
        raise ValueError("External Nsight capture requires CUDA")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.matmul.allow_tf32 = False
    workload, details = make_workload(args, device)
    for _ in range(args.warmup):
        workload()
    synchronize(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    # Nsight runs omit Kineto so two profilers do not compete for CUDA tracing.
    if args.external:
        if device.type != "cuda":
            raise ValueError("External Nsight capture requires CUDA")
        torch.cuda.cudart().cudaProfilerStart()
        try:
            for _ in range(args.steps):
                with torch.cuda.nvtx.range(args.workload):
                    workload()
            synchronize(device)
        finally:
            torch.cuda.cudart().cudaProfilerStop()
        events = []
    else:
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True,
                                    with_stack=False) as profiler:
            for _ in range(args.steps):
                with torch.profiler.record_function(f"miniinfer::{args.workload}"):
                    workload()
                profiler.step()
        profiler.export_chrome_trace(str(args.output_dir / "trace.json"))
        events = [{"name": event.key, "calls": event.count,
                   "self_cpu_time_us": event.self_cpu_time_total,
                   "self_device_time_us": event.self_device_time_total,
                   "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                   "self_device_memory_bytes": event.self_device_memory_usage}
                  for event in profiler.key_averages()]
    field = "self_device_time_us" if device.type == "cuda" else "self_cpu_time_us"
    events.sort(key=lambda event: event[field], reverse=True)
    report = {"schema_version": 1, "experiment": "v6-profile", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "config": {**vars(args), "output_dir": str(args.output_dir)}, "workload": details,
              "environment": {"torch": torch.__version__, "python": platform.python_version(),
                              "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
                              "device": str(device), "git_commit": git_revision()},
              "events": events, "note": "Profiled execution includes tracing overhead. Operator self times are not end-to-end speedups or proof of a bandwidth bottleneck."}
    (args.output_dir / "profile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Profile observations", "", "This is an instrumented run, not a benchmark.", "",
             "| Operation | Calls | Self CPU time (us) | Self device time (us) |",
             "| --- | ---: | ---: | ---: |"]
    lines += [f"| {event['name']} | {event['calls']} | {event['self_cpu_time_us']:.2f} | {event['self_device_time_us']:.2f} |"
              for event in events[:15]]
    lines += ["", "Compare separate benchmark reports before claiming a speedup. Use the trace to inspect allocations and launches. Nsight Compute counters are needed to investigate memory bandwidth and occupancy."]
    (args.output_dir / "observations.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=["generation", "rmsnorm"], default="generation")
    parser.add_argument("--implementation", choices=["eager", "compiled", "triton"], default="eager")
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--kv-cache", action="store_true")
    parser.add_argument("--external", action="store_true", help="Nsight capture range; disable PyTorch tracing")
    for name, default in (("steps", 3), ("warmup", 2), ("threads", 4), ("rows", 32), ("width", 1024),
                          ("batch-size", 1), ("prompt-tokens", 16), ("new-tokens", 4), ("seed", 42)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved profiling artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
