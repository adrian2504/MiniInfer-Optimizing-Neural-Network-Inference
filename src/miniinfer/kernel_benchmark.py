"""Benchmark RMSNorm implementations, independently of transformer generation."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import torch

from miniinfer.kernels.rmsnorm import MAX_WIDTH, rmsnorm_math, triton_rmsnorm, validate_inputs
from miniinfer.metrics import percentile
from miniinfer.optimization import compiler_snapshot

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
TOLERANCES = {torch.float32: (1e-5, 1e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}


def parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use dimensions such as 32x1024 or 2x16x128") from exc
    if not shape or any(size < 1 for size in shape) or shape[-1] > MAX_WIDTH:
        raise argparse.ArgumentTypeError(f"Dimensions must be positive; last dimension must be <= {MAX_WIDTH}")
    return shape


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def check_output(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise ValueError("RMSNorm output shape, dtype, or device differs from the reference")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("RMSNorm produced non-finite values")
    rtol, atol = TOLERANCES[expected.dtype]
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise ValueError(f"RMSNorm correctness check failed: {exc}") from exc
    difference = (actual.double() - expected.double()).abs()
    return {"passed": True, "rtol": rtol, "atol": atol,
            "max_absolute_error": difference.max().item(),
            "mean_absolute_error": difference.mean().item()}


def time_calls(function, device, iterations):
    # Each sample is an average over repeated calls, not one isolated launch.
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        elapsed_ms = (time.perf_counter() - start) * 1000
    average = elapsed_ms / iterations
    if not math.isfinite(average) or average <= 0:
        raise ValueError("Invalid timing sample; increase --iterations")
    return average


@torch.inference_mode()
def benchmark_case(shape, dtype_name, args, device):
    # Generate on CPU so each implementation receives exactly the same tensors.
    generator = torch.Generator().manual_seed(args.seed)
    x_cpu = torch.randn(shape, generator=generator).to(DTYPES[dtype_name])
    weight_cpu = torch.randn(shape[-1], generator=generator).to(DTYPES[dtype_name])
    input_hash = hashlib.sha256(x_cpu.view(torch.uint8).numpy().tobytes()
                                + weight_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    x, weight = x_cpu.to(device), weight_cpu.to(device)
    validate_inputs(x, weight, args.eps)
    # Independent FP64 oracle, computed from the actual rounded input values.
    oracle = x_cpu.double()
    oracle = (oracle * torch.rsqrt(oracle.square().mean(-1, keepdim=True) + args.eps)
              * weight_cpu.double()).to(x.dtype).to(device)
    functions = {"eager": lambda: rmsnorm_math(x, weight, args.eps)}
    if "compiled" in args.implementations:
        torch.compiler.reset()
        compiled = torch.compile(rmsnorm_math, backend="inductor", fullgraph=True, dynamic=False)
        functions["compiled"] = lambda: compiled(x, weight, args.eps)
    if "triton" in args.implementations:
        functions["triton"] = lambda: triton_rmsnorm(x, weight, args.eps)
    results = {}
    before = compiler_snapshot() if "compiled" in functions else None
    for name, function in functions.items():
        synchronize(device)
        start = time.perf_counter()
        actual = function()
        synchronize(device)
        first_ms = (time.perf_counter() - start) * 1000
        correctness = check_output(actual, oracle)
        del actual
        for _ in range(args.warmup):
            function()
        synchronize(device)
        results[name] = {"correctness": correctness, "first_call_wall_ms": first_ms, "samples_ms": []}
    warm = compiler_snapshot() if before is not None else None
    order_generator = random.Random(args.seed)
    orders = []
    for _ in range(args.runs):
        order = list(functions)
        order_generator.shuffle(order)
        orders.append(order)
        for name in order:
            results[name]["samples_ms"].append(time_calls(functions[name], device, args.iterations))
    if before is not None:
        after = compiler_snapshot()
        if after["unique_graphs"] != warm["unique_graphs"]:
            raise ValueError("Compilation occurred during measurement; increase --warmup")
        if after["unique_graphs"] <= before["unique_graphs"]:
            raise ValueError("No compiled graph was captured")
        results["compiled"]["compiler_diagnostics"] = {key: after[key] - before[key] for key in after}
    baseline_median = statistics.median(results["eager"]["samples_ms"])
    for result in results.values():
        samples = result["samples_ms"]
        result["mean_ms"] = statistics.mean(samples)
        result["median_ms"] = statistics.median(samples)
        result["p95_ms"] = percentile(samples, 0.95)
        result["speedup_vs_eager"] = baseline_median / result["median_ms"]
    return {"shape": list(shape), "dtype": dtype_name, "input_sha256": input_hash,
            "measurement_orders": orders, "implementations": results}


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if min(args.runs, args.iterations, args.warmup, args.threads) < 1:
        raise ValueError("runs, iterations, warmup, and threads must be positive")
    if not math.isfinite(args.eps) or args.eps <= 0:
        raise ValueError("eps must be finite and positive")
    if "eager" not in args.implementations or len(set(args.implementations)) != len(args.implementations):
        raise ValueError("Include eager as the baseline and do not repeat implementations")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or torch.version.hip is not None):
        raise ValueError("An NVIDIA CUDA device is required; use --device cpu --implementations eager compiled locally")
    if "triton" in args.implementations:
        if device.type != "cuda":
            raise ValueError("Triton requires CUDA; use --implementations eager compiled on CPU")
        if util.find_spec("triton") is None:
            raise ValueError("Install Triton with: pip install -e '.[kernels]'")
    if "bf16" in args.dtypes and device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("This GPU lacks native BF16 support; select --dtypes fp32 fp16")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.matmul.allow_tf32 = False
    cases = [benchmark_case(shape, dtype, args, device) for shape in args.shapes for dtype in args.dtypes]
    try:
        triton_version = metadata.version("triton")
    except metadata.PackageNotFoundError:
        triton_version = None
    from miniinfer.benchmark import git_revision
    report = {
        "schema_version": 1, "experiment": "v3-rmsnorm-forward", "scope": "standalone-operation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_performance": device.type == "cuda",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "triton": triton_version, "cuda_runtime": torch.version.cuda,
                        "device": str(device), "device_name": torch.cuda.get_device_name(device)
                        if device.type == "cuda" else platform.processor(), "git_commit": git_revision()},
        "timing": {
            "method": "cuda-events" if device.type == "cuda" else "wall-clock",
            "sample": "elapsed time for iterations repeated calls divided by iterations",
            "cache_policy": "reuse the same input and weight; no cache flush",
            "allocation_policy": "each call allocates output; eager also allocates intermediates",
            "note": "CUDA event intervals can include host dispatch gaps. These are operation timings, not generation speedups. First calls include JIT/compile work and may reuse disk caches.",
        },
        "cases": cases,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(serialized + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--implementations", nargs="+", choices=["eager", "compiled", "triton"],
                        default=["eager", "compiled", "triton"])
    parser.add_argument("--shapes", nargs="+", type=parse_shape, default=[(1, 128), (32, 1024), (128, 4096)])
    parser.add_argument("--dtypes", nargs="+", choices=list(DTYPES), default=["fp32", "fp16", "bf16"])
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results") / f"rmsnorm-{time.time_ns()}.json")
    args = parser.parse_args()
    try:
        report = run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print("shape | dtype | implementation | median ms | speedup vs eager")
    for case in report["cases"]:
        for name, result in case["implementations"].items():
            shape = "x".join(map(str, case["shape"]))
            print(f"{shape} | {case['dtype']} | {name} | {result['median_ms']:.6f} | {result['speedup_vs_eager']:.2f}x")
    print(f"Saved operation timings to {args.output}")


if __name__ == "__main__":
    main()
