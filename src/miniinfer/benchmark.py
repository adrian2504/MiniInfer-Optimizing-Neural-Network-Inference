"""Compare inference optimizations using fixed-length decoding without a KV cache."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import tempfile

import torch
import transformers
from transformers import AutoTokenizer

from miniinfer.metrics import summarize
from miniinfer.telemetry import GpuSampler
from miniinfer.optimization import load_model, optimization_metadata, validate_optimization, compiler_snapshot
from miniinfer.quality import prepare_evaluation, save_candidate, compare_quality


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def generate_trial(model, input_ids, new_tokens):
    if new_tokens < 1:
        raise ValueError("new_tokens must be positive")
    device = input_ids.device
    sequence = input_ids
    synchronize(device)
    start = time.perf_counter()
    first_token_time = None
    for step in range(new_tokens):
        output = model(input_ids=sequence, use_cache=False)
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sequence = torch.cat((sequence, token), dim=1)
        del output
        if step == 0:
            synchronize(device)
            first_token_time = time.perf_counter()
    synchronize(device)
    end = time.perf_counter()
    return {
        "ttft_ms": (first_token_time - start) * 1000,
        "latency_ms": (end - start) * 1000,
        "output_ids": sequence[:, input_ids.shape[1]:].cpu().tolist(),
    }


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    for field in ("prompt_tokens", "new_tokens", "batch_size", "runs", "threads"):
        if getattr(args, field) < 1:
            raise ValueError(f"{field} must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu for a smoke check")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    validate_optimization(args, device)
    if args.eval_text:
        args.check_quality = True
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = load_model(args, device)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    tokenizer = None
    if args.smoke:
        generator = torch.Generator().manual_seed(args.seed)
        ids = torch.randint(2, model.config.vocab_size, (1, args.prompt_tokens), generator=generator)
        model_id = "random-tiny-gpt2-smoke-only"
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=resolved_revision or args.revision,
                                                 trust_remote_code=False)
        encoded = tokenizer.encode(args.prompt, add_special_tokens=False)
        if not encoded:
            raise ValueError("Prompt must tokenize to at least one token")
        repeated = encoded * ((args.prompt_tokens + len(encoded) - 1) // len(encoded))
        ids = torch.tensor([repeated[:args.prompt_tokens]], dtype=torch.long)
        model_id = args.model
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit and args.prompt_tokens + args.new_tokens > context_limit:
        raise ValueError(f"Prompt plus output exceeds model context limit {context_limit}")
    evaluation = prepare_evaluation(args, tokenizer, context_limit) if args.check_quality else None
    optimization = optimization_metadata(args, model)
    ids = ids.repeat(args.batch_size, 1).to(device)
    compile_before = compiler_snapshot() if args.compile else None
    setup_start = time.perf_counter()
    if args.compile:
        # The sequence grows at every step; dynamic shapes avoid a graph per length.
        model = torch.compile(model, backend="inductor", dynamic=True)
    synchronize(device)
    setup_ms = (time.perf_counter() - setup_start) * 1000
    first_run = generate_trial(model, ids, args.new_tokens)
    warmup_start = time.perf_counter()
    for _ in range(args.warmup):
        generate_trial(model, ids, args.new_tokens)
    synchronize(device)
    warmup_ms = (time.perf_counter() - warmup_start) * 1000
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    compile_warm = compiler_snapshot() if args.compile else None
    with GpuSampler(device) as sampler:
        samples = [generate_trial(model, ids, args.new_tokens) for _ in range(args.runs)]
    if args.compile:
        compile_after = compiler_snapshot()
        if compile_after["unique_graphs"] != compile_warm["unique_graphs"]:
            raise ValueError("Compilation occurred during measured trials; increase --warmup and rerun")
        optimization["compiler_diagnostics"] = {
            key: compile_after[key] - compile_before[key] for key in compile_after
        }
        if optimization["compiler_diagnostics"]["unique_graphs"] == 0:
            raise ValueError("No compiled graphs were captured; rerun in a fresh process and check compiler settings")
    report = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": f"v2-{optimization['execution']}-{args.dtype}-{args.quantization}-no-cache",
        "optimization": optimization,
        "startup": {
            "wrapper_setup_ms": setup_ms,
            "first_generation_ms": first_run["latency_ms"],
            "first_ttft_ms": first_run["ttft_ms"],
            "warmup_total_ms": warmup_ms,
            "note": "First generation includes lazy compilation when enabled, plus execution; not pure compile time. Disk compiler caches may be reused.",
        },
        "smoke_only": args.smoke,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": {"id": model_id, "resolved_revision": resolved_revision},
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "git_commit": git_revision(), "torch_threads": torch.get_num_threads(),
        },
        "input_ids_sha256": hashlib.sha256(json.dumps(ids.cpu().tolist()).encode()).hexdigest(),
        "summary": summarize(samples, args.batch_size, args.new_tokens),
        "memory": {
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        },
        "telemetry": sampler.report(),
        "samples": samples,
    }
    # Snapshot memory and timing above before quality work or loading a reference.
    report["quality"] = None
    if evaluation:
        sequences, dataset = evaluation
        with tempfile.TemporaryDirectory(prefix="miniinfer-quality-") as directory:
            save_candidate(model, sequences, device, directory)
            del model
            if args.compile:
                torch.compiler.reset()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            reference = load_model(args, device, reference=True, revision=resolved_revision)
            reference_trial = generate_trial(reference, ids, args.new_tokens)
            report["quality"] = {
                "dataset": dataset,
                **compare_quality(reference, sequences, device, directory,
                                  reference_trial["output_ids"], samples[0]["output_ids"]),
            }
            del reference
    serialized = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve earlier experiments instead of silently replacing them.
    with args.output.open("x") as handle:
        handle.write(serialized)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--compile", action="store_true", help="Compile the model with TorchInductor")
    parser.add_argument("--quantization", choices=["none", "int8", "int4"], default="none",
                        help="int4 uses bitsandbytes NF4; CUDA pretrained models only")
    parser.add_argument("--check-quality", action="store_true", help="Compare against eager FP32 after timing")
    parser.add_argument("--eval-text", type=Path, help="Held-out UTF-8 text, one document per line; enables quality checks")
    parser.add_argument("--eval-tokens", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--revision", default="main", help="Use a model commit SHA for reproducible experiments")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--prompt", default="Explain how a transformer computes attention.")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Offline random tiny model; no performance claims")
    parser.add_argument("--output", type=Path, default=Path("results") / f"baseline-{time.time_ns()}.json")
    args = parser.parse_args()
    try:
        report = run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved raw samples and metadata to {args.output}")


if __name__ == "__main__":
    main()
