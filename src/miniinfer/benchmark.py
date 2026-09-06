"""Version 1: eager FP32, fixed-length greedy decoding without a KV cache."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from miniinfer.metrics import summarize
from miniinfer.telemetry import GpuSampler


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
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.smoke:
        config = GPT2Config(
            vocab_size=128, n_positions=256, n_embd=64, n_layer=2, n_head=2,
            bos_token_id=0, eos_token_id=1,
        )
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config)
        ids = torch.randint(2, config.vocab_size, (1, args.prompt_tokens))
        model_id = "random-tiny-gpt2-smoke-only"
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, torch_dtype=torch.float32,
            attn_implementation="eager", trust_remote_code=False,
        )
        encoded = tokenizer.encode(args.prompt, add_special_tokens=False)
        if not encoded:
            raise ValueError("Prompt must tokenize to at least one token")
        repeated = (encoded * ((args.prompt_tokens + len(encoded) - 1) // len(encoded)))
        ids = torch.tensor([repeated[:args.prompt_tokens]], dtype=torch.long)
        model_id = args.model
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit and args.prompt_tokens + args.new_tokens > context_limit:
        raise ValueError(f"Prompt plus output exceeds model context limit {context_limit}")
    model = model.to(device=device, dtype=torch.float32).eval()
    ids = ids.repeat(args.batch_size, 1).to(device)
    for _ in range(args.warmup):
        generate_trial(model, ids, args.new_tokens)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with GpuSampler(device) as sampler:
        samples = [generate_trial(model, ids, args.new_tokens) for _ in range(args.runs)]
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "v1-eager-fp32-no-cache",
        "smoke_only": args.smoke,
        "config": {**vars(args), "output": str(args.output)},
        "model": {"id": model_id, "resolved_revision": getattr(model.config, "_commit_hash", None)},
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve earlier experiments instead of silently replacing them.
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved raw samples and metadata to {args.output}")


if __name__ == "__main__":
    main()
