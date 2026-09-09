"""Run matched cached/uncached FP32 experiments in separate processes."""

import argparse
from datetime import datetime, timezone
from importlib import util
import json
from pathlib import Path
import random
import subprocess
import sys

from transformers import AutoConfig

from miniinfer.compare import compare_reports


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if (not args.lengths or min(args.lengths) < 1 or len(set(args.lengths)) != len(args.lengths)
            or min(args.new_tokens, args.batch_size, args.runs, args.threads) < 1 or args.warmup < 0):
        raise ValueError("Use unique positive lengths, positive counts, and nonnegative warmup")
    if not args.no_plots and util.find_spec("matplotlib") is None:
        raise ValueError("Install plots with: pip install -e '.[plots]' (or use --no-plots)")
    if args.smoke:
        context_limit, revision = 256, args.revision
    else:
        config = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        context_limit = getattr(config, "max_position_embeddings", None)
        revision = getattr(config, "_commit_hash", None)
        if not revision:
            raise ValueError("The sweep requires a resolved Hugging Face model revision")
    if context_limit and max(args.lengths) + args.new_tokens > context_limit:
        raise ValueError(f"Prompt plus output exceeds context limit {context_limit}; choose shorter --lengths or a larger-context model")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = []
    order_generator = random.Random(args.seed)
    for length in sorted(args.lengths):
        modes = ["uncached", "cached"]
        order_generator.shuffle(modes)
        reports = {}
        for mode in modes:
            destination = args.output_dir / f"{length}-{mode}.json"
            command = [sys.executable, "-m", "miniinfer.benchmark", "--device", args.device,
                       "--model", args.model, "--revision", revision, "--dtype", "fp32",
                       "--prompt", args.prompt, "--prompt-tokens", str(length),
                       "--new-tokens", str(args.new_tokens), "--batch-size", str(args.batch_size),
                       "--runs", str(args.runs), "--warmup", str(args.warmup),
                       "--threads", str(args.threads), "--seed", str(args.seed),
                       "--output", str(destination)]
            if args.smoke:
                command.append("--smoke")
            if mode == "cached":
                command.append("--kv-cache")
            print(f"Prompt {length}: {mode}", flush=True)
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise ValueError(f"{mode} run at length {length} failed. Earlier raw reports remain in {args.output_dir}.\n{completed.stderr}")
            reports[mode] = json.loads(destination.read_text())
        comparison = compare_reports(reports["uncached"], reports["cached"])
        if reports["uncached"]["samples"][0]["output_ids"] != reports["cached"]["samples"][0]["output_ids"]:
            raise ValueError("Separate processes produced different tokens; no validated sweep was saved")
        cases.append({
            "prompt_tokens": length, "run_order": modes, "comparison": comparison,
            **{mode: {"report_file": f"{length}-{mode}.json", "summary": report["summary"],
                      "memory": report["memory"], "cache": report["samples"][0]["cache"]}
               for mode, report in reports.items()},
        })
    report = {
        "schema_version": 1, "experiment": "v4-kv-cache-length-sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "smoke_only": args.smoke,
        "model": reports["uncached"]["model"], "environment": reports["uncached"]["environment"],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "context_limit": context_limit, "cases": cases,
    }
    with (args.output_dir / "sweep.json").open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not args.no_plots:
        from miniinfer.cache_plot import plot_sweep
        plot_sweep(report, args.output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Explain how a transformer computes attention.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved matched runs and sweep.json to {args.output_dir}")


if __name__ == "__main__":
    main()
