"""Plot only saved KV-cache sweep measurements; missing GPU metrics stay missing."""

import argparse
import json
from pathlib import Path


def plot_sweep(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if report.get("experiment") != "v4-kv-cache-length-sweep" or not report.get("cases"):
        raise ValueError("Expected a nonempty Version 4 cache sweep report")
    directory = Path(directory)
    paths = [directory / "latency.png", directory / "memory.png"]
    if any(path.exists() for path in paths):
        raise FileExistsError("Plot files already exist; choose a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    cases = sorted(report["cases"], key=lambda case: case["prompt_tokens"])
    lengths = [case["prompt_tokens"] for case in cases]
    label = "Random-model smoke check" if report["smoke_only"] else report["model"]["id"]
    subtitle = (f"{label} · {report['environment']['device']} · FP32 · "
                f"batch {report['config']['batch_size']} · {report['config']['new_tokens']} output tokens")
    styles = {"uncached": {"color": "#526779", "marker": "o", "label": "No cache"},
              "cached": {"color": "#087f8c", "marker": "s", "label": "KV cache"}}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        figure.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.80, wspace=0.30)
        for axis, key, title in zip(axes, ("latency_p50_ms", "ttft_median_ms"), ("Generation latency (median)", "Time to first token (median)")):
            for mode, style in styles.items():
                axis.plot(lengths, [case[mode]["summary"][key] for case in cases], **style)
            axis.set(xlabel="Prompt length (tokens)", ylabel="Milliseconds", title=title)
            axis.set_ylim(bottom=0)
            axis.grid(alpha=0.2)
            axis.legend()
        figure.suptitle(subtitle, fontsize=11, y=0.97)
        with paths[0].open("xb") as handle:
            figure.savefig(handle, format="png", dpi=160)
        plt.close(figure)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        figure.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.80, wspace=0.30)
        for mode, style in styles.items():
            axes[0].plot(lengths, [case[mode]["cache"]["tensor_bytes"] / 2**20 for case in cases], **style)
        axes[0].set(title="Final live K/V tensors", xlabel="Prompt length (tokens)", ylabel="MiB")
        axes[0].legend()
        available = all(case[mode]["memory"]["peak_cuda_allocated_bytes"] is not None for case in cases for mode in styles)
        if available:
            for mode, style in styles.items():
                axes[1].plot(lengths, [case[mode]["memory"]["peak_cuda_allocated_bytes"] / 2**20 for case in cases], **style)
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, "CUDA memory unavailable\non this device", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set(title="Peak CUDA tensor allocation", xlabel="Prompt length (tokens)", ylabel="MiB")
        if not available:
            axes[1].set_axis_off()
        for axis in axes:
            axis.set_ylim(bottom=0)
            axis.grid(alpha=0.2)
        figure.suptitle(subtitle, fontsize=11, y=0.97)
        with paths[1].open("xb") as handle:
            figure.savefig(handle, format="png", dpi=160)
        plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        plot_sweep(json.loads(args.report.read_text()), args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
