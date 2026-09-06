import json
import subprocess
import sys

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from miniinfer.benchmark import generate_trial


def test_uncached_loop_matches_transformers_greedy_generation():
    torch.manual_seed(7)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2,
        eos_token_id=None, pad_token_id=0,
    )).eval()
    inputs = torch.tensor([[3, 4, 5], [6, 7, 8]])
    trial = generate_trial(model, inputs, 4)
    expected = model.generate(
        inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=4,
        do_sample=False, use_cache=False, pad_token_id=0,
    )
    assert trial["output_ids"] == expected[:, 3:].tolist()
    assert 0 < trial["ttft_ms"] <= trial["latency_ms"]


def test_cli_produces_a_complete_honest_smoke_report(tmp_path):
    destination = tmp_path / "result.json"
    command = [sys.executable, "-m", "miniinfer.benchmark", "--smoke", "--device", "cpu",
               "--prompt-tokens", "8", "--new-tokens", "2", "--runs", "2", "--warmup", "1",
               "--output", str(destination)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(destination.read_text())
    assert report["smoke_only"] is True
    assert len(report["samples"]) == 2
    assert report["memory"]["peak_cuda_allocated_bytes"] is None
    assert report["telemetry"]["mean_gpu_utilization_percent"] is None
    assert report["samples"][0]["output_ids"] == report["samples"][1]["output_ids"]
    assert report["summary"]["output_tokens_per_second"] > 0
    original = destination.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert destination.read_bytes() == original


def test_invalid_token_count():
    with pytest.raises(ValueError):
        generate_trial(None, torch.tensor([[1]]), 0)
