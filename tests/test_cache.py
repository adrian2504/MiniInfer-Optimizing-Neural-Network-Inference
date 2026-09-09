import json
import subprocess
import sys

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from miniinfer.benchmark import build_parser, run
from miniinfer.generation import generate_trial


def small_model(kind):
    torch.manual_seed(17)
    if kind == "gpt2":
        config = GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=4,
                            eos_token_id=None, pad_token_id=0)
        config._attn_implementation = "eager"
        return GPT2LMHeadModel(config).eval()
    config = LlamaConfig(vocab_size=64, max_position_embeddings=64, hidden_size=32,
                         intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


@pytest.mark.parametrize("kind", ["gpt2", "llama"])
@pytest.mark.parametrize("batch,prompt,new_tokens", [(1, 1, 1), (2, 7, 4), (1, 60, 4)])
def test_cache_matches_uncached_and_hugging_face(kind, batch, prompt, new_tokens):
    model = small_model(kind)
    ids = torch.randint(2, 64, (batch, prompt))
    before = ids.clone()
    cached = generate_trial(model, ids, new_tokens, kv_cache=True)
    uncached = generate_trial(model, ids, new_tokens)
    expected = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=new_tokens,
                              use_cache=True, do_sample=False, pad_token_id=0)
    assert cached["output_ids"] == uncached["output_ids"] == expected[:, prompt:].tolist()
    assert torch.equal(ids, before)
    assert 0 < cached["ttft_ms"] <= cached["latency_ms"]
    length = prompt + new_tokens - 1
    assert cached["cache"]["sequence_length"] == length
    kv_width = 32 if kind == "gpt2" else 16
    assert cached["cache"]["tensor_bytes"] == 2 * 2 * batch * length * kv_width * 4
    assert uncached["cache"]["tensor_bytes"] == 0


def test_cache_inputs_positions_and_trial_isolation():
    model = small_model("llama")
    calls = []
    def hook(module, args, kwargs):
        calls.append((kwargs["input_ids"].shape[1], kwargs["attention_mask"].shape[1],
                      kwargs["cache_position"].tolist(), kwargs["past_key_values"].get_seq_length(),
                      kwargs["past_key_values"]))
    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    ids = torch.tensor([[3, 7, 9]])
    first = generate_trial(model, ids, 3, kv_cache=True)
    second = generate_trial(model, ids, 3, kv_cache=True)
    handle.remove()
    assert [(n, mask, pos, length) for n, mask, pos, length, _ in calls[:3]] == [
        (3, 3, [0, 1, 2], 0), (1, 4, [3], 3), (1, 5, [4], 4)]
    assert calls[0][-1] is not calls[3][-1]
    assert first["output_ids"] == second["output_ids"]


@pytest.mark.parametrize("cache", [False, True])
def test_context_overflow_rejected_before_forward(cache):
    model = small_model("gpt2")
    with pytest.raises(ValueError, match="context limit"):
        generate_trial(model, torch.ones((1, 63), dtype=torch.long), 2, kv_cache=cache)


def test_cache_cli_and_compile_guard(tmp_path):
    args = build_parser().parse_args(["--smoke", "--device", "cpu", "--kv-cache", "--check-quality",
                                     "--prompt-tokens", "8", "--new-tokens", "3", "--runs", "2",
                                     "--eval-tokens", "8", "--output", str(tmp_path / "cached.json")])
    report = run(args)
    assert report["cache_validation"]["passed"]
    assert report["cache_validation"]["trials_checked"] == 2
    assert report["quality"]["generated_token_agreement"] == 1
    args.output = tmp_path / "invalid.json"
    args.compile = True
    with pytest.raises(ValueError, match="omit --compile"):
        run(args)
    assert not args.output.exists()


def test_cache_parity_failure_prevents_result(tmp_path, monkeypatch):
    import miniinfer.benchmark as benchmark
    original = benchmark.generate_trial
    def different(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("kv_cache"):
            result["output_ids"][0][0] = -1
        return result
    monkeypatch.setattr(benchmark, "generate_trial", different)
    args = build_parser().parse_args(["--smoke", "--device", "cpu", "--kv-cache",
                                     "--prompt-tokens", "8", "--new-tokens", "2", "--runs", "1",
                                     "--output", str(tmp_path / "invalid.json")])
    with pytest.raises(ValueError, match="tokens differ"):
        run(args)
    assert not args.output.exists()


def test_sweep_reports_plots_and_context_guard(tmp_path):
    pytest.importorskip("matplotlib")
    destination = tmp_path / "sweep"
    command = [sys.executable, "-m", "miniinfer.cache_sweep", "--smoke", "--device", "cpu",
               "--lengths", "4", "8", "--new-tokens", "2", "--runs", "2", "--warmup", "1",
               "--output-dir", str(destination)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((destination / "sweep.json").read_text())
    assert [case["prompt_tokens"] for case in report["cases"]] == [4, 8]
    sizes = [case["cached"]["cache"]["tensor_bytes"] for case in report["cases"]]
    assert sizes[1] > sizes[0] > 0
    for case in report["cases"]:
        assert case["comparison"]["output_throughput_speedup"] > 0
        assert case["cached"]["memory"]["peak_cuda_allocated_bytes"] is None
    from PIL import Image
    for name in ("latency.png", "memory.png"):
        with Image.open(destination / name) as picture:
            assert picture.width >= 1000
            picture.verify()
    assert subprocess.run(command, capture_output=True).returncode != 0
    invalid = tmp_path / "too-long"
    failed = subprocess.run([sys.executable, "-m", "miniinfer.cache_sweep", "--smoke", "--device", "cpu",
                             "--lengths", "256", "--output-dir", str(invalid)], capture_output=True, text=True)
    assert failed.returncode != 0
    assert "context limit" in failed.stderr
    assert not invalid.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("kind", ["gpt2", "llama"])
def test_cached_cuda_parity(kind):
    model = small_model(kind).to("cuda")
    ids = torch.tensor([[3, 7, 8], [9, 12, 15]], device="cuda")
    cached = generate_trial(model, ids, 4, kv_cache=True)
    assert cached["output_ids"] == generate_trial(model, ids, 4)["output_ids"]
    assert cached["cache"]["tensor_bytes"] > 0
