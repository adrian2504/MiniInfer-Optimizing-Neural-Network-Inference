import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from miniinfer.compare import compare_reports
from miniinfer.optimization import load_model, validate_optimization
from miniinfer.quality import compare_quality, prepare_evaluation, save_candidate


def options(**overrides):
    return SimpleNamespace(**({
        "compile": False, "warmup": 1, "dtype": "fp32", "quantization": "none",
        "smoke": True, "seed": 42, "eval_samples": 2, "eval_tokens": 8, "eval_text": None,
    } | overrides))


@pytest.mark.parametrize("overrides,device,message", [
    ({"compile": True, "warmup": 0}, "cpu", "warmup"),
    ({"compile": True}, "mps", "CPU and CUDA"),
    ({"quantization": "int8"}, "cpu", "pretrained model on CUDA"),
    ({"quantization": "int4"}, "cuda", "smoke mode"),
    ({"quantization": "int8", "smoke": False}, "cuda", "dtype"),
])
def test_invalid_combinations(overrides, device, message):
    with pytest.raises(ValueError, match=message):
        validate_optimization(options(**overrides), torch.device(device))


def test_cuda_bf16_requires_native_support(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: False)
    with pytest.raises(ValueError, match="native BF16"):
        validate_optimization(options(dtype="bf16"), torch.device("cuda"))


def test_smoke_precision_preserves_initial_weights():
    reference = load_model(options(), torch.device("cpu"), reference=True)
    candidate = load_model(options(dtype="bf16"), torch.device("cpu"))
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        assert actual.dtype == torch.bfloat16
        assert torch.equal(expected.bfloat16(), actual)


class FixedModel:
    def __init__(self, bias=0.0):
        self.bias = bias

    def __call__(self, input_ids, use_cache):
        assert use_cache is False
        logits = torch.zeros(*input_ids.shape, 4)
        logits[..., 0] = self.bias
        return SimpleNamespace(logits=logits)


def test_quality_uses_shifted_targets_and_token_weighted_loss(tmp_path):
    sequences = [torch.tensor([[1, 0, 0]]), torch.tensor([[1, 2]])]
    save_candidate(FixedModel(2), sequences, "cpu", tmp_path)
    report = compare_quality(FixedModel(), sequences, "cpu", tmp_path, [[1, 2]], [[1, 3]])
    expected_loss = math.log(math.exp(2) + 3) - 4 / 3
    assert report["reference_perplexity"] == pytest.approx(4)
    assert report["candidate_nll"] == pytest.approx(expected_loss)
    assert report["logit_mean_absolute_error"] == pytest.approx(0.5)
    assert report["logit_max_absolute_error"] == 2
    assert report["generated_token_agreement"] == 0.5
    assert report["generated_sequences_exact_match"] == 0


def test_evaluation_reads_separate_documents_and_hashes_content(tmp_path):
    path = tmp_path / "held-out.txt"
    path.write_text("\na\nabcdef\nxyz\nignored\n")
    tokenizer = SimpleNamespace(encode=lambda line, **kwargs: list(range(len(line)))[:kwargs["max_length"]])
    sequences, dataset = prepare_evaluation(options(smoke=False, eval_text=path, eval_tokens=4), tokenizer, 16)
    assert [x.shape[1] for x in sequences] == [4, 3]
    assert dataset["predicted_tokens"] == 5
    assert dataset["synthetic"] is False
    assert len(dataset["source_sha256"]) == 64
    path.write_text("a\n")
    with pytest.raises(ValueError, match="at least two tokens"):
        prepare_evaluation(options(smoke=False, eval_text=path), tokenizer, 16)


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
def test_precision_cli_quality_and_comparison(tmp_path, dtype):
    path = tmp_path / f"{dtype}.json"
    subprocess.run([
        sys.executable, "-m", "miniinfer.benchmark", "--smoke", "--device", "cpu",
        "--dtype", dtype, "--check-quality", "--prompt-tokens", "8", "--new-tokens", "3",
        "--runs", "2", "--warmup", "1", "--eval-tokens", "8", "--eval-samples", "2",
        "--output", str(path),
    ], check=True, capture_output=True, text=True)
    report = json.loads(path.read_text())
    assert report["schema_version"] == 2
    assert report["quality"]["dataset"]["predicted_tokens"] == 14
    assert report["startup"]["first_generation_ms"] > 0
    assert math.isfinite(report["quality"]["candidate_perplexity"])
    assert 0 <= report["quality"]["generated_token_agreement"] <= 1
    if dtype == "fp32":
        assert report["quality"]["logit_max_absolute_error"] == 0
        assert report["quality"]["generated_token_agreement"] == 1
        candidate = deepcopy(report)
        candidate["summary"]["output_tokens_per_second"] *= 2
        assert compare_reports(report, candidate)["output_throughput_speedup"] == 2
        candidate["config"]["batch_size"] += 1
        with pytest.raises(ValueError, match="batch_size"):
            compare_reports(report, candidate)
        candidate = deepcopy(report)
        candidate["input_ids_sha256"] = "different"
        with pytest.raises(ValueError, match="input_ids_sha256"):
            compare_reports(report, candidate)


@pytest.mark.skipif(os.environ.get("MINIINFER_TEST_COMPILE") != "1",
                    reason="Set MINIINFER_TEST_COMPILE=1 to run the real Inductor integration test")
def test_real_inductor_generation_matches_fp32(tmp_path):
    path = tmp_path / "compiled.json"
    subprocess.run([
        sys.executable, "-m", "miniinfer.benchmark", "--smoke", "--device", "cpu",
        "--compile", "--check-quality", "--prompt-tokens", "8", "--new-tokens", "3",
        "--runs", "2", "--warmup", "1", "--eval-tokens", "8", "--eval-samples", "2",
        "--output", str(path),
    ], check=True, capture_output=True, text=True)
    report = json.loads(path.read_text())
    assert report["optimization"]["compile_backend"] == "inductor"
    assert report["quality"]["generated_token_agreement"] == 1
    assert report["quality"]["logit_max_absolute_error"] < 1e-4


@pytest.mark.parametrize("quantization", ["int8", "int4"])
def test_quantized_loader_keeps_model_on_one_device(monkeypatch, quantization):
    import miniinfer.optimization as optimization

    calls = []
    fake_model = SimpleNamespace(is_loaded_in_8bit=quantization == "int8",
                                 is_loaded_in_4bit=quantization == "int4")
    fake_model.eval = lambda: fake_model
    # No .to method: quantized models must not be recast after loading.
    monkeypatch.setattr(optimization, "BitsAndBytesConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(optimization.AutoModelForCausalLM, "from_pretrained",
                        lambda name, **kwargs: calls.append((name, kwargs)) or fake_model)
    args = options(smoke=False, quantization=quantization, dtype="fp16", model="test-model", revision="main")
    assert load_model(args, torch.device("cuda"), revision="fixed-commit") is fake_model
    _, kwargs = calls[0]
    assert kwargs["revision"] == "fixed-commit"
    assert kwargs["device_map"] == {"": "cuda"}
    assert kwargs["torch_dtype"] == torch.float16
    config = kwargs["quantization_config"]
    assert config["load_in_8bit"] == (quantization == "int8")
    assert config["load_in_4bit"] == (quantization == "int4")
    assert config["bnb_4bit_quant_type"] == "nf4"
    assert config["bnb_4bit_compute_dtype"] == torch.float16
