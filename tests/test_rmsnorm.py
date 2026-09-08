import argparse
import json
import os
import subprocess
import sys

import pytest
import torch

from miniinfer.kernel_benchmark import check_output, parse_shape
from miniinfer.kernels import rmsnorm, triton_rmsnorm


def oracle(x, weight, eps=1e-6):
    values = x.double()
    return (values / torch.sqrt(torch.mean(values ** 2, dim=-1, keepdim=True) + eps)
            * weight.double()).to(x.dtype)


@pytest.mark.parametrize("shape", [(1,), (3, 7), (2, 3, 127), (5, 128), (2, 513), (1, 4097), (1, 16384)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reference_matches_fp64(shape, dtype):
    generator = torch.Generator().manual_seed(12)
    x = torch.randn(shape, generator=generator).to(dtype)
    weight = torch.randn(shape[-1], generator=generator).to(dtype)
    original = x.clone()
    output = rmsnorm(x, weight)
    check_output(output, oracle(x, weight))
    assert torch.equal(x, original)
    assert output.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("value", [0.0, 1e-4, 1000.0, -1000.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_zero_small_and_large_inputs(value, dtype):
    x = torch.full((3, 129), value, dtype=dtype)
    weight = torch.linspace(-2, 2, 129).to(dtype)
    check_output(rmsnorm(x, weight), oracle(x, weight))


def test_reference_matches_pytorch_rms_norm():
    torch.manual_seed(4)
    x, weight = torch.randn(5, 37), torch.randn(37)
    expected = torch.nn.functional.rms_norm(x, (37,), weight, eps=1e-6)
    torch.testing.assert_close(rmsnorm(x, weight), expected)


@pytest.mark.parametrize("x,weight,eps,message", [
    (torch.empty(0, 4), torch.ones(4), 1e-6, "nonempty"),
    (torch.ones(1, 16385), torch.ones(16385), 1e-6, "nonempty"),
    (torch.ones(2, 4), torch.ones(3), 1e-6, "Weight"),
    (torch.ones(2, 4), torch.ones(4).half(), 1e-6, "dtype"),
    (torch.ones(2, 4).long(), torch.ones(4).long(), 1e-6, "dtype"),
    (torch.ones(4, 2).T, torch.ones(4), 1e-6, "contiguous"),
    (torch.ones(2, 4), torch.ones(8)[::2], 1e-6, "contiguous"),
    (torch.ones(2, 4), torch.ones(4), 0, "eps"),
    (torch.ones(2, 4), torch.ones(4), float("nan"), "eps"),
    (torch.ones(2, 4, requires_grad=True), torch.ones(4), 1e-6, "inference-only"),
])
def test_invalid_inputs(x, weight, eps, message):
    with pytest.raises(ValueError, match=message):
        rmsnorm(x, weight, eps)


def test_triton_rejects_cpu_without_importing_triton():
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        triton_rmsnorm(torch.ones(2, 4), torch.ones(4))


def test_correctness_failure_is_not_reported_as_success():
    with pytest.raises(ValueError, match="correctness"):
        check_output(torch.zeros(2, 4), torch.ones(2, 4))
    with pytest.raises(ValueError, match="non-finite"):
        check_output(torch.full((2, 4), float("nan")), torch.ones(2, 4))


@pytest.mark.parametrize("value", ["0x128", "2x0", "2x16385", "abc", "2x"])
def test_bad_shape_argument(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape(value)


def test_kernel_cli_report_and_no_overwrite(tmp_path):
    path = tmp_path / "kernel.json"
    command = [sys.executable, "-m", "miniinfer.kernel_benchmark", "--device", "cpu",
               "--implementations", "eager", "--shapes", "2x7", "2x3x128", "--dtypes", "fp32", "bf16",
               "--runs", "2", "--iterations", "3", "--warmup", "1", "--output", str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(path.read_text())
    assert report["scope"] == "standalone-operation"
    assert report["gpu_performance"] is False
    assert len(report["cases"]) == 4
    for case in report["cases"]:
        result = case["implementations"]["eager"]
        assert result["correctness"]["passed"]
        assert len(result["samples_ms"]) == 2
        assert result["median_ms"] > 0
        assert result["speedup_vs_eager"] == 1
    original = path.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert path.read_bytes() == original


@pytest.mark.skipif(os.environ.get("MINIINFER_TEST_COMPILE") != "1", reason="Opt-in real Inductor test")
def test_compiled_kernel_sweep(tmp_path):
    path = tmp_path / "compiled-kernel.json"
    subprocess.run([sys.executable, "-m", "miniinfer.kernel_benchmark", "--device", "cpu",
                    "--implementations", "eager", "compiled", "--shapes", "3x127", "2x256",
                    "--dtypes", "fp32", "fp16", "bf16", "--runs", "2", "--iterations", "3",
                    "--warmup", "1", "--output", str(path)], check=True, capture_output=True, text=True)
    report = json.loads(path.read_text())
    assert len(report["cases"]) == 6
    for case in report["cases"]:
        result = case["implementations"]["compiled"]
        assert result["correctness"]["passed"]
        assert result["compiler_diagnostics"]["unique_graphs"] > 0
        assert result["compiler_diagnostics"]["graph_breaks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                    reason="Requires NVIDIA CUDA and Triton")
@pytest.mark.parametrize("shape", [(1,), (3, 7), (2, 3, 127), (5, 128), (3, 513), (1, 4097), (2, 16384)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_triton_against_reference_on_cuda(shape, dtype):
    pytest.importorskip("triton")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("Native BF16 unavailable")
    torch.manual_seed(9)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    for scale in (0.0, 1e-4, 1.0, 1000.0):
        x = torch.randn(shape, device="cuda", dtype=dtype) * scale
        output = triton_rmsnorm(x, weight)
        torch.cuda.synchronize()
        check_output(output, oracle(x, weight))
        check_output(output, rmsnorm(x, weight))
