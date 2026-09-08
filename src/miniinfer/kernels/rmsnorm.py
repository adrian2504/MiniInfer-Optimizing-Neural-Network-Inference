"""RMSNorm over the last dimension; FP32 arithmetic, input-dtype output."""

import math

import torch

SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
MAX_WIDTH = 16384


def validate_inputs(x, weight, eps):
    if x.ndim < 1 or x.numel() == 0 or x.shape[-1] > MAX_WIDTH:
        raise ValueError(f"Input must be nonempty with a last dimension of 1–{MAX_WIDTH}")
    if weight.shape != (x.shape[-1],):
        raise ValueError("Weight must be a vector matching the last input dimension")
    if x.dtype not in SUPPORTED_DTYPES or weight.dtype != x.dtype:
        raise ValueError("Input and weight must share fp32, fp16, or bf16 dtype")
    if x.device != weight.device:
        raise ValueError("Input and weight must be on the same device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Input and weight must be contiguous")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise ValueError("These kernels are inference-only; use torch.inference_mode()")


def rmsnorm_math(x, weight, eps):
    # Accumulate squares in FP32 so FP16 values don't overflow when squared.
    values = x.float()
    inverse_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return (values * inverse_rms * weight.float()).to(x.dtype)


def rmsnorm(x, weight, eps=1e-6):
    validate_inputs(x, weight, eps)
    return rmsnorm_math(x, weight, eps)


def triton_rmsnorm(x, weight, eps=1e-6):
    validate_inputs(x, weight, eps)
    if x.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("The Triton kernel requires an NVIDIA CUDA device")
    try:
        from miniinfer.kernels.triton_rmsnorm import launch
    except ImportError as exc:
        raise ValueError("Install Triton on Linux with: pip install -e '.[kernels]'") from exc
    # Device guard also handles callers using a non-default CUDA device.
    with torch.cuda.device(x.device):
        return launch(x, weight, eps)
