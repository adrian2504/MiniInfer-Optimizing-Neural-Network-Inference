"""Model loading and the switches used in Version 2 experiments."""

from importlib import metadata, util

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GPT2Config, GPT2LMHeadModel

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def validate_optimization(args, device):
    if args.compile and device.type == "mps":
        raise ValueError("This benchmark supports compilation on CPU and CUDA only")
    if args.compile and args.warmup < 1:
        raise ValueError("Compiled runs require --warmup of at least 1")
    if args.dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("This CUDA device does not support native BF16; use fp16 or fp32")
    if args.quantization != "none":
        if device.type != "cuda" or args.smoke:
            raise ValueError("Quantization requires a pretrained model on CUDA; smoke mode is not supported")
        if args.compile:
            raise ValueError("Compare quantization and compilation separately in Version 2")
        if args.dtype == "fp32":
            raise ValueError("Quantization requires --dtype fp16 or bf16 for unquantized layers and 4-bit compute")
        minimum = (7, 5) if args.quantization == "int8" else (6, 0)
        if torch.cuda.get_device_capability(device) < minimum:
            raise ValueError(f"{args.quantization} requires CUDA compute capability {minimum} or newer")
        if any(util.find_spec(name) is None for name in ("bitsandbytes", "accelerate")):
            raise ValueError("Install quantization dependencies with: pip install -e '.[quantization]'")


def load_model(args, device, *, reference=False, revision=None):
    # Reset before every load so smoke comparisons use identical random weights.
    torch.manual_seed(args.seed)
    dtype = torch.float32 if reference else DTYPES[args.dtype]
    quantized = not reference and args.quantization != "none"
    if args.smoke:
        config = GPT2Config(
            vocab_size=128, n_positions=256, n_embd=64, n_layer=2, n_head=2,
            bos_token_id=0, eos_token_id=1,
        )
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config)
    else:
        options = dict(revision=revision or args.revision, torch_dtype=dtype,
                       attn_implementation="eager", trust_remote_code=False)
        if quantized:
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=args.quantization == "int8",
                load_in_4bit=args.quantization == "int4",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=False,
            )
            # One device, no automatic CPU offload to change the benchmark.
            options["device_map"] = {"": str(device)}
        model = AutoModelForCausalLM.from_pretrained(args.model, **options)
        if quantized and not (getattr(model, "is_loaded_in_8bit", False)
                              or getattr(model, "is_loaded_in_4bit", False)):
            raise ValueError("The model was not loaded with the requested quantization")
    if not quantized:
        model = model.to(device=device, dtype=dtype)
    return model.eval()


def optimization_metadata(args, model):
    versions = {}
    for package in ("bitsandbytes", "accelerate", "triton"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "execution": "compiled" if args.compile else "eager",
        "dtype": args.dtype,
        "quantization": args.quantization,
        "quantization_config": model.config.quantization_config.to_dict()
        if getattr(model.config, "quantization_config", None) is not None else None,
        "compile_backend": "inductor" if args.compile else None,
        "compile_dynamic": True if args.compile else None,
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "optional_packages": versions,
    }


def compiler_snapshot():
    # Diagnostic API for our pinned PyTorch 2.8; revisit when upgrading PyTorch.
    from torch._dynamo.utils import counters

    return {
        "unique_graphs": counters["stats"]["unique_graphs"],
        "graph_breaks": sum(counters["graph_break"].values()),
    }
