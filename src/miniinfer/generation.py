"""Fixed-length greedy decoding, with an optional per-trial KV cache."""

import time

import torch
from transformers import DynamicCache


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def cache_summary(cache):
    if cache is None:
        return {"enabled": False, "sequence_length": 0, "tensor_bytes": 0}
    tensors = [tensor for layer in cache.layers for tensor in (layer.keys, layer.values)
               if tensor is not None]
    return {
        "enabled": True,
        "sequence_length": int(cache.get_seq_length()),
        "tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors),
    }


@torch.inference_mode()
def generate_trial(model, input_ids, new_tokens, kv_cache=False):
    if new_tokens < 1:
        raise ValueError("new_tokens must be positive")
    if input_ids.ndim != 2 or min(input_ids.shape) < 1 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a nonempty [batch, tokens] int64 tensor without padding")
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit and input_ids.shape[1] + new_tokens > limit:
        raise ValueError(f"Prompt plus output exceeds model context limit {limit}")
    device = input_ids.device
    sequence = input_ids
    # A new cache for every trial: no request can reuse an earlier request's state.
    cache = None
    synchronize(device)
    start = time.perf_counter()
    if kv_cache:
        cache = DynamicCache(config=model.config)
        positions = torch.arange(input_ids.shape[1], device=device)
        attention_mask = torch.ones_like(input_ids)
    first_token_time = None
    for step in range(new_tokens):
        if kv_cache:
            current = input_ids if step == 0 else sequence[:, -1:]
            output = model(input_ids=current, use_cache=True, past_key_values=cache,
                           attention_mask=attention_mask, cache_position=positions)
            if output.past_key_values is not cache:
                raise ValueError("Model must update and return the supplied DynamicCache")
        else:
            output = model(input_ids=sequence, use_cache=False)
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sequence = torch.cat((sequence, token), dim=1)
        del output
        if step == 0:
            synchronize(device)
            first_token_time = time.perf_counter()
        if kv_cache and step + 1 < new_tokens:
            attention_mask = torch.cat((attention_mask, attention_mask.new_ones((input_ids.shape[0], 1))), dim=1)
            positions = positions[-1:] + 1
    synchronize(device)
    end = time.perf_counter()
    return {
        "ttft_ms": (first_token_time - start) * 1000,
        "latency_ms": (end - start) * 1000,
        "output_ids": sequence[:, input_ids.shape[1]:].cpu().tolist(),
        # Count live K/V tensor elements after timing; allocator overhead is separate.
        "cache": cache_summary(cache),
    }
