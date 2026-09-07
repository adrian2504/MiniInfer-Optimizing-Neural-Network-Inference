"""Untimed quality checks on fixed text, against the eager FP32 model."""

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def prepare_evaluation(args, tokenizer, context_limit):
    if args.eval_samples < 1 or args.eval_tokens < 2:
        raise ValueError("eval_samples must be positive and eval_tokens must be at least 2")
    if context_limit and args.eval_tokens > context_limit:
        raise ValueError("eval_tokens exceeds the model context limit")
    if args.smoke:
        if args.eval_text:
            raise ValueError("Use a pretrained model to evaluate text; smoke quality uses synthetic tokens")
        generator = torch.Generator().manual_seed(args.seed + 1)
        sequences = [torch.randint(2, 128, (1, args.eval_tokens), generator=generator)
                     for _ in range(args.eval_samples)]
        source_hash = None
    else:
        if not args.eval_text:
            raise ValueError("--check-quality requires --eval-text with held-out text (one document per line)")
        raw = args.eval_text.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        sequences = []
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            ids = tokenizer.encode(line, add_special_tokens=False, truncation=True,
                                   max_length=args.eval_tokens)
            if len(ids) < 2:
                continue
            sequences.append(torch.tensor([ids], dtype=torch.long))
            if len(sequences) == args.eval_samples:
                break
        if not sequences:
            raise ValueError("Evaluation text must contain a document with at least two tokens")
    return sequences, {
        "source_sha256": source_hash,
        "input_ids_sha256": hashlib.sha256(json.dumps([x.tolist() for x in sequences]).encode()).hexdigest(),
        "documents": len(sequences),
        "predicted_tokens": sum(x.shape[1] - 1 for x in sequences),
        "max_tokens_per_document": args.eval_tokens,
        "synthetic": args.smoke,
    }


@torch.inference_mode()
def logits_for(model, ids, device):
    logits = model(input_ids=ids.to(device), use_cache=False).logits[:, :-1, :].float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits during quality evaluation")
    return logits


@torch.inference_mode()
def save_candidate(model, sequences, device, directory):
    # Keep one document's logits in RAM at a time; the FP32 model loads later.
    for index, ids in enumerate(sequences):
        torch.save(logits_for(model, ids, device), Path(directory) / f"{index}.pt")


@torch.inference_mode()
def compare_quality(reference, sequences, device, directory, reference_ids, candidate_ids):
    reference_nll = candidate_nll = absolute_error = 0.0
    max_error = 0.0
    elements = tokens = agreement = 0
    for index, ids in enumerate(sequences):
        expected = logits_for(reference, ids, device)
        actual = torch.load(Path(directory) / f"{index}.pt", weights_only=True, map_location="cpu")
        targets = ids[:, 1:].reshape(-1)
        for logits, is_reference in ((expected, True), (actual, False)):
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets, reduction="sum").item()
            if is_reference:
                reference_nll += nll
            else:
                candidate_nll += nll
        delta = (actual - expected).abs()
        absolute_error += delta.double().sum().item()
        max_error = max(max_error, delta.max().item())
        elements += delta.numel()
        tokens += targets.numel()
        agreement += (actual.argmax(-1) == expected.argmax(-1)).sum().item()
    reference_loss = reference_nll / tokens
    candidate_loss = candidate_nll / tokens
    def perplexity(loss):
        return math.exp(loss) if loss < 709 else None
    generated_expected = torch.tensor(reference_ids)
    generated_actual = torch.tensor(candidate_ids)
    return {
        "reference": "eager-fp32-no-cache",
        "reference_nll": reference_loss,
        "candidate_nll": candidate_loss,
        "nll_delta": candidate_loss - reference_loss,
        "reference_perplexity": perplexity(reference_loss),
        "candidate_perplexity": perplexity(candidate_loss),
        "logit_mean_absolute_error": absolute_error / elements,
        "logit_max_absolute_error": max_error,
        "teacher_forced_token_agreement": agreement / tokens,
        "generated_token_agreement": (generated_expected == generated_actual).float().mean().item(),
        "generated_sequences_exact_match": (generated_expected == generated_actual).all(dim=1).float().mean().item(),
        "reference_output_ids": reference_ids,
    }
