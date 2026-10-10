"""Core SFT math: batching with response-only labels, response log-probs, and the SFT loss.

These functions are shared by SFT training, DPO (which scores chosen/rejected responses
with the same log-prob code), and submission_adapter.py.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100  # label value that the loss skips


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def _encode(tokenizer, text):
    """Plain token ids for `text`, with no BOS/EOS added by the tokenizer."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def _pad_id(tokenizer):
    """Padding id to fill empty slots. Its value never matters: padded slots are masked."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    return 0 if pad_id is None else pad_id


def build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=True):
    """Tokenize prompt+response pairs into one right-padded LM batch.

    Per row:
        input_ids = prompt_ids + response_ids + [eos]      (then cut to max_length)
        labels    = [-100] * len(prompt_ids) + response_ids + [eos]   (same cut)

    The prompt and response are tokenized SEPARATELY and then joined, so we know
    exactly where the response starts. Tokenizing the joined string could merge
    tokens across the boundary and shift the mask.

    The EOS is appended BEFORE truncation (manual 4.1), so a response that is cut
    at max_length loses its EOS.

    Padding: labels are -100 and attention_mask is 0 on padded slots. The mask is
    built from the real lengths, never by comparing ids to pad_token_id, because
    for Qwen the pad token and the EOS token are the same id.
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = _pad_id(tokenizer)

    rows = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = _encode(tokenizer, prompt)
        response_ids = _encode(tokenizer, response)
        if eos_id is not None:
            response_ids = response_ids + [eos_id]

        input_ids = (prompt_ids + response_ids)[:max_length]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
        rows.append((input_ids, labels))

    if pad_to_max_length:
        width = max_length
    else:
        width = max(1, max(len(ids) for ids, _ in rows))

    batch_ids, batch_mask, batch_labels = [], [], []
    for input_ids, labels in rows:
        n_pad = width - len(input_ids)
        batch_ids.append(input_ids + [pad_id] * n_pad)
        batch_mask.append([1] * len(input_ids) + [0] * n_pad)
        batch_labels.append(labels + [IGNORE_INDEX] * n_pad)

    return {
        "input_ids": torch.tensor(batch_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Log-probabilities and loss
# ---------------------------------------------------------------------------

def response_logprobs_from_batch(model, batch):
    """Summed response-token log-probs per row, plus the response-token count per row.

    A causal LM's logits at position t predict the token at position t+1, so we
    compare logits[:, :-1] with labels[:, 1:] (the "shift").
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()        # fp32 for a stable softmax, even under fp16 autocast
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)

    # gather() cannot index with -100, so point masked slots at token 0;
    # their values are zeroed out by the mask afterwards.
    safe_targets = targets.masked_fill(~mask, 0)

    # log p(target) = logit(target) - logsumexp(all logits)
    # Same as log_softmax + gather, without building a full [B, T, vocab] log-prob tensor.
    target_logits = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = target_logits - torch.logsumexp(logits, dim=-1)

    token_logps = token_logps * mask
    return token_logps.sum(dim=-1), mask.sum(dim=-1)


def sft_loss_from_batch(model, batch):
    """Response-only SFT loss, averaged over all response tokens in the batch.

    Reduction choice (manual 4.1 allows two; we use this one everywhere):
        loss = -(sum of response-token log-probs) / (number of response tokens)
    """
    summed_logps, token_counts = response_logprobs_from_batch(model, batch)
    return -summed_logps.sum() / token_counts.sum().clamp_min(1)


# ---------------------------------------------------------------------------
# Convenience wrappers taking raw strings (used by submission_adapter.py)
# ---------------------------------------------------------------------------

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    summed_logps, _ = response_logprobs_from_batch(model, batch)
    return summed_logps


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    return sft_loss_from_batch(model, batch)