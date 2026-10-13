"""SFT batching, response log-probs and loss. DPO reuses the same log-prob code."""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def _encode(tokenizer, text):
    return list(tokenizer.encode(text, add_special_tokens=False))


def _pad_id(tokenizer):
    # value doesn't matter, padded positions are masked out
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    return 0 if pad_id is None else pad_id


def build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=True):
    """Right-padded batch where only response tokens (+ EOS) have labels.

    Prompt and response are tokenized separately so the response boundary is exact.
    EOS is added before truncating to max_length, so a cut-off response has no EOS.
    The attention mask comes from the real lengths (Qwen's pad id equals its EOS id).
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


def response_logprobs_from_batch(model, batch):
    """Per-row sum of response-token log-probs, and the number of response tokens."""
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # position t predicts token t+1; upcast to fp32 for a stable logsumexp
    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)

    # -100 can't be used as an index; masked positions are zeroed below anyway
    safe_targets = targets.masked_fill(~mask, 0)

    # log p = logit - logsumexp, avoids materializing a full log_softmax tensor
    target_logits = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = target_logits - torch.logsumexp(logits, dim=-1)

    token_logps = token_logps * mask
    return token_logps.sum(dim=-1), mask.sum(dim=-1)


def sft_loss_from_batch(model, batch):
    """Mean negative log-likelihood over all response tokens in the batch (token-mean)."""
    summed_logps, token_counts = response_logprobs_from_batch(model, batch)
    return -summed_logps.sum() / token_counts.sum().clamp_min(1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    summed_logps, _ = response_logprobs_from_batch(model, batch)
    return summed_logps


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    return sft_loss_from_batch(model, batch)