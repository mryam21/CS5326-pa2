"""Extra tests for alignment/sft.py: truncation, unusual tokenizers, padding and row order."""

from __future__ import annotations

import math

import torch
from torch import nn

from alignment.sft import IGNORE_INDEX, build_lm_batch, compute_response_logprobs, compute_sft_loss
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


class PadIsEosTokenizer(ToyTokenizer):
    """Like Qwen: the pad id and the EOS id are the same token."""
    pad_token_id = 1


def lengths(tok, prompt, response):
    return len(tok.encode(prompt)), len(tok.encode(response))


def test_padding_goes_to_max_length_on_the_right():
    tok = ToyTokenizer()
    batch = build_lm_batch(tok, ["ab", "a"], ["c", "cd"], max_length=10)
    assert batch["input_ids"].shape == (2, 10)
    # row 0: a b c <eos> then 6 pads
    assert batch["attention_mask"][0].tolist() == [1] * 4 + [0] * 6
    assert batch["labels"][0, 4:].eq(IGNORE_INDEX).all()


def test_truncation_cuts_response_and_its_eos():
    tok = ToyTokenizer()
    batch = build_lm_batch(tok, ["abc"], ["defgh"], max_length=5)
    assert batch["input_ids"][0].tolist() == tok.encode("abcde")
    assert batch["labels"][0].tolist() == [IGNORE_INDEX] * 3 + tok.encode("de")
    assert tok.eos_token_id not in batch["labels"][0].tolist()


def test_eos_survives_when_it_exactly_fits():
    tok = ToyTokenizer()
    batch = build_lm_batch(tok, ["ab"], ["cd"], max_length=5)  # 2 + 2 + eos = 5
    assert batch["labels"][0, -1].item() == tok.eos_token_id


def test_prompt_longer_than_max_length_gives_no_response_tokens():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    batch = build_lm_batch(tok, ["abcdefgh"], ["xyz"], max_length=4)
    assert batch["labels"].eq(IGNORE_INDEX).all()
    logps = compute_response_logprobs(model, tok, ["abcdefgh"], ["xyz"], max_length=4)
    assert logps.tolist() == [0.0]
    loss = compute_sft_loss(model, tok, ["abcdefgh"], ["xyz"], max_length=4)
    assert torch.isfinite(loss) and loss.item() == 0.0


def test_pad_equal_to_eos_keeps_the_eos_label():
    tok = PadIsEosTokenizer()
    batch = build_lm_batch(tok, ["ab"], ["c"], max_length=8)
    row = batch["labels"][0].tolist()
    assert row[:4] == [IGNORE_INDEX, IGNORE_INDEX, tok.encode("c")[0], tok.eos_token_id]
    assert batch["attention_mask"][0].tolist() == [1] * 4 + [0] * 4


def test_missing_pad_token_falls_back_without_crashing():
    tok = NoPadTokenizer()
    batch = build_lm_batch(tok, ["a", "abc"], ["b", "d"], max_length=6)
    assert batch["input_ids"][0, 3:].eq(tok.eos_token_id).all()
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0, 0]


def test_missing_eos_token_appends_nothing():
    tok = NoEosTokenizer()
    batch = build_lm_batch(tok, ["ab"], ["cd"], max_length=6)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0]
    assert batch["labels"][0].tolist() == [IGNORE_INDEX] * 2 + tok.encode("cd") + [IGNORE_INDEX] * 2


def test_row_order_is_preserved():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.9)
    prompts, responses = ["p: ", "q: ", "r: "], ["aaaa", "z", "mm"]
    together = compute_response_logprobs(model, tok, prompts, responses, max_length=12)
    one_by_one = torch.stack([
        compute_response_logprobs(model, tok, [p], [r], max_length=12)[0]
        for p, r in zip(prompts, responses)
    ])
    torch.testing.assert_close(together, one_by_one)


def test_logprob_matches_a_hand_computation():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.7)
    # The toy model predicts the same distribution at every position.
    log_probs = torch.log_softmax(model.logit_bias.detach(), dim=-1)
    response = "ab"
    expected = sum(log_probs[i] for i in tok.encode(response)) + log_probs[tok.eos_token_id]

    got = compute_response_logprobs(model, tok, ["Q: "], [response], max_length=20)[0]
    torch.testing.assert_close(got, expected)

    loss = compute_sft_loss(model, tok, ["Q: "], [response], max_length=20)
    torch.testing.assert_close(loss, -expected / 3)  # 'a', 'b', <eos>


def test_padding_amount_does_not_change_scores():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    short = compute_response_logprobs(model, tok, ["x"], ["yz"], max_length=5)
    long = compute_response_logprobs(model, tok, ["x"], ["yz"], max_length=50)
    torch.testing.assert_close(short, long)


def test_a_few_sgd_steps_lower_the_loss():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    prompts, responses = ["Q: ", "Q: "], ["yes", "yes"]
    first = compute_sft_loss(model, tok, prompts, responses, max_length=16).item()
    for _ in range(5):
        optimizer.zero_grad()
        compute_sft_loss(model, tok, prompts, responses, max_length=16).backward()
        optimizer.step()
    last = compute_sft_loss(model, tok, prompts, responses, max_length=16).item()
    assert last < first and math.isfinite(last)