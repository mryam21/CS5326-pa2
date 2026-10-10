"""Edge-case checks for alignment/dpo.py beyond the public smoke tests."""

from __future__ import annotations

import math

import torch

from alignment.dpo import (
    build_dpo_batch,
    compute_dpo_loss,
    dpo_loss_from_logps,
    dpo_metrics,
    policy_and_reference_logps,
)
from alignment.sft import IGNORE_INDEX, compute_response_logprobs
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_loss_matches_a_hand_computation():
    pc, pr = torch.tensor([-1.0, -4.0]), torch.tensor([-3.0, -2.0])
    rc, rr = torch.tensor([-2.0, -3.0]), torch.tensor([-2.0, -3.0])
    beta = 0.5
    # pair 1: beta*((-1+3) - 0) = 1.0 ; pair 2: beta*((-4+2) - 0) = -1.0
    expected = (-math.log(1 / (1 + math.exp(-1.0))) - math.log(1 / (1 + math.exp(1.0)))) / 2
    got = dpo_loss_from_logps(pc, pr, rc, rr, beta)
    assert abs(got.item() - expected) < 1e-6


def test_loss_stays_finite_for_extreme_logits():
    pc, pr = torch.tensor([-5000.0]), torch.tensor([0.0])
    zero = torch.zeros(1)
    loss = dpo_loss_from_logps(pc, pr, zero, zero, beta=1.0)
    assert torch.isfinite(loss)
    assert abs(loss.item() - 5000.0) < 1e-3  # -logsigmoid(-5000) ~= 5000
    naive = -torch.log(torch.sigmoid(torch.tensor(-5000.0)))
    assert torch.isinf(naive)  # why logsigmoid is needed


def test_identical_policy_and_reference_give_log2_zero_accuracy_zero_margin():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.6)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=0.6)
    batch = build_dpo_batch(tok, ["Q: ", "Ask: "], ["good", "yes"], ["bad", "nope"], max_length=16)
    logps = policy_and_reference_logps(policy, reference, batch)
    loss = dpo_loss_from_logps(*logps, beta=0.1)
    metrics = dpo_metrics(*logps, beta=0.1)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))
    assert metrics["correct"].sum().item() == 0  # strict r_w > r_l (manual 10.5)
    assert metrics["margins"].abs().max().item() == 0


def test_dpo_batch_masks_prompts_and_keeps_row_order():
    tok = ToyTokenizer()
    prompts, chosen, rejected = ["Task: ", "Q: "], ["a", "longer one"], ["bbbbbb", "c"]
    batch = build_dpo_batch(tok, prompts, chosen, rejected, max_length=20)
    for side, responses in (("chosen", chosen), ("rejected", rejected)):
        for row, (prompt, response) in enumerate(zip(prompts, responses)):
            n_prompt = len(tok.encode(prompt))
            labels = batch[side]["labels"][row]
            assert labels[:n_prompt].eq(IGNORE_INDEX).all()
            n_resp = len(tok.encode(response)) + 1
            assert labels[n_prompt:n_prompt + n_resp].tolist() == tok.encode(response) + [tok.eos_token_id]


def test_logps_equal_independent_scoring():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.8)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=-0.3)
    prompts, chosen, rejected = ["p: ", "q: "], ["yes", "ok"], ["no", "never"]
    batch = build_dpo_batch(tok, prompts, chosen, rejected, max_length=16)
    pc, pr, rc, rr = policy_and_reference_logps(policy, reference, batch)
    torch.testing.assert_close(pc, compute_response_logprobs(policy, tok, prompts, chosen, 16))
    torch.testing.assert_close(pr, compute_response_logprobs(policy, tok, prompts, rejected, 16))
    torch.testing.assert_close(rc, compute_response_logprobs(reference, tok, prompts, chosen, 16).detach())
    torch.testing.assert_close(rr, compute_response_logprobs(reference, tok, prompts, rejected, 16).detach())


def test_reference_gets_no_gradient_and_is_not_modified():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    before = reference.logit_bias.detach().clone()
    loss = compute_dpo_loss(policy, reference, tok, ["Pick: "], ["a"], ["b"], beta=0.1, max_length=12)
    loss.backward()
    assert reference.logit_bias.grad is None
    assert reference.logit_bias.requires_grad  # we must not silently change the caller's model
    torch.testing.assert_close(reference.logit_bias.detach(), before)


def test_optimization_raises_reward_accuracy_and_margin():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size)
    reference = ToyCausalLM(tok.vocab_size)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)
    prompts, chosen, rejected = ["Q: ", "Q: "], ["yes", "sure"], ["no", "nah"]
    batch = build_dpo_batch(tok, prompts, chosen, rejected, max_length=16)

    first = dpo_loss_from_logps(*policy_and_reference_logps(policy, reference, batch), beta=0.5).item()
    for _ in range(10):
        optimizer.zero_grad()
        dpo_loss_from_logps(*policy_and_reference_logps(policy, reference, batch), beta=0.5).backward()
        optimizer.step()
    logps = policy_and_reference_logps(policy, reference, batch)
    metrics = dpo_metrics(*logps, beta=0.5)
    assert dpo_loss_from_logps(*logps, beta=0.5).item() < first
    assert metrics["correct"].mean().item() == 1.0
    assert metrics["margins"].mean().item() > 0


def test_higher_beta_gives_sharper_loss_for_the_same_gap():
    pc, pr = torch.tensor([-1.0]), torch.tensor([-2.0])  # policy prefers chosen by 1 nat
    zero = torch.zeros(1)
    assert dpo_loss_from_logps(pc, pr, zero, zero, 1.0) < dpo_loss_from_logps(pc, pr, zero, zero, 0.1)