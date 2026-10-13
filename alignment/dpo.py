"""
DPO loss and metrics, used by both from-scratch DPO runs.

loss = -log sigmoid(beta * ((log pi(y_w) - log pi(y_l)) - (log ref(y_w) - log ref(y_l))))
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alignment.sft import build_lm_batch, response_logprobs_from_batch


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length,
                    pad_to_max_length=True):
    """Chosen and rejected LM batches for the same prompts, in the same row order."""
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses and rejected_responses must have the same length")
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen_responses, max_length, pad_to_max_length),
        "rejected": build_lm_batch(tokenizer, prompts, rejected_responses, max_length, pad_to_max_length),
    }


def dpo_logits(policy_chosen_logps, policy_rejected_logps,
               reference_chosen_logps, reference_rejected_logps, beta):
    """beta * (delta_policy - delta_ref), one value per pair."""
    delta_policy = policy_chosen_logps - policy_rejected_logps
    delta_ref = reference_chosen_logps - reference_rejected_logps
    return beta * (delta_policy - delta_ref)


def dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                        reference_chosen_logps, reference_rejected_logps, beta):
    """Mean DPO loss over the batch. logsigmoid stays finite where log(sigmoid(z)) underflows."""
    logits = dpo_logits(policy_chosen_logps, policy_rejected_logps,
                        reference_chosen_logps, reference_rejected_logps, beta)
    return -F.logsigmoid(logits).mean()


@torch.no_grad()
def dpo_metrics(policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps, beta):
    """Per-pair implicit rewards r = beta * (log pi - log ref), accuracy (strict r_w > r_l) and margin."""
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    return {
        "chosen_rewards": chosen_rewards.detach(),
        "rejected_rewards": rejected_rewards.detach(),
        "correct": (chosen_rewards > rejected_rewards).float(),
        "margins": (chosen_rewards - rejected_rewards).detach(),
    }


def policy_and_reference_logps(policy_model, reference_model, batch):
    """Policy and reference log-probs for chosen/rejected. No gradients through the reference,
    whose outputs are moved to the policy's device (it may sit on another GPU)."""
    policy_chosen, _ = response_logprobs_from_batch(policy_model, batch["chosen"])
    policy_rejected, _ = response_logprobs_from_batch(policy_model, batch["rejected"])

    with torch.no_grad():
        reference_chosen, _ = response_logprobs_from_batch(reference_model, batch["chosen"])
        reference_rejected, _ = response_logprobs_from_batch(reference_model, batch["rejected"])

    device = policy_chosen.device
    return (policy_chosen, policy_rejected,
            reference_chosen.to(device, torch.float32), reference_rejected.to(device, torch.float32))


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    logps = policy_and_reference_logps(policy_model, reference_model, batch)
    return dpo_loss_from_logps(*logps, beta)