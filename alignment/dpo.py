"""Core DPO math (manual Section 2.4 and 10), shared by both from-scratch DPO runs.

Per pair (x, chosen y_w, rejected y_l):

    delta_policy = log pi_theta(y_w|x) - log pi_theta(y_l|x)
    delta_ref    = log pi_ref(y_w|x)   - log pi_ref(y_l|x)
    loss         = -log sigmoid( beta * (delta_policy - delta_ref) )

Response log-probs come from alignment.sft, the same response-only scoring used for SFT.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alignment.sft import build_lm_batch, response_logprobs_from_batch


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length,
                    pad_to_max_length=True):
    """Two ordinary response-masked LM batches over the SAME prompts, kept in the same row order."""
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
    """Scalar batch-mean DPO loss.

    -logsigmoid(z) is used instead of -log(sigmoid(z)): for very negative z, sigmoid(z)
    underflows to 0 and log(0) = -inf, while logsigmoid computes the same value stably.
    """
    logits = dpo_logits(policy_chosen_logps, policy_rejected_logps,
                        reference_chosen_logps, reference_rejected_logps, beta)
    return -F.logsigmoid(logits).mean()


@torch.no_grad()
def dpo_metrics(policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps, beta):
    """Implicit rewards (manual 10.5), returned per pair so they can be averaged over a whole set.

    r_w = beta * (log pi_theta(y_w|x) - log pi_ref(y_w|x))
    r_l = beta * (log pi_theta(y_l|x) - log pi_ref(y_l|x))
    accuracy: r_w > r_l (strict, so identical policy/reference gives 0, not 0.5)
    margin:   r_w - r_l
    """
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    return {
        "chosen_rewards": chosen_rewards.detach(),
        "rejected_rewards": rejected_rewards.detach(),
        "correct": (chosen_rewards > rejected_rewards).float(),
        "margins": (chosen_rewards - rejected_rewards).detach(),
    }


def policy_and_reference_logps(policy_model, reference_model, batch):
    """The four per-pair log-probs. Gradients flow only through the policy.

    The reference may live on another GPU (and in fp16); its results are moved to the
    policy's device and cast to fp32 so the loss is computed in one place.
    """
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