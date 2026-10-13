"""Prompt template used everywhere (baseline, SFT, DPO, evaluation)."""

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# models sometimes start a new fake turn after answering; cut generation there
STOP_STRING = "\n### Instruction:"


def format_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())