"""Single prompt template shared by baseline generation, SFT, DPO, and evaluation."""

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# The base model tends to keep going and invent a new turn; we stop there.
STOP_STRING = "\n### Instruction:"


def format_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())