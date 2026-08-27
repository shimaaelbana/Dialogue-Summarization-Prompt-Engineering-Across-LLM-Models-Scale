"""Prompt construction for zero-shot, one-shot, and few-shot dialogue summarization."""
from typing import List


def build_few_shot_prompt(
    dataset,
    split: str,
    example_indices: List[int],
    target_dialogue: str,
) -> str:
    """Build an in-context-learning prompt.

    `example_indices` supplies full dialogue/summary pairs shown to the model
    before the dialogue it must summarize. An empty list produces a
    zero-shot prompt; one index is one-shot; several indices is few-shot.
    """
    prompt = ""
    for idx in example_indices:
        ex_dialogue = dataset[split][idx]["dialogue"]
        ex_summary = dataset[split][idx]["summary"]
        prompt += f"Dialogue:\n\n{ex_dialogue}\n\nWhat was going on?\n{ex_summary}\n\n\n"

    prompt += f"Dialogue:\n\n{target_dialogue}\n\nWhat was going on?\n"
    return prompt
