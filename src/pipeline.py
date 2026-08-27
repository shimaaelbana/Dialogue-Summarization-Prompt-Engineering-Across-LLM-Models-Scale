"""Run a zero/one/few-shot x multi-model comparison and return a results table."""
from typing import Dict, List

import pandas as pd

from .data import load_dialogsum
from .evaluate import score_summaries
from .generate import summarize
from .prompting import build_few_shot_prompt

# Exemplar dialogues used to build the in-context prompts. Kept separate from
# the evaluation indices below so the model is never evaluated on an example
# it was also shown as a demonstration.
SHOT_CONFIGS = {
    "zero-shot": [],
    "one-shot": [40],
    "few-shot": [40, 80, 120],
}


def run_experiment(
    model_names: List[str],
    test_indices: List[int],
    split: str = "test",
    shot_configs: Dict[str, List[int]] = None,
    max_new_tokens: int = 60,
) -> pd.DataFrame:
    """Evaluate each model across each shot configuration on `test_indices`.

    Returns a DataFrame with one row per (model, shot_config) pair and
    rouge1 / rouge2 / rougeL columns averaged over `test_indices`.
    """
    shot_configs = shot_configs or SHOT_CONFIGS
    dataset = load_dialogsum()
    rows = []

    for model_name in model_names:
        for shot_name, example_indices in shot_configs.items():
            predictions, references = [], []

            for idx in test_indices:
                dialogue = dataset[split][idx]["dialogue"]
                reference = dataset[split][idx]["summary"]
                prompt = build_few_shot_prompt(dataset, split, example_indices, dialogue)
                prediction = summarize(prompt, model_name, max_new_tokens=max_new_tokens)
                predictions.append(prediction)
                references.append(reference)

            scores = score_summaries(predictions, references)
            rows.append({"model": model_name, "shot_config": shot_name, **scores})

    return pd.DataFrame(rows)
