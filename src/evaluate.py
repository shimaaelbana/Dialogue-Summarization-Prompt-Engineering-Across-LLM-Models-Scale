"""ROUGE-based evaluation of generated summaries."""
from typing import Dict, List

from rouge_score import rouge_scorer


def score_summaries(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute average ROUGE-1 / ROUGE-2 / ROUGE-L F-measure across a batch."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure

    n = max(len(predictions), 1)
    return {key: value / n for key, value in totals.items()}
