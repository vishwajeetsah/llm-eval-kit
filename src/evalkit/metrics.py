"""Evaluation metrics.

All metrics take ``(prediction, reference)`` strings and return a float in
``[0, 1]``. They are deliberately simple and transparent — good defaults
for learning how evaluation works before reaching for heavier libraries.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalized strings are identical, else 0.0."""
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference (SQuAD-style)."""
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_match(prediction: str, reference: str) -> float:
    """1.0 if the normalized reference appears inside the prediction."""
    return 1.0 if normalize(reference) in normalize(prediction) else 0.0


#: Registry of metric name -> function, used by the runner and CLI.
METRICS: dict[str, callable] = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "contains_match": contains_match,
}


def available_metrics() -> list[str]:
    """Return the names of all registered metrics."""
    return sorted(METRICS)
