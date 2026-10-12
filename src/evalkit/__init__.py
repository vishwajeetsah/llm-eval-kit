"""EvalKit: a lightweight LLM evaluation harness for students and practitioners.

Run small, reproducible evaluations of language models: load a dataset of
prompt/reference pairs, generate predictions with any model client, score
them with standard metrics, and produce a readable report.
"""

from evalkit.datasets import EvalExample, load_jsonl, sample_dataset
from evalkit.metrics import contains_match, exact_match, token_f1
from evalkit.models import BaseModel, MockModel, ModelResponse, OpenAICompatibleModel
from evalkit.report import save_report, to_json, to_markdown
from evalkit.runner import EvalResult, run_eval, summarize

__version__ = "0.1.0"

__all__ = [
    "EvalExample",
    "EvalResult",
    "BaseModel",
    "MockModel",
    "ModelResponse",
    "OpenAICompatibleModel",
    "contains_match",
    "exact_match",
    "load_jsonl",
    "run_eval",
    "sample_dataset",
    "save_report",
    "summarize",
    "to_json",
    "to_markdown",
    "token_f1",
]
