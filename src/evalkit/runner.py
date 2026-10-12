"""The evaluation runner: generate predictions, score them, summarize."""

from __future__ import annotations

from dataclasses import dataclass, field

from evalkit import metrics as metrics_module
from evalkit.datasets import EvalExample
from evalkit.models import BaseModel


@dataclass
class EvalResult:
    """The scored outcome for one example."""

    example_id: str
    prompt: str
    reference: str
    prediction: str
    latency_s: float
    scores: dict[str, float] = field(default_factory=dict)


def run_eval(
    examples: list[EvalExample],
    model: BaseModel,
    metric_names: list[str] | None = None,
    verbose: bool = False,
) -> list[EvalResult]:
    """Run every example through ``model`` and score with each metric."""
    metric_names = metric_names or ["exact_match", "token_f1"]
    scorers = {}
    for name in metric_names:
        if name not in metrics_module.METRICS:
            raise ValueError(
                f"unknown metric {name!r}; available: {metrics_module.available_metrics()}"
            )
        scorers[name] = metrics_module.METRICS[name]

    results: list[EvalResult] = []
    for i, example in enumerate(examples, start=1):
        response = model.generate(example.prompt)
        scores = {
            name: scorer(response.text, example.reference)
            for name, scorer in scorers.items()
        }
        results.append(
            EvalResult(
                example_id=example.id,
                prompt=example.prompt,
                reference=example.reference,
                prediction=response.text,
                latency_s=response.latency_s,
                scores=scores,
            )
        )
        if verbose:
            score_str = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
            print(f"[{i}/{len(examples)}] {example.id}: {score_str}")
    return results


def summarize(results: list[EvalResult]) -> dict:
    """Aggregate per-metric means and latency stats over all results."""
    if not results:
        return {"n": 0}
    metric_names = list(results[0].scores)
    summary: dict = {"n": len(results)}
    for name in metric_names:
        values = [r.scores[name] for r in results]
        summary[name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    latencies = [r.latency_s for r in results]
    summary["latency_s"] = {
        "mean": sum(latencies) / len(latencies),
        "max": max(latencies),
    }
    return summary
