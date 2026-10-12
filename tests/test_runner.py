"""Unit tests for the eval runner, using the deterministic mock model."""

from evalkit.datasets import EvalExample, sample_dataset
from evalkit.models import MockModel
from evalkit.runner import run_eval, summarize


def _examples():
    return [
        EvalExample(id="a", prompt="What is 2+2?", reference="4"),
        EvalExample(id="b", prompt="Capital of France?", reference="Paris"),
    ]


def test_run_eval_oracle_gets_perfect_scores():
    examples = _examples()
    model = MockModel(
        mode="oracle",
        answers={ex.prompt: ex.reference for ex in examples},
    )
    results = run_eval(examples, model, metric_names=["exact_match", "token_f1"])
    assert len(results) == 2
    for r in results:
        assert r.scores["exact_match"] == 1.0
        assert r.scores["token_f1"] == 1.0
        assert r.latency_s >= 0


def test_run_eval_unknown_metric_raises():
    try:
        run_eval(_examples(), MockModel(), metric_names=["nope"])
    except ValueError as exc:
        assert "unknown metric" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_summarize():
    results = run_eval(sample_dataset(), MockModel(mode="echo"))
    summary = summarize(results)
    assert summary["n"] == len(sample_dataset())
    assert 0.0 <= summary["exact_match"]["mean"] <= 1.0
    assert summary["latency_s"]["mean"] >= 0


def test_summarize_empty():
    assert summarize([]) == {"n": 0}
