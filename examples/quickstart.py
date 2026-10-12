"""Quickstart: run an end-to-end eval with the mock model (no API key needed)."""

from pathlib import Path

from evalkit.datasets import load_jsonl
from evalkit.models import MockModel
from evalkit.report import save_report
from evalkit.runner import run_eval, summarize

HERE = Path(__file__).resolve().parent


def main() -> None:
    examples = load_jsonl(HERE / "sample_dataset.jsonl")

    # Oracle mode answers every prompt with its reference -> perfect scores.
    # Swap in MockModel(mode="echo") to see realistic (imperfect) scores.
    model = MockModel(
        mode="oracle",
        answers={ex.prompt: ex.reference for ex in examples},
    )

    results = run_eval(examples, model, verbose=True)
    summary = summarize(results)

    out = save_report(results, summary, HERE / "quickstart_report.md")
    print(f"\nWrote {out}")
    print(f"exact_match mean: {summary['exact_match']['mean']:.3f}")


if __name__ == "__main__":
    main()
