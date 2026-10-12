"""Smoke test for report generation."""

from evalkit.datasets import sample_dataset
from evalkit.models import MockModel
from evalkit.report import to_json, to_markdown
from evalkit.runner import run_eval, summarize


def test_markdown_report_contains_sections():
    results = run_eval(sample_dataset(), MockModel(mode="echo"))
    summary = summarize(results)
    md = to_markdown(results, summary)
    assert "# Eval Report" in md
    assert "## Summary" in md
    assert "exact_match" in md
    assert "## Per-example results" in md


def test_json_report_round_trips():
    import json

    results = run_eval(sample_dataset(), MockModel(mode="echo"))
    summary = summarize(results)
    payload = json.loads(to_json(results, summary))
    assert payload["summary"]["n"] == len(sample_dataset())
    assert len(payload["results"]) == len(sample_dataset())
