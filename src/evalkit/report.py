"""Report generation: human-readable Markdown and machine-readable JSON."""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.runner import EvalResult


def to_markdown(results: list[EvalResult], summary: dict, title: str = "Eval Report") -> str:
    """Render results and summary as a Markdown document."""
    lines = [f"# {title}", ""]
    lines.append(f"**Examples evaluated:** {summary.get('n', 0)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Mean | Min | Max |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, stats in summary.items():
        if name in ("n", "latency_s"):
            continue
        lines.append(
            f"| {name} | {stats['mean']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |"
        )
    if "latency_s" in summary:
        lat = summary["latency_s"]
        lines.append(f"| latency_s | {lat['mean']:.3f} | — | {lat['max']:.3f} |")
    lines.append("")
    lines.append("## Per-example results")
    lines.append("")
    for r in results:
        score_str = ", ".join(f"{k}={v:.2f}" for k, v in r.scores.items())
        lines.append(f"### {r.example_id} ({score_str})")
        lines.append("")
        lines.append(f"- **Prompt:** {r.prompt}")
        lines.append(f"- **Reference:** {r.reference}")
        lines.append(f"- **Prediction:** {r.prediction}")
        lines.append("")
    return "\n".join(lines)


def to_json(results: list[EvalResult], summary: dict) -> str:
    """Serialize results and summary as JSON."""
    payload = {
        "summary": summary,
        "results": [
            {
                "example_id": r.example_id,
                "prompt": r.prompt,
                "reference": r.reference,
                "prediction": r.prediction,
                "latency_s": r.latency_s,
                "scores": r.scores,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2)


def save_report(
    results: list[EvalResult],
    summary: dict,
    path: str | Path,
    fmt: str = "markdown",
) -> Path:
    """Write the report to ``path`` in ``markdown`` or ``json`` format."""
    path = Path(path)
    if fmt == "markdown":
        path.write_text(to_markdown(results, summary), encoding="utf-8")
    elif fmt == "json":
        path.write_text(to_json(results, summary), encoding="utf-8")
    else:
        raise ValueError(f"unknown format {fmt!r}; use 'markdown' or 'json'")
    return path
