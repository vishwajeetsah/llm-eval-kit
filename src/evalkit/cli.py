"""Command-line interface for EvalKit."""

from __future__ import annotations

import argparse
import os
import sys

from evalkit.datasets import load_jsonl, sample_dataset
from evalkit.metrics import available_metrics
from evalkit.models import MockModel, OpenAICompatibleModel
from evalkit.report import save_report
from evalkit.runner import run_eval, summarize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalkit",
        description="A lightweight LLM evaluation harness.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run an evaluation.")
    run_p.add_argument(
        "--dataset",
        default=None,
        help="Path to a JSONL dataset. Defaults to the built-in sample set.",
    )
    run_p.add_argument(
        "--model",
        default="mock",
        choices=["mock", "openai-compatible"],
        help="Which model client to use (default: mock).",
    )
    run_p.add_argument("--base-url", default="https://api.openai.com/v1")
    run_p.add_argument("--model-name", default="gpt-4o-mini")
    run_p.add_argument(
        "--api-key",
        default=None,
        help="API key (or set the EVALKIT_API_KEY env var).",
    )
    run_p.add_argument(
        "--metrics",
        default="exact_match,token_f1",
        help=f"Comma-separated metrics. Available: {', '.join(available_metrics())}",
    )
    run_p.add_argument("--output", default="eval_report.md")
    run_p.add_argument(
        "--format", default="markdown", choices=["markdown", "json"]
    )
    run_p.add_argument("--verbose", action="store_true")

    sub.add_parser("metrics", help="List available metrics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "metrics":
        print("\n".join(available_metrics()))
        return 0

    examples = load_jsonl(args.dataset) if args.dataset else sample_dataset()
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    if args.model == "mock":
        model = MockModel(mode="echo")
    else:
        api_key = args.api_key or os.environ.get("EVALKIT_API_KEY")
        model = OpenAICompatibleModel(
            model=args.model_name, api_key=api_key, base_url=args.base_url
        )

    print(f"Running eval: {len(examples)} examples, model={model.name}")
    results = run_eval(examples, model, metric_names=metric_names, verbose=args.verbose)
    summary = summarize(results)
    out = save_report(results, summary, args.output, fmt=args.format)
    print(f"Report written to {out}")
    for name, stats in summary.items():
        if name in ("n", "latency_s"):
            continue
        print(f"  {name}: mean={stats['mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
