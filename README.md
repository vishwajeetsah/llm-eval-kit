# llm-eval-kit

A lightweight **LLM evaluation harness** for students and practitioners. Load a dataset of prompts, run them through any model, score the outputs with standard metrics, and get a readable report — in under a minute, with zero API keys required.

> **Why evals?** If you can only learn one applied-AI skill in 2026, make it evaluation. Benchmarks and harnesses are how teams decide whether a model is actually getting better — for accuracy, safety, cost, and latency. This project teaches you that workflow hands-on.

## Quickstart (no API key needed)

```bash
pip install -e .
python examples/quickstart.py        # runs the mock model end-to-end
evalkit run --verbose                # CLI version, built-in sample dataset
evalkit metrics                      # list available metrics
```

## Using a real model

Any OpenAI-compatible endpoint works — OpenAI, Ollama, vLLM, LM Studio, Together…

```bash
export EVALKIT_API_KEY="sk-..."                       # for hosted APIs
evalkit run --model openai-compatible \
  --model-name gpt-4o-mini \
  --dataset examples/sample_dataset.jsonl \
  --output report.md
```

Local Ollama example (no key needed):

```bash
evalkit run --model openai-compatible \
  --base-url http://localhost:11434/v1 \
  --model-name llama3.1
```

## How it works

```
dataset (JSONL) ──▶ model client ──▶ metrics ──▶ Markdown / JSON report
```

- **`datasets.py`** — `EvalExample` dataclass + `load_jsonl()` loader. Bring your own benchmark as JSONL: `{"id": ..., "prompt": ..., "reference": ...}`.
- **`models.py`** — `BaseModel` interface, `MockModel` (deterministic, offline), `OpenAICompatibleModel` (stdlib-only HTTP client).
- **`metrics.py`** — `exact_match`, `token_f1`, `contains_match`, all SQuAD-style normalized. Register new ones in `METRICS`.
- **`runner.py`** — `run_eval()` generates + scores every example; `summarize()` aggregates means and latency.
- **`report.py`** — renders a Markdown summary table with per-example details, or JSON for pipelines.

## Project structure

```
src/evalkit/        # the library
tests/              # pytest suite (mock-based, runs offline)
examples/           # sample dataset + quickstart script
.github/workflows/  # CI: pytest on Python 3.10–3.12
```

## Run the tests

```bash
pip install pytest
pytest -q
```

## Ideas to extend it

- [ ] Add BLEU / ROUGE / BERTScore metrics
- [ ] Add an LLM-as-judge scorer (rubric-based grading)
- [ ] Parallelize the runner with `asyncio` and add rate limiting
- [ ] Track cost per run (tokens × price) in the summary
- [ ] Add a Streamlit dashboard for comparing runs side-by-side
- [ ] Dataset builders: MMLU / GSM8K subsets via Hugging Face `datasets`

## Contributing

Issues and PRs welcome — especially new metrics, model clients, and example datasets. Keep it dependency-light and student-friendly.

## License

MIT — see [LICENSE](LICENSE).
