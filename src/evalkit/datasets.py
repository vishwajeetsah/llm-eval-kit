"""Dataset loading utilities.

Datasets are lists of :class:`EvalExample` — a prompt plus the reference
answer you score the model against. The canonical on-disk format is JSONL,
one example per line::

    {"id": "ex-1", "prompt": "What is 2 + 2?", "reference": "4"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalExample:
    """A single evaluation example."""

    id: str
    prompt: str
    reference: str
    metadata: dict = field(default_factory=dict)


def load_jsonl(path: str | Path) -> list[EvalExample]:
    """Load evaluation examples from a JSONL file.

    Each line must be a JSON object with ``id``, ``prompt`` and
    ``reference`` keys. Extra keys are kept in ``metadata``.
    """
    examples: list[EvalExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            for key in ("id", "prompt", "reference"):
                if key not in obj:
                    raise ValueError(f"{path}:{lineno}: missing required key {key!r}")
            metadata = {k: v for k, v in obj.items() if k not in ("id", "prompt", "reference")}
            examples.append(
                EvalExample(
                    id=str(obj["id"]),
                    prompt=str(obj["prompt"]),
                    reference=str(obj["reference"]),
                    metadata=metadata,
                )
            )
    if not examples:
        raise ValueError(f"{path}: no examples found")
    return examples


def sample_dataset() -> list[EvalExample]:
    """Return a tiny built-in dataset for smoke-testing the harness."""
    return [
        EvalExample(id="math-1", prompt="What is 7 * 8?", reference="56"),
        EvalExample(id="math-2", prompt="What is 15 - 7?", reference="8"),
        EvalExample(
            id="geo-1",
            prompt="What is the capital of France?",
            reference="Paris",
        ),
        EvalExample(
            id="sci-1",
            prompt="What planet is known as the Red Planet?",
            reference="Mars",
        ),
        EvalExample(
            id="lit-1",
            prompt="Who wrote Romeo and Juliet?",
            reference="William Shakespeare",
        ),
    ]
