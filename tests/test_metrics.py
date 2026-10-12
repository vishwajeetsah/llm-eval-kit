"""Unit tests for evalkit.metrics."""

from evalkit.metrics import (
    available_metrics,
    contains_match,
    exact_match,
    normalize,
    token_f1,
)


def test_normalize():
    assert normalize("  The Quick, Brown Fox! ") == "quick brown fox"
    assert normalize("An apple") == "apple"


def test_exact_match():
    assert exact_match("Paris", "paris") == 1.0
    assert exact_match("The Paris", "Paris") == 1.0
    assert exact_match("London", "Paris") == 0.0


def test_token_f1_perfect():
    assert token_f1("the cat sat", "cat sat") == 1.0


def test_token_f1_partial():
    # prediction has 2 tokens, 1 overlaps with the 2-token reference
    # precision=0.5, recall=0.5 -> f1=0.5
    assert token_f1("cat dog", "cat bird") == 0.5


def test_token_f1_no_overlap():
    assert token_f1("apple", "orange") == 0.0


def test_token_f1_empty():
    assert token_f1("", "") == 1.0
    assert token_f1("", "something") == 0.0


def test_contains_match():
    assert contains_match("The capital is Paris, France.", "paris") == 1.0
    assert contains_match("London", "Paris") == 0.0


def test_available_metrics():
    names = available_metrics()
    assert {"exact_match", "token_f1", "contains_match"} <= set(names)
