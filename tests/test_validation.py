"""
Regression tests for keyword and topic validation rules.

Covers all 6 keyword validation rules in validator.py and the topic ID
format check. Each test is named after the rule it covers.
"""

from __future__ import annotations

import pytest

from openalex.validator import validate_keywords, validate_topic_format


# ---------------------------------------------------------------------------
# validate_keywords — rule: non-empty
# ---------------------------------------------------------------------------

def test_empty_query_is_invalid():
    errors = validate_keywords("")
    assert any("empty" in e.lower() for e in errors)


def test_whitespace_only_query_is_invalid():
    errors = validate_keywords("   ")
    assert any("empty" in e.lower() for e in errors)


def test_valid_simple_query():
    assert validate_keywords('("quantum computing" OR "qubit")') == []


# ---------------------------------------------------------------------------
# validate_keywords — rule: balanced parentheses
# ---------------------------------------------------------------------------

def test_unbalanced_open_paren():
    errors = validate_keywords('("quantum computing" OR "qubit"')
    assert any("parenthes" in e.lower() for e in errors)


def test_unbalanced_close_paren():
    errors = validate_keywords('"quantum computing" OR "qubit")')
    assert any("parenthes" in e.lower() for e in errors)


def test_balanced_nested_parens():
    assert validate_keywords('(("a" OR "b") AND ("c" OR "d"))') == []


# ---------------------------------------------------------------------------
# validate_keywords — rule: even double quotes
# ---------------------------------------------------------------------------

def test_odd_double_quotes():
    errors = validate_keywords('"quantum computing OR "qubit"')
    assert any("quote" in e.lower() for e in errors)


def test_even_double_quotes_valid():
    assert validate_keywords('"quantum computing" OR "qubit"') == []


# ---------------------------------------------------------------------------
# validate_keywords — rule: operators must be uppercase
# ---------------------------------------------------------------------------

def test_lowercase_or_is_invalid():
    errors = validate_keywords('"quantum computing" or "qubit"')
    assert any("uppercase" in e.lower() or "operator" in e.lower() for e in errors)


def test_lowercase_and_is_invalid():
    errors = validate_keywords('"quantum computing" and "qubit"')
    assert any("uppercase" in e.lower() or "operator" in e.lower() for e in errors)


def test_lowercase_not_is_invalid():
    errors = validate_keywords('"quantum computing" not "qubit"')
    assert any("uppercase" in e.lower() or "operator" in e.lower() for e in errors)


def test_uppercase_operators_valid():
    assert validate_keywords('"quantum computing" OR "qubit"') == []
    assert validate_keywords('"a" AND "b"') == []
    # AND NOT / OR NOT are valid Boolean patterns and must not be flagged
    assert validate_keywords('"a" AND NOT "b"') == [], (
        "AND NOT is valid Boolean syntax — validator must not flag it as adjacent operators"
    )
    assert validate_keywords('"a" OR NOT "b"') == []


# ---------------------------------------------------------------------------
# validate_keywords — rule: no adjacent operators
# ---------------------------------------------------------------------------

def test_adjacent_or_or_is_invalid():
    errors = validate_keywords('"a" OR OR "b"')
    assert any("adjacent" in e.lower() or "operator" in e.lower() for e in errors)


def test_adjacent_and_or_is_invalid():
    errors = validate_keywords('"a" AND OR "b"')
    assert any("adjacent" in e.lower() or "operator" in e.lower() for e in errors)


def test_not_not_is_invalid():
    """NOT NOT is redundant and flagged — unlike AND NOT / OR NOT which are valid."""
    errors = validate_keywords('"a" NOT NOT "b"')
    assert any("adjacent" in e.lower() or "operator" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# validate_keywords — rule: no empty parentheses
# ---------------------------------------------------------------------------

def test_empty_parens_is_invalid():
    errors = validate_keywords('"quantum computing" OR ()')
    assert any("empty" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# validate_topic_format
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tid", ["T10020", "T00001", "T99999", "T12345"])
def test_valid_topic_ids(tid):
    assert validate_topic_format(tid) is True


@pytest.mark.parametrize("tid", [
    "T1002",       # only 4 digits
    "T100200",     # 6 digits
    "t10020",      # lowercase t
    "10020",       # no T prefix
    '"T10020"',    # quoted (Python list paste)
    "T10020 ",     # trailing space
    " T10020",     # leading space
    "T1002O",      # letter O instead of zero
    "CORE_IDS",    # Python variable name (from notebook paste)
])
def test_invalid_topic_ids(tid):
    assert validate_topic_format(tid) is False, f"Expected {tid!r} to fail format check"
