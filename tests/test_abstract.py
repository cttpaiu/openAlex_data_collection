"""
Regression tests for abstract reconstruction.

OpenAlex stores abstracts as an inverted index: {word: [positions]}.
reconstruct_abstract() must rebuild the plain text correctly.
"""

from __future__ import annotations

import pytest

from openalex.utils import reconstruct_abstract


def test_simple_abstract():
    inverted = {"Hello": [0], "world": [1]}
    assert reconstruct_abstract(inverted) == "Hello world"


def test_multi_position_word():
    """A word that appears at multiple positions."""
    inverted = {"the": [0, 3], "cat": [1], "sat": [2], "mat": [4]}
    result = reconstruct_abstract(inverted)
    assert result == "the cat sat the mat"


def test_out_of_order_positions():
    """Words stored in arbitrary order must be reassembled by position."""
    inverted = {"B": [1], "A": [0], "C": [2]}
    assert reconstruct_abstract(inverted) == "A B C"


def test_none_returns_empty_string():
    """None input (missing abstract in API response) returns empty string."""
    assert reconstruct_abstract(None) == ""


def test_empty_dict_returns_empty_string():
    assert reconstruct_abstract({}) == ""


def test_single_word():
    assert reconstruct_abstract({"quantum": [0]}) == "quantum"
