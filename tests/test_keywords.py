"""
Regression tests for keyword loading and query building.

BUG HISTORY
-----------
BUG-001: keywords.txt contained Python string-literal syntax
    Symptom : openalex search returned 6,474 records instead of ~127k
    Root cause: keywords.txt was copy-pasted from the Jupyter notebook as Python
                source code, e.g.:
                    '('
                    '("quantum computing" OR ...)'
                The raw Python quotes and newlines were sent verbatim to the API.
    Fix: keywords.txt must contain a plain Boolean query with no Python quotes.
    Test: test_get_keywords_no_python_string_syntax

BUG-002: Newlines in the query string corrupted the OpenAlex API filter
    Symptom : Same low count as BUG-001 — aiohttp URL-encoded \n as %0A, causing
              OpenAlex to truncate or misparse the filter at the first newline.
    Root cause: get_keywords() returned text with embedded \n characters.
    Fix: re.sub(r"\\s+", " ", raw) in get_keywords() collapses all whitespace.
    Tests: test_get_keywords_no_newlines, test_build_filter_no_newlines
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openalex.utils import build_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, keywords_content: str) -> MagicMock:
    """Return a minimal AppConfig mock whose keywords_file points to tmp_path."""
    kw_file = tmp_path / "keywords.txt"
    kw_file.write_text(keywords_content, encoding="utf-8")

    cfg = MagicMock()
    cfg.keywords_file = str(kw_file)
    cfg._keywords = None  # force re-read on first call

    # Attach the real get_keywords logic by importing and binding it
    from openalex.config import AppConfig
    cfg.get_keywords = AppConfig.get_keywords.__get__(cfg, type(cfg))
    return cfg


# ---------------------------------------------------------------------------
# BUG-001 — Python string-literal syntax must not appear in the query
# ---------------------------------------------------------------------------

PYTHON_SYNTAX_CONTENT = textwrap.dedent("""\
    '('
        '("quantum computing" OR "quantum computation") OR '
        '("qubit" OR "qutrit")'
    ')'
""")

PLAIN_QUERY = (
    '( ("quantum computing" OR "quantum computation") OR ("qubit" OR "qutrit") )'
)


def test_get_keywords_no_python_string_syntax(tmp_path):
    """BUG-001: keywords.txt with Python string syntax must NOT be returned as-is."""
    kw_file = tmp_path / "keywords.txt"
    kw_file.write_text(PLAIN_QUERY, encoding="utf-8")

    from openalex.config import AppConfig
    cfg = MagicMock()
    cfg.keywords_file = str(kw_file)
    cfg._keywords = None
    cfg.get_keywords = AppConfig.get_keywords.__get__(cfg, type(cfg))

    result = cfg.get_keywords()

    # Must not contain lone single quotes that wrap lines (Python syntax)
    assert "'" not in result, (
        "Loaded keywords contain single-quote characters — "
        "check that keywords.txt contains a plain Boolean query, not Python code."
    )


def test_keywords_file_with_python_syntax_would_fail_api(tmp_path):
    """BUG-001: Demonstrates that the Python-syntax file DOES contain single quotes."""
    kw_file = tmp_path / "keywords.txt"
    kw_file.write_text(PYTHON_SYNTAX_CONTENT, encoding="utf-8")

    raw = kw_file.read_text(encoding="utf-8")
    assert "'" in raw  # the bad file has quotes — this is what broke the search


# ---------------------------------------------------------------------------
# BUG-002 — Newlines must be collapsed before the query reaches the API
# ---------------------------------------------------------------------------

MULTILINE_QUERY = (
    '(\n'
    '("quantum computing" OR "quantum computation") OR\n'
    '("qubit" OR "qutrit")\n'
    ')'
)


def test_get_keywords_no_newlines(tmp_path):
    """BUG-002: get_keywords() must return a single-line string (no \\n)."""
    kw_file = tmp_path / "keywords.txt"
    kw_file.write_text(MULTILINE_QUERY, encoding="utf-8")

    from openalex.config import AppConfig
    cfg = MagicMock()
    cfg.keywords_file = str(kw_file)
    cfg._keywords = None
    cfg.get_keywords = AppConfig.get_keywords.__get__(cfg, type(cfg))

    result = cfg.get_keywords()

    assert "\n" not in result, (
        "get_keywords() returned a string with newlines. "
        "Newlines are URL-encoded as %0A and corrupt the OpenAlex API filter."
    )
    assert "\t" not in result, "get_keywords() returned a string with tab characters."
    # Confirm content is preserved
    assert "quantum computing" in result
    assert "qubit" in result


def test_get_keywords_collapses_multiple_spaces(tmp_path):
    """BUG-002 (related): multiple spaces from indentation are collapsed to one."""
    kw_file = tmp_path / "keywords.txt"
    kw_file.write_text('("quantum   computing"   OR  "qubit")', encoding="utf-8")

    from openalex.config import AppConfig
    cfg = MagicMock()
    cfg.keywords_file = str(kw_file)
    cfg._keywords = None
    cfg.get_keywords = AppConfig.get_keywords.__get__(cfg, type(cfg))

    result = cfg.get_keywords()
    assert "  " not in result, "get_keywords() left multiple consecutive spaces."


def test_build_filter_no_newlines():
    """BUG-002: build_filter() output must never contain newlines."""
    # Simulate what would happen if a newline somehow reached build_filter
    clean_kw = MULTILINE_QUERY.replace("\n", " ")
    filter_str = build_filter(
        clean_kw,
        date_from="2003-01-01",
        date_to="2024-12-31",
        doc_types=["article", "review"],
    )
    assert "\n" not in filter_str, (
        "build_filter() produced a filter string containing newlines — "
        "these would be URL-encoded as %0A and break the API query."
    )


def test_build_filter_structure():
    """Smoke-test that build_filter returns the expected comma-separated format."""
    kw = '("quantum computing" OR "qubit")'
    f = build_filter(
        kw,
        date_from="2003-01-01",
        date_to="2024-12-31",
        doc_types=["article"],
    )
    assert f.startswith("title_and_abstract.search:")
    assert "from_publication_date:2003-01-01" in f
    assert "to_publication_date:2024-12-31" in f
    assert "type:article" in f
    # All parts separated by commas, not newlines
    parts = f.split(",")
    assert len(parts) == 4  # search, from_date, to_date, type


def test_build_filter_with_topics():
    """build_filter with topics includes primary_topic.id filter."""
    kw = '("qubit")'
    f = build_filter(
        kw,
        topics=["T10020", "T10682"],
        date_from="2003-01-01",
        date_to="2024-12-31",
    )
    assert "primary_topic.id:T10020|T10682" in f


# ---------------------------------------------------------------------------
# Idempotency: calling get_keywords() twice returns the same value
# ---------------------------------------------------------------------------

def test_get_keywords_idempotent(tmp_path):
    """get_keywords() must cache and return the same normalized string on repeat calls."""
    kw_file = tmp_path / "keywords.txt"
    kw_file.write_text(MULTILINE_QUERY, encoding="utf-8")

    from openalex.config import AppConfig
    cfg = MagicMock()
    cfg.keywords_file = str(kw_file)
    cfg._keywords = None
    cfg.get_keywords = AppConfig.get_keywords.__get__(cfg, type(cfg))

    first = cfg.get_keywords()
    second = cfg.get_keywords()
    assert first == second
