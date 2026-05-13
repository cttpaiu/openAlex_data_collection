"""Unit tests for openalex/wos.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from openalex.wos import (
    AGGREGATE_FILENAMES,
    COUNTRY_FILE_TO_CODE,
    WosRecord,
    country_code_for_file,
    normalize_doi,
    normalize_title,
)


# ─────────────────────────────────────────────────────────────────────────────
# Country-code mapping
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("China.xlsx", "CN"),
        ("US.xlsx", "US"),
        ("UK.xlsx", "GB"),
        ("England.xlsx", "GB"),
        ("Scotland.xlsx", "GB"),
        ("Hong Kong.xlsx", "CN"),   # HK → CN via normalize_country_code
        ("South Korea.xlsx", "KR"),
        ("Czech Republic.xlsx", "CZ"),
        ("Saudi Arabia.xlsx", "SA"),
        ("UAE.xlsx", "AE"),
        ("South Africa.xlsx", "ZA"),
    ],
)
def test_country_code_for_file_known_countries(filename, expected):
    assert country_code_for_file(filename) == expected


@pytest.mark.parametrize(
    "filename",
    list(AGGREGATE_FILENAMES),
)
def test_country_code_for_file_returns_none_for_aggregate_files(filename):
    assert country_code_for_file(filename) is None


def test_country_code_for_file_unknown_filename_returns_none():
    assert country_code_for_file("Atlantis.xlsx") is None


def test_country_file_mapping_covers_every_country_export(tmp_path):
    """Sanity-check the mapping table itself.

    Every per-country .xlsx we ship in data/QSM must have a mapping
    entry; aggregate files don't need one. This test only exercises the
    map itself (not the actual filesystem) so it stays green when the
    QSM data folder isn't checked in.
    """
    # Every key in the map must resolve to a non-None ISO code.
    for stem, raw_code in COUNTRY_FILE_TO_CODE.items():
        code = country_code_for_file(f"{stem}.xlsx")
        assert code is not None, f"mapping for {stem!r} is missing/normalised away"
        assert len(code) == 2, f"code for {stem!r} not alpha-2: {code!r}"


# ─────────────────────────────────────────────────────────────────────────────
# DOI normalisation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10.1038/Foo", "10.1038/foo"),
        ("  10.1038/foo  ", "10.1038/foo"),
        ("https://doi.org/10.1038/Foo", "10.1038/foo"),
        ("http://doi.org/10.1038/foo", "10.1038/foo"),
        ("DOI:10.1038/Foo", "10.1038/foo"),
        ("", None),
        (None, None),
    ],
)
def test_normalize_doi(raw, expected):
    assert normalize_doi(raw) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Title normalisation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Quantum Computing: A Survey", "quantum computing a survey"),
        ("  multiple   spaces  ", "multiple spaces"),
        ("Müller über Bose-Einstein", "muller uber bose einstein"),
        ("Title (with) [brackets] & punctuation!", "title with brackets punctuation"),
        ("", None),
        (None, None),
        ("   ", None),
    ],
)
def test_normalize_title(raw, expected):
    assert normalize_title(raw) == expected


def test_normalize_title_diacritics_collapse():
    """NFKD strip lets WoS 'Schrödinger' match OpenAlex 'Schrodinger'."""
    a = normalize_title("Schrödinger equation")
    b = normalize_title("Schrodinger equation")
    assert a == b


# ─────────────────────────────────────────────────────────────────────────────
# WosRecord parsing
# ─────────────────────────────────────────────────────────────────────────────


def _row(**overrides) -> dict[str, object]:
    """Build a minimal WoS row with the columns the parser uses."""
    base: dict[str, object] = {
        "Accession Number": "WOS:000123",
        "DOI": "10.1038/foo",
        "Article Title": "Some Title",
        "Authors": "Smith, J.; Doe, A.",
        "Source": "NATURE",
        "Document Type": "Article",
        "Publication Date": 2020,
        "Times Cited": 42,
        "Top 1%": None,
        "Top 10%": 1,
        "Percentile in Subject Area": 89.5,
    }
    base.update(overrides)
    return base


def test_wos_record_basic_parse():
    rec = WosRecord.from_row(_row())
    assert rec.accession_number == "WOS:000123"
    assert rec.doi == "10.1038/foo"
    assert rec.doi_normalized == "10.1038/foo"
    assert rec.title == "Some Title"
    assert rec.title_normalized == "some title"
    assert rec.publication_year == 2020
    assert rec.times_cited == 42
    assert rec.top_1_percent is False
    assert rec.top_10_percent is True
    assert rec.percentile == pytest.approx(89.5)


def test_wos_record_handles_missing_doi_and_title():
    rec = WosRecord.from_row(_row(DOI=None, **{"Article Title": None}))
    assert rec.doi is None
    assert rec.doi_normalized is None
    assert rec.title is None
    assert rec.title_normalized is None


def test_wos_record_top_percentile_blanks_are_false():
    """WoS leaves Top 1%/10% blank for non-marked papers."""
    rec = WosRecord.from_row(_row(**{"Top 1%": None, "Top 10%": ""}))
    assert rec.top_1_percent is False
    assert rec.top_10_percent is False


def test_wos_record_top_percentile_truthy_int():
    rec = WosRecord.from_row(_row(**{"Top 1%": 1, "Top 10%": 1}))
    assert rec.top_1_percent is True
    assert rec.top_10_percent is True


def test_wos_record_non_numeric_year_falls_back_to_none():
    rec = WosRecord.from_row(_row(**{"Publication Date": "early-access"}))
    assert rec.publication_year is None


def test_wos_record_doi_with_url_is_normalized():
    rec = WosRecord.from_row(_row(DOI="HTTPS://doi.org/10.1038/Foo"))
    assert rec.doi == "HTTPS://doi.org/10.1038/Foo"
    assert rec.doi_normalized == "10.1038/foo"
