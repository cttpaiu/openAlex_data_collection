"""Web-of-Science Excel import helpers.

Pure helper module — no DB, no network. Loads per-country WoS export
files from ``data/QSM/*.xlsx`` and provides title-fingerprint /
DOI-normalisation utilities so the importer command can match against
existing OpenAlex papers in the DuckDB.

Filename → country code mapping covers every file we ship; unknown
filenames raise so the importer fails loudly rather than silently
skipping a country.

The ``WosRecord`` dataclass is the row-level structure other modules
consume; ``read_wos_country_file`` is the streaming entry point.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from openalex.imputation import normalize_country_code

# Master / aggregate files we deliberately skip during paper-level import.
AGGREGATE_FILENAMES: frozenset[str] = frozenset({
    "1 QSM-Total Publications Country Wise.xlsx",
    "Countries Quantum Sensing and Metrology - WoS.xlsx",
})

# Filename stem → ISO-3166-1 alpha-2. Maintained here (rather than via
# pycountry) because WoS exports use familiar short names ("UK", "US",
# "Hong Kong") that don't all match pycountry's official names.
COUNTRY_FILE_TO_CODE: dict[str, str] = {
    "Argentina": "AR",
    "Australia": "AU",
    "Austria": "AT",
    "Belgium": "BE",
    "Brazil": "BR",
    "Canada": "CA",
    "China": "CN",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "England": "GB",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Hong Kong": "HK",   # normalize_country_code maps to CN downstream
    "India": "IN",
    "Iran": "IR",
    "Israel": "IL",
    "Italy": "IT",
    "Japan": "JP",
    "Mexico": "MX",
    "Netherlands": "NL",
    "Norway": "NO",
    "Poland": "PL",
    "Russia": "RU",
    "Saudi Arabia": "SA",
    "Scotland": "GB",
    "Singapore": "SG",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Taiwan": "TW",
    "Turkey": "TR",
    "UAE": "AE",
    "UK": "GB",
    "US": "US",
}


def country_code_for_file(filename: str) -> str | None:
    """Return ISO-3166-1 alpha-2 for a WoS country-file name, or None
    for the aggregate / unrelated files."""
    name = Path(filename).name
    if name in AGGREGATE_FILENAMES:
        return None
    stem = Path(name).stem
    raw = COUNTRY_FILE_TO_CODE.get(stem)
    return normalize_country_code(raw) if raw else None


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation utilities
# ─────────────────────────────────────────────────────────────────────────────


def normalize_doi(raw: str | None) -> str | None:
    """Lowercase + strip URL/doi: prefix + whitespace."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s or None


_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_title(raw: str | None) -> str | None:
    """Aggressive normalisation for fuzzy title matching.

    Steps: NFKD-fold to strip diacritics, lower, drop punctuation, collapse
    whitespace. Returns ``None`` for empty input.
    """
    if raw is None:
        return None
    s = str(raw)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s or None


# ─────────────────────────────────────────────────────────────────────────────
# Row schema
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WosRecord:
    """One row from a WoS country Excel export.

    Only the columns we actually use downstream. The full sheet has 24
    columns; we keep the lightweight projection so iterating large
    files (~8k rows per country, 38 files) stays cheap.
    """

    accession_number: str | None
    doi: str | None             # raw, as written in the spreadsheet
    doi_normalized: str | None  # lowercase + prefix-stripped
    title: str | None
    title_normalized: str | None
    authors: str | None         # semicolon-separated string
    source: str | None          # journal title
    document_type: str | None
    publication_year: int | None
    times_cited: int | None
    top_1_percent: bool
    top_10_percent: bool
    percentile: float | None

    @classmethod
    def from_row(cls, row: dict[str, object]) -> "WosRecord":
        def _as_int(value: object) -> int | None:
            if value is None or value == "":
                return None
            try:
                return int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

        def _as_float(value: object) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

        def _as_bool_top(value: object) -> bool:
            """WoS marks top-percentile cells as 1 (often int); blanks are nulls."""
            if value is None or value == "":
                return False
            try:
                return int(value) == 1  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return bool(value)

        doi_raw = row.get("DOI")
        title_raw = row.get("Article Title")
        return cls(
            accession_number=(str(row.get("Accession Number")).strip() or None)
            if row.get("Accession Number") is not None else None,
            doi=str(doi_raw).strip() if doi_raw else None,
            doi_normalized=normalize_doi(doi_raw) if doi_raw else None,
            title=str(title_raw).strip() if title_raw else None,
            title_normalized=normalize_title(title_raw) if title_raw else None,
            authors=(str(row.get("Authors")).strip() or None)
            if row.get("Authors") is not None else None,
            source=(str(row.get("Source")).strip() or None)
            if row.get("Source") is not None else None,
            document_type=(str(row.get("Document Type")).strip() or None)
            if row.get("Document Type") is not None else None,
            publication_year=_as_int(row.get("Publication Date")),
            times_cited=_as_int(row.get("Times Cited")),
            top_1_percent=_as_bool_top(row.get("Top 1%")),
            top_10_percent=_as_bool_top(row.get("Top 10%")),
            percentile=_as_float(row.get("Percentile in Subject Area")),
        )


def read_wos_country_file(path: Path) -> Iterator[WosRecord]:
    """Yield :class:`WosRecord` rows from one country export.

    Lazy import of ``polars`` so unrelated CLI commands stay fast at
    startup. Aggregate files (master sheets) raise ``ValueError`` —
    callers should pre-filter via :data:`AGGREGATE_FILENAMES`.
    """
    if path.name in AGGREGATE_FILENAMES:
        raise ValueError(
            f"{path.name} is an aggregate file, not a per-paper export."
        )

    import polars as pl

    df = pl.read_excel(str(path))
    for row in df.iter_rows(named=True):
        yield WosRecord.from_row(row)
