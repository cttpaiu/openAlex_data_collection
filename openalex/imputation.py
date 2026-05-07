"""Country imputation helpers for affiliation strings."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CountryInference:
    country_code: str | None
    status: str  # unambiguous | ambiguous | none
    matched_terms: tuple[str, ...]


COUNTRY_NORMALIZATION: dict[str, str] = {
    "HK": "CN",  # project requirement: treat Hong Kong as China
}


def normalize_country_code(code: str | None) -> str | None:
    if not code:
        return None
    c = code.strip().upper()
    if not c:
        return None
    return COUNTRY_NORMALIZATION.get(c, c)


COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "US": ("united states", "united states of america", "usa", "u.s.a.", "u.s.", "america"),
    "CN": ("china", "people's republic of china", "peoples republic of china", "pr china", "p.r. china"),
    "JP": ("japan",),
    "DE": ("germany", "deutschland"),
    "GB": ("united kingdom", "uk", "england", "scotland", "wales", "northern ireland", "great britain"),
    "FR": ("france",),
    "IT": ("italy",),
    "ES": ("spain",),
    "NL": ("netherlands", "the netherlands"),
    "CH": ("switzerland",),
    "SE": ("sweden",),
    "NO": ("norway",),
    "FI": ("finland",),
    "DK": ("denmark",),
    "BE": ("belgium",),
    "AT": ("austria",),
    "CA": ("canada",),
    "AU": ("australia",),
    "NZ": ("new zealand",),
    "IN": ("india",),
    "KR": ("south korea", "republic of korea", "korea"),
    "RU": ("russia", "russian federation"),
    "IL": ("israel",),
    "SG": ("singapore",),
    "TW": ("taiwan",),
    "BR": ("brazil",),
    "PL": ("poland",),
    "IR": ("iran", "iran, islamic republic of"),
    "IE": ("ireland",),
    "PT": ("portugal",),
    "GR": ("greece",),
    "CZ": ("czech republic", "czechia"),
    "HU": ("hungary",),
    "TR": ("turkey", "türkiye", "turkiye"),
    "MX": ("mexico",),
    "AR": ("argentina",),
    "CL": ("chile",),
    "ZA": ("south africa",),
    "HK": ("hong kong", "hong kong sar"),
}


def _boundary_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


COUNTRY_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (code, term, _boundary_pattern(term))
    for code, terms in COUNTRY_ALIASES.items()
    for term in terms
]


def infer_country_from_affiliation(raw_affiliation: str) -> CountryInference:
    text = (raw_affiliation or "").strip()
    if not text:
        return CountryInference(country_code=None, status="none", matched_terms=())

    matched_codes: set[str] = set()
    matched_terms: list[str] = []
    for code, term, pattern in COUNTRY_PATTERNS:
        if pattern.search(text):
            matched_codes.add(code)
            matched_terms.append(term)

    if len(matched_codes) == 1:
        code = normalize_country_code(next(iter(matched_codes)))
        return CountryInference(country_code=code, status="unambiguous", matched_terms=tuple(matched_terms))
    if len(matched_codes) > 1:
        return CountryInference(country_code=None, status="ambiguous", matched_terms=tuple(matched_terms))
    return CountryInference(country_code=None, status="none", matched_terms=())
