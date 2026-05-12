"""Country imputation helpers for affiliation strings.

Also hosts dataclasses for structured LLM responses (Stage 1: institution
imputation, Stage 3: country imputation) and the embedding-based
``InstitutionMatcher`` used to dedup against existing OpenAlex institution
records before creating any synthetic entries.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, model_validator


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
    "US": ("united states", "united states of america", "usa", "u.s.a.", "u.s.a", "u.s.", "america"),
    "CN": ("china", "people's republic of china", "peoples republic of china", "pr china", "p.r. china", "p.r.c.", "p.r.c"),
    "JP": ("japan",),
    "DE": ("germany", "deutschland"),
    "GB": ("united kingdom", "uk", "u.k.", "u.k", "england", "scotland", "wales", "northern ireland", "great britain"),
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


# ─────────────────────────────────────────────────────────────────────────────
# Structured LLM response dataclasses (Stage 1 + Stage 3)
# ─────────────────────────────────────────────────────────────────────────────


def _to_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


@dataclass
class InstitutionPrediction:
    """Stage 1: LLM extraction of institution + country from raw affiliation."""

    row_id: int
    institution_name: str | None
    country_code: str | None
    confidence: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InstitutionPrediction":
        cc = _to_str_or_none(d.get("country_code"))
        return cls(
            row_id=int(d.get("row_id", 0)),
            institution_name=_to_str_or_none(d.get("institution_name")),
            country_code=cc.upper() if cc else None,
            confidence=float(d.get("confidence") or 0.0),
        )

    @classmethod
    def from_pydantic(cls, item: "InstitutionPredictionItem") -> "InstitutionPrediction":
        cc = _to_str_or_none(item.country_code)
        return cls(
            row_id=int(item.row_id),
            institution_name=_to_str_or_none(item.institution_name),
            country_code=cc.upper() if cc else None,
            confidence=float(item.confidence or 0.0),
        )


@dataclass
class CountryPrediction:
    """Stage 3: LLM inference of country from raw affiliation only."""

    row_id: int
    country_code: str | None
    status: str
    confidence: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CountryPrediction":
        cc = _to_str_or_none(d.get("country_code"))
        return cls(
            row_id=int(d.get("row_id", 0)),
            country_code=cc.upper() if cc else None,
            status=(_to_str_or_none(d.get("status")) or "none").lower(),
            confidence=float(d.get("confidence") or 0.0),
        )

    @classmethod
    def from_pydantic(cls, item: "CountryPredictionItem") -> "CountryPrediction":
        cc = _to_str_or_none(item.country_code)
        return cls(
            row_id=int(item.row_id),
            country_code=cc.upper() if cc else None,
            status=(_to_str_or_none(item.status) or "none").lower(),
            confidence=float(item.confidence or 0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic batch-response schemas for langchain .with_structured_output(...)
# ─────────────────────────────────────────────────────────────────────────────


class InstitutionPredictionItem(BaseModel):
    row_id: int = Field(description="Echo of the input row_id")
    institution_name: str | None = Field(
        default=None,
        description="Primary institution name (university, lab, company); no department or address",
    )
    country_code: str | None = Field(
        default=None, description="ISO-3166-1 alpha-2, or null if unknown"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class InstitutionPredictionResponse(BaseModel):
    predictions: list[InstitutionPredictionItem]

    @model_validator(mode="before")
    @classmethod
    def _drop_incomplete(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("predictions"), list):
            data["predictions"] = [
                p for p in data["predictions"]
                if isinstance(p, dict) and p.get("row_id") is not None
            ]
        return data


class CountryPredictionItem(BaseModel):
    row_id: int = Field(description="Echo of the input row_id")
    country_code: str | None = Field(
        default=None, description="ISO-3166-1 alpha-2, or null if unknown"
    )
    status: str = Field(
        default="none",
        description="One of: unambiguous, ambiguous, none",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CountryPredictionResponse(BaseModel):
    predictions: list[CountryPredictionItem]

    @model_validator(mode="before")
    @classmethod
    def _drop_incomplete(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("predictions"), list):
            data["predictions"] = [
                p for p in data["predictions"]
                if isinstance(p, dict) and p.get("row_id") is not None
            ]
        return data


# ─────────────────────────────────────────────────────────────────────────────
# Embedding-based institution matcher (Stage 1 dedup)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InstitutionRecord:
    id: str
    display_name: str
    country_code: str | None


@dataclass(frozen=True)
class InstitutionMatch:
    institution_id: str
    display_name: str
    country_code: str | None
    score: float


def synthetic_institution_id(name: str) -> str:
    """Stable synthetic ID for institutions we create from raw affiliations.

    SHA1 over normalized name → first 10 hex chars, prefixed ``IMP_``.
    Idempotent: same name always yields same ID, so re-runs don't duplicate.
    """
    norm = re.sub(r"\s+", " ", (name or "").strip().lower())
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:10]
    return f"IMP_{digest}"


class InstitutionMatcher:
    """Embedding-based fuzzy matcher.

    Loads sentence-transformers lazily (heavy import) so CLI commands that
    don't need imputation start fast.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self.records: list[InstitutionRecord] = []
        self._embeddings = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, records: list[InstitutionRecord]) -> None:
        self.records = records
        if not records:
            self._embeddings = None
            return
        model = self._load_model()
        names = [r.display_name for r in records]
        self._embeddings = model.encode(
            names, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True
        )

    def find_match(self, query: str, threshold: float = 0.78) -> InstitutionMatch | None:
        candidates = self.top_k(query, k=1)
        if not candidates:
            return None
        best = candidates[0]
        return best if best.score >= threshold else None

    def top_k(self, query: str, k: int = 3) -> list[InstitutionMatch]:
        if not self.records or self._embeddings is None or not query:
            return []
        from sentence_transformers import util  # heavy
        model = self._load_model()
        query_emb = model.encode(
            [query], convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True
        )
        scores = util.cos_sim(query_emb, self._embeddings)[0]
        k = min(k, len(self.records))
        topk = scores.topk(k)
        return [
            InstitutionMatch(
                institution_id=self.records[int(i)].id,
                display_name=self.records[int(i)].display_name,
                country_code=self.records[int(i)].country_code,
                score=float(scores[int(i)]),
            )
            for i in topk.indices.tolist()
        ]
