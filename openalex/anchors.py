"""Anchor-paper parsing and presence checks."""

from __future__ import annotations

import asyncio
import re
import string
import unicodedata
from dataclasses import dataclass
from sre_compile import CH_UNICODE
from typing import Any

from rich.console import Console
from rich.table import Table

from openalex.api_client import AsyncOpenAlexClient

console = Console()

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.IGNORECASE)
_PUNCT_TRANSLATOR = str.maketrans({c: " " for c in string.punctuation})

import re


def build_topic_filter_chunks(api_filter: str, chunk_size: int = 40) -> list[str]:
    """
    Split an OpenAlex filter into multiple filters if it contains
    more than `chunk_size` primary_topic.id values.

    Returns:
        List[str]: One or more complete filter strings.
    """

    match = re.search(r"primary_topic\.id:([^,]+)", api_filter)

    # No topic filter present
    if not match:
        return [api_filter]

    topics_str = match.group(1)
    topics = topics_str.split("|")

    # Already small enough
    if len(topics) <= chunk_size:
        return [api_filter]

    chunked_filters = []

    for i in range(0, len(topics), chunk_size):
        chunk_topics = topics[i : i + chunk_size]
        chunk_filter = api_filter.replace(
            topics_str,
            "|".join(chunk_topics),
            1,  # Replace only the topic section
        )
        chunked_filters.append(chunk_filter)

    return chunked_filters


@dataclass(frozen=True)
class AnchorEntry:
    raw: str
    kind: str  # "doi" | "title"
    normalized: str


@dataclass(frozen=True)
class AnchorCheckResult:
    found: list[AnchorEntry]
    missing: list[AnchorEntry]
    invalid: list[str]


def normalize_doi(value: str) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None
    candidate = _DOI_PREFIX_RE.sub("", candidate).strip()
    candidate = candidate.lower()
    if _DOI_RE.match(candidate):
        return candidate
    return None


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    if not text:
        return ""
    text = text.translate(_PUNCT_TRANSLATOR)
    return re.sub(r"\s+", " ", text).strip()


def parse_anchor_entries(entries: list[str]) -> tuple[list[AnchorEntry], list[str]]:
    parsed: list[AnchorEntry] = []
    invalid: list[str] = []

    for raw in entries:
        value = raw.strip()
        if not value:
            continue
        doi = normalize_doi(value)
        if doi is not None:
            parsed.append(AnchorEntry(raw=value, kind="doi", normalized=doi))
            continue

        looks_like_doi = value.casefold().startswith(
            (
                "10.",
                "doi:",
                "https://doi.org/",
                "http://doi.org/",
                "https://dx.doi.org/",
                "http://dx.doi.org/",
            )
        )
        if looks_like_doi:
            invalid.append(value)
            continue

        title = normalize_title(value)
        if not title:
            invalid.append(value)
            continue
        parsed.append(AnchorEntry(raw=value, kind="title", normalized=title))

    return parsed, invalid


def _cfg_api_keys(cfg: Any) -> list[str]:
    """Support AppConfig.api_keys plus older test/config shims with api_key."""
    keys = getattr(cfg, "api_keys", None)
    if keys:
        return list(keys)
    key = getattr(cfg, "api_key", "")
    return [key] if key else []


async def _check_title_anchor(
    client: AsyncOpenAlexClient, api_filter: str, entry: AnchorEntry
) -> bool:
    """Query OpenAlex for a specific title anchor using title.search — O(1) per anchor."""
    words = entry.normalized.split()
    search_phrase = " ".join(words[:10])
    title_filter = f"{api_filter},title.search:{search_phrase}"
    data = await client.fetch_page(
        title_filter, cursor="*", extra_params={"select": "id,title", "per_page": "10"}
    )
    if not data:
        return False
    for paper in data.get("results", []):
        if normalize_title(paper.get("title") or "") == entry.normalized:
            return True
    return False


async def check_anchor_coverage(
    cfg: Any, api_filter: str, anchors: list[AnchorEntry]
) -> AnchorCheckResult:
    found: list[AnchorEntry] = []
    missing: list[AnchorEntry] = []

    doi_anchors = [a for a in anchors if a.kind == "doi"]
    title_anchors = [a for a in anchors if a.kind == "title"]

    found_dois: set[str] = set()

    if doi_anchors:
        unique_dois = sorted({a.normalized for a in doi_anchors})
        # OpenAlex supports OR logic for DOIs: doi:doi1|doi2|doi3
        # Small batches: combined URL also carries the keywords boolean and
        # 100+ topic IDs, so a 50-DOI batch overflows OpenAlex's URL length
        # limit and the server returns an HTML 400 page instead of JSON.
        batch_size = 10
        doi_batches = [
            unique_dois[i : i + batch_size]
            for i in range(0, len(unique_dois), batch_size)
        ]

        async with AsyncOpenAlexClient(
            api_keys=_cfg_api_keys(cfg),
            email=cfg.email,
            per_page=200,  # We want to see which ones matched
            max_retries=cfg.max_retries,
            retry_delay=cfg.retry_delay,
        ) as client:
            for batch in doi_batches:
                batch_filter = batch[0]
                # We check if these DOIs match the CURRENT keyword filter
                # We use fetch_page to get the actual results that matched
                for chunk_filter in build_topic_filter_chunks(api_filter):
                    data = await client.fetch_page(
                        f"{chunk_filter},doi:{batch_filter}",
                        cursor="*",
                        extra_params={"select": "doi"},
                    )
                    if data and "results" in data:
                        for res in data["results"]:
                            res_doi = normalize_doi(res.get("doi") or "")
                            if res_doi:
                                found_dois.add(res_doi)

    for entry in doi_anchors:
        (found if entry.normalized in found_dois else missing).append(entry)

    if title_anchors:
        async with AsyncOpenAlexClient(
            api_keys=_cfg_api_keys(cfg),
            email=cfg.email,
            per_page=cfg.per_page,
            max_retries=cfg.max_retries,
            retry_delay=cfg.retry_delay,
            concurrent_requests=cfg.concurrent_requests,
        ) as client:
            hits = await asyncio.gather(
                *(
                    _check_title_anchor(client, api_filter, entry)
                    for entry in title_anchors
                )
            )
        for entry, hit in zip(title_anchors, hits):
            (found if hit else missing).append(entry)

    return AnchorCheckResult(found=found, missing=missing, invalid=[])


def print_anchor_summary(result: AnchorCheckResult, context_name: str) -> None:
    total = len(result.found) + len(result.missing)
    table = Table(title=f"Anchor Coverage — {context_name}", show_lines=False)
    table.add_column("Metric", style="white")
    table.add_column("Count", justify="right", style="green")
    table.add_row("Anchors checked", str(total))
    table.add_row("Found", str(len(result.found)))
    table.add_row(
        "Missing", f"[red]{len(result.missing)}[/red]" if result.missing else "0"
    )
    if result.invalid:
        table.add_row("Invalid entries", f"[red]{len(result.invalid)}[/red]")
    console.print(table)

    if result.invalid:
        console.print("[bold red]✗ Invalid anchor entries:[/bold red]")
        for entry in result.invalid:
            console.print(f"  [red]• {entry}[/red]")

    if result.missing:
        console.print("[bold red]✗ Missing anchors:[/bold red]")
        for entry in result.missing:
            console.print(f"  [red]• [{entry.kind}] {entry.raw}[/red]")
    else:
        console.print(
            "[bold green]✓ All anchors are present in this result set.[/bold green]"
        )
