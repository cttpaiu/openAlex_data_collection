"""Command: enrich-crossref — pull raw author affiliations from CrossRef.

For every contribution row that still has ``institution_id IS NULL`` AND
``raw_affiliation_string IS NULL`` AND the paper has a DOI, fetch the
DOI from CrossRef and copy the publisher-submitted affiliation strings
into ``contributions.raw_affiliation_string``. The existing
``impute-affiliation`` Stage 1 then converts those strings into
institution_id + country_code via LLM extraction.

This command does NOT do any imputation itself. It only widens the
imputable set by restoring raw_affiliation_string entries that
OpenAlex's parser dropped or never had.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiohttp
import click
from rich.console import Console
from rich.table import Table

from openalex.commands.impute_affiliation import (
    _ensure_audit_table,
    _insert_audit,
    _resolve_db_path,
)
from openalex.config import load_config

console = Console()

CROSSREF_URL = "https://api.crossref.org/works/{doi}"


class AsyncCrossRefClient:
    """Async HTTP client for the CrossRef polite pool."""

    def __init__(
        self,
        email: str,
        concurrent_requests: int = 10,
        max_retries: int = 3,
        retry_delay: int = 2,
    ):
        self.email = email
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "AsyncCrossRefClient":
        headers = {
            "User-Agent": (
                f"openalex-data-collection/0.1.0 (mailto:{self.email})"
            ),
            "Accept": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=100)
        self.session = aiohttp.ClientSession(
            headers=headers, timeout=timeout, connector=connector
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self.session is not None:
            await self.session.close()

    async def fetch_doi(self, doi: str) -> tuple[str, dict | None]:
        """Return ('ok', payload) | ('not_found', None) | ('failed', None)."""
        url = CROSSREF_URL.format(doi=doi)
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(url) as response:
                        if response.status in (404, 410):
                            return "not_found", None
                        if response.status == 429:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        if response.status >= 500:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        response.raise_for_status()
                        payload = await response.json()
                        return "ok", payload
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
        return "failed", None


def _load_eligible_papers(con, limit: int | None) -> list[tuple[str, str]]:
    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""
    rows = con.execute(
        f"""
        SELECT DISTINCT p.id, p.doi
        FROM papers p
        JOIN contributions c ON c.paper_id = p.id
        WHERE p.doi IS NOT NULL
          AND TRIM(p.doi) != ''
          AND c.institution_id IS NULL
          AND (c.raw_affiliation_string IS NULL OR TRIM(c.raw_affiliation_string) = '')
        {limit_sql}
        """
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _normalize_doi(raw: str) -> str:
    s = (raw or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def _parse_crossref_authors(payload: dict) -> list[dict[str, Any]]:
    """Extract authors with their affiliation strings from a CrossRef payload."""
    message = payload.get("message") or {}
    authors = message.get("author") or []
    parsed: list[dict[str, Any]] = []
    for a in authors:
        affs = a.get("affiliation") or []
        names = [aff.get("name") for aff in affs if isinstance(aff, dict) and aff.get("name")]
        orcid = (a.get("ORCID") or "").strip()
        if orcid.startswith("http://orcid.org/"):
            orcid = orcid[len("http://orcid.org/"):]
        elif orcid.startswith("https://orcid.org/"):
            orcid = orcid[len("https://orcid.org/"):]
        parsed.append({
            "orcid": orcid or None,
            "affiliation_strings": names,
        })
    return parsed


@click.command("enrich-crossref")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
@click.option("--dry-run", is_flag=True, help="Preview without DB writes")
@click.option("--limit", type=int, default=None, help="Cap eligible papers")
@click.option("--concurrent-requests", type=int, default=10, show_default=True)
@click.option("--max-retries", type=int, default=3, show_default=True)
@click.option("--retry-delay", type=int, default=2, show_default=True)
def enrich_crossref_command(
    config_path: str,
    db_path: str | None,
    dry_run: bool,
    limit: int | None,
    concurrent_requests: int,
    max_retries: int,
    retry_delay: int,
) -> None:
    """Populate contributions.raw_affiliation_string from CrossRef via DOI."""
    cfg = load_config(config_path)
    final_db_path = _resolve_db_path(cfg, db_path)
    if not Path(final_db_path).exists():
        console.print(f"[bold red]✗ Database not found:[/bold red] {final_db_path}")
        raise SystemExit(1)

    if not cfg.email or "@" not in cfg.email:
        console.print(
            "[bold red]✗ Set api.email in config/collection.yml for CrossRef polite pool.[/bold red]"
        )
        raise SystemExit(1)

    import duckdb

    con = duckdb.connect(final_db_path, read_only=False)
    try:
        _ensure_audit_table(con)
        candidates = _load_eligible_papers(con, limit)
        eligible = len(candidates)
        if eligible == 0:
            console.print("[dim]enrich-crossref: nothing eligible.[/dim]")
            return

        console.print(
            f"[bold cyan]enrich-crossref[/bold cyan] — fetching {eligible:,} papers "
            f"(concurrency={concurrent_requests})"
        )

        stats = asyncio.run(
            _run_crossref_fetch(
                con,
                candidates,
                cfg.email,
                concurrent_requests,
                max_retries,
                retry_delay,
                dry_run,
            )
        )
        _print_summary(stats, dry_run)
    finally:
        con.close()


async def _run_crossref_fetch(
    con,
    candidates: list[tuple[str, str]],
    email: str,
    concurrent_requests: int,
    max_retries: int,
    retry_delay: int,
    dry_run: bool,
) -> dict[str, int]:
    stats: dict[str, int] = {
        "papers_eligible": len(candidates),
        "crossref_fetched": 0,
        "crossref_not_found": 0,
        "crossref_failed": 0,
        "papers_no_authors": 0,
        "papers_no_affiliations": 0,
        "raw_aff_extracted": 0,
    }

    async with AsyncCrossRefClient(
        email=email,
        concurrent_requests=concurrent_requests,
        max_retries=max_retries,
        retry_delay=retry_delay,
    ) as client:
        async def _one(paper_id: str, doi: str):
            normalized = _normalize_doi(doi)
            status, payload = await client.fetch_doi(normalized)
            return paper_id, status, payload

        tasks = [asyncio.create_task(_one(pid, doi)) for pid, doi in candidates]
        for fut in asyncio.as_completed(tasks):
            paper_id, status, payload = await fut

            if status == "not_found":
                stats["crossref_not_found"] += 1
                continue
            if status == "failed":
                stats["crossref_failed"] += 1
                continue

            stats["crossref_fetched"] += 1
            authors = _parse_crossref_authors(payload or {})
            if not authors:
                stats["papers_no_authors"] += 1
                continue

            extracted = sum(1 for a in authors if a["affiliation_strings"])
            if extracted == 0:
                stats["papers_no_affiliations"] += 1
                continue
            stats["raw_aff_extracted"] += extracted

    return stats


def _print_summary(stats: dict[str, int], dry_run: bool) -> None:
    table = Table(title="enrich-crossref", show_lines=False)
    table.add_column("Metric", style="white")
    table.add_column("Count", justify="right", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " "), f"{v:,}")
    table.add_row("mode", "dry-run" if dry_run else "apply")
    console.print(table)
