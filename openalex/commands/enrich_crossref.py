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


def _load_target_rows(con, paper_id: str) -> list[dict[str, Any]]:
    """All contribution rows for a paper (for 1:1 matching against CrossRef).

    Matching needs the *full* author roster; filtering to rows still needing
    raw_affiliation_string happens at write time (COALESCE in the UPDATE
    plus a WHERE-still-NULL guard).
    """
    rows = con.execute(
        """
        SELECT c.row_id, c.author_id, c.author_position, a.orcid,
               c.raw_affiliation_string, c.institution_id
        FROM contributions c
        LEFT JOIN authors a ON a.id = c.author_id
        WHERE c.paper_id = ?
        ORDER BY c.row_id
        """,
        [paper_id],
    ).fetchall()
    return [
        {
            "row_id": r[0],
            "author_id": r[1],
            "author_position": r[2],
            "orcid": (r[3] or "").strip() or None,
            "has_raw_aff": bool(r[4] and r[4].strip()),
            "has_institution": r[5] is not None,
        }
        for r in rows
    ]


def _match_authors(
    crossref_authors: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> tuple[list[tuple[dict, dict, str]], dict[str, int]]:
    """Return list of (target_row, crossref_author, match_method) plus stats.

    Match path: ORCID first, fall back to position index iff list lengths
    match (so we don't pair the wrong authors when CrossRef includes
    corrigenda/extra entries).
    """
    s = {"matched_by_orcid": 0, "matched_by_position": 0, "unmatched_author": 0, "author_count_mismatch": 0}
    matched: list[tuple[dict, dict, str]] = []

    # ORCID pass
    cr_by_orcid = {a["orcid"]: a for a in crossref_authors if a.get("orcid")}
    remaining_targets = []
    remaining_cr = list(crossref_authors)
    for t in target_rows:
        if t["orcid"] and t["orcid"] in cr_by_orcid:
            cr = cr_by_orcid[t["orcid"]]
            matched.append((t, cr, "crossref orcid"))
            s["matched_by_orcid"] += 1
            if cr in remaining_cr:
                remaining_cr.remove(cr)
        else:
            remaining_targets.append(t)

    # Position-index pass — only if list lengths line up
    if remaining_targets and remaining_cr:
        if len(remaining_targets) == len(remaining_cr):
            for idx, t in enumerate(remaining_targets):
                cr = remaining_cr[idx]
                matched.append((t, cr, f"crossref position[{idx}]"))
                s["matched_by_position"] += 1
        else:
            s["author_count_mismatch"] += len(remaining_targets)

    return matched, s


def _flush_paper(
    con,
    paper_id: str,
    matched: list[tuple[dict, dict, str]],
) -> int:
    """Write raw_affiliation_string for one paper's matched rows; commit."""
    written = 0
    audit_rows: list[dict[str, Any]] = []
    for target, cr_author, method in matched:
        affs = cr_author.get("affiliation_strings") or []
        if not affs:
            continue
        raw = " | ".join(a for a in affs if a)
        if not raw:
            continue
        con.execute(
            """
            UPDATE contributions
            SET raw_affiliation_string = COALESCE(raw_affiliation_string, ?)
            WHERE row_id = ?
              AND (raw_affiliation_string IS NULL OR TRIM(raw_affiliation_string) = '')
            """,
            [raw, target["row_id"]],
        )
        audit_rows.append({
            "row_id": target["row_id"],
            "paper_id": paper_id,
            "country_code": None,
            "matched_terms": method,
            "raw_affiliation_string": raw,
            "source": "crossref",
            "confidence": 1.0,
        })
        written += 1
    if audit_rows:
        _insert_audit(con, audit_rows, stage="crossref")
        con.commit()
    return written


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
        "matched_by_orcid": 0,
        "matched_by_position": 0,
        "author_count_mismatch": 0,
        "raw_aff_populated": 0,
        "papers_enriched": 0,
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

            if not any(a["affiliation_strings"] for a in authors):
                stats["papers_no_affiliations"] += 1
                continue

            targets = _load_target_rows(con, paper_id)
            if not targets:
                # All eligible rows for this paper got filled by a prior run.
                continue

            matched, match_stats = _match_authors(authors, targets)
            stats["matched_by_orcid"] += match_stats["matched_by_orcid"]
            stats["matched_by_position"] += match_stats["matched_by_position"]
            stats["author_count_mismatch"] += match_stats["author_count_mismatch"]

            # Writable: matched row needs raw_affiliation_string AND CrossRef
            # actually has affiliations for that author.
            writable = [
                (t, cr, m) for (t, cr, m) in matched
                if cr.get("affiliation_strings") and not t["has_raw_aff"]
            ]
            if not writable:
                continue

            if dry_run:
                stats["raw_aff_populated"] += len(writable)
                stats["papers_enriched"] += 1
            else:
                written = _flush_paper(con, paper_id, writable)
                stats["raw_aff_populated"] += written
                if written:
                    stats["papers_enriched"] += 1

    return stats


def _print_summary(stats: dict[str, int], dry_run: bool) -> None:
    table = Table(title="enrich-crossref", show_lines=False)
    table.add_column("Metric", style="white")
    table.add_column("Count", justify="right", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " "), f"{v:,}")
    table.add_row("mode", "dry-run" if dry_run else "apply")
    console.print(table)
