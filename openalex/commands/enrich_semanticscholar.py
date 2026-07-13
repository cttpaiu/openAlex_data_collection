"""Command: enrich-semanticscholar — pull raw author affiliations from Semantic Scholar."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
import click
from rich.console import Console
from rich.table import Table

from openalex.commands.impute_affiliation import (
    _StageProgress,
    _ensure_audit_table,
    _insert_audit,
    _resolve_db_path,
)
from openalex.config import load_config
from openalex.wos import normalize_doi

console = Console()

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=authors.affiliations"


class AsyncS2Client:
    """Async HTTP client for Semantic Scholar API."""

    def __init__(
        self,
        api_key: str | None = None,
        concurrent_requests: int = 1, # S2 rate limit is strict (1 req/sec for free tier)
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "AsyncS2Client":
        headers = {
            "User-Agent": "openalex-data-collection/0.1.0",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=10)
        self.session = aiohttp.ClientSession(
            headers=headers, timeout=timeout, connector=connector
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self.session is not None:
            await self.session.close()

    async def fetch_doi(self, doi: str) -> tuple[str, dict | None]:
        """Return ('ok', payload) | ('not_found', None) | ('failed', None)."""
        url = S2_URL.format(doi=doi)
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(url) as response:
                        if response.status == 404:
                            return "not_found", None
                        if response.status == 429:
                            wait = self.retry_delay * (2 ** attempt)
                            await asyncio.sleep(wait)
                            continue
                        response.raise_for_status()
                        payload = await response.json()
                        return "ok", payload
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
        return "failed", None


def _load_eligible_papers(con, limit: int | None) -> List[tuple[str, str]]:
    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""
    rows = con.execute(
        f"""
        SELECT DISTINCT p.id, p.doi
        FROM papers p
        JOIN contributions c ON c.paper_id = p.id
        WHERE p.doi IS NOT NULL AND TRIM(p.doi) != ''
          AND c.institution_id IS NULL
          AND (c.raw_affiliation_string IS NULL OR TRIM(c.raw_affiliation_string) = '')
        {limit_sql}
        """
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _parse_s2_authors(payload: dict) -> List[dict[str, Any]]:
    """Extract authors with their affiliation strings from Semantic Scholar payload."""
    authors = payload.get("authors") or []
    parsed: List[dict[str, Any]] = []
    for a in authors:
        affs = a.get("affiliations") or []
        # affs is a list of strings in S2 API
        parsed.append({
            "name": a.get("name"),
            "affiliation_strings": [aff for aff in affs if isinstance(aff, str) and aff.strip()],
        })
    return parsed


def _match_authors(
    s2_authors: List[dict[str, Any]],
    target_rows: List[dict[str, Any]],
) -> tuple[List[tuple[dict, dict, str]], dict[str, int]]:
    """Match S2 authors to DB rows by position index."""
    s = {"matched_by_position": 0, "author_count_mismatch": 0}
    matched: List[tuple[dict, dict, str]] = []

    # S2 doesn't give ORCIDs reliably in this endpoint, so we use position fallback
    if len(s2_authors) == len(target_rows):
        for idx, (s2_a, t_r) in enumerate(zip(s2_authors, target_rows)):
            matched.append((t_r, s2_a, f"s2 position[{idx}]"))
            s["matched_by_position"] += 1
    else:
        s["author_count_mismatch"] += len(target_rows)

    return matched, s


def _flush_paper(
    con,
    paper_id: str,
    matched: List[tuple[dict, dict, str]],
) -> int:
    """Write raw_affiliation_string for one paper; commit."""
    written = 0
    audit_rows: List[dict[str, Any]] = []
    for target, s2_author, method in matched:
        affs = s2_author.get("affiliation_strings") or []
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
            "source": "semanticscholar",
            "confidence": 1.0,
        })
        written += 1
    if audit_rows:
        _insert_audit(con, audit_rows, stage="semanticscholar")
        con.commit()
    return written


@click.command("semanticscholar")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
@click.option("--dry-run", is_flag=True, help="Preview without DB writes")
@click.option("--limit", type=int, default=None, help="Cap eligible papers")
@click.option("--api-key", help="Semantic Scholar API key")
@click.option("--concurrent-requests", type=int, default=1, show_default=True)
def enrich_semanticscholar_command(
    config_path: str,
    db_path: str | None,
    dry_run: bool,
    limit: int | None,
    api_key: str | None,
    concurrent_requests: int,
) -> None:
    """Populate contributions.raw_affiliation_string from Semantic Scholar via DOI."""
    cfg = load_config(config_path)
    final_db_path = _resolve_db_path(cfg, db_path)
    if not Path(final_db_path).exists():
        console.print(f"[bold red]✗ Database not found:[/bold red] {final_db_path}")
        raise SystemExit(1)

    import duckdb

    con = duckdb.connect(final_db_path, read_only=False)
    try:
        _ensure_audit_table(con)
        candidates = _load_eligible_papers(con, limit)
        eligible = len(candidates)
        if eligible == 0:
            console.print("[dim]enrich-semanticscholar: nothing eligible.[/dim]")
            return

        console.print(
            f"[bold cyan]enrich-semanticscholar[/bold cyan] — fetching {eligible:,} papers "
            f"(concurrency={concurrent_requests})"
        )

        stats = asyncio.run(
            _run_s2_fetch(
                con,
                candidates,
                api_key or cfg.semanticscholar_api_key,
                concurrent_requests,
                dry_run,
            )
        )
        _print_summary(stats, dry_run)
    finally:
        con.close()


async def _run_s2_fetch(
    con,
    candidates: List[tuple[str, str]],
    api_key: str | None,
    concurrent_requests: int,
    dry_run: bool,
) -> Dict[str, int]:
    stats: Dict[str, int] = {
        "papers_eligible": len(candidates),
        "s2_fetched": 0,
        "s2_not_found": 0,
        "s2_failed": 0,
        "papers_no_authors": 0,
        "papers_no_affiliations": 0,
        "matched_by_position": 0,
        "author_count_mismatch": 0,
        "raw_aff_populated": 0,
        "papers_enriched": 0,
    }

    async with AsyncS2Client(
        api_key=api_key,
        concurrent_requests=concurrent_requests,
    ) as client:
        async def _one(paper_id: str, doi: str):
            normalized = normalize_doi(doi)
            status, payload = await client.fetch_doi(normalized)
            return paper_id, status, payload

        total = len(candidates)
        with _StageProgress(
            f"enrich-semanticscholar ({total:,} papers)", total, unit="paper"
        ) as bar:
            tasks = [asyncio.create_task(_one(pid, doi)) for pid, doi in candidates]
            # S2 rate limits are very strict, better to process somewhat sequentially if free tier
            for fut in asyncio.as_completed(tasks):
                paper_id, status, payload = await fut
                outcome = "?"

                if status == "not_found":
                    stats["s2_not_found"] += 1
                    outcome = "404"
                elif status == "failed":
                    stats["s2_failed"] += 1
                    outcome = "[red]failed[/red]"
                else:
                    stats["s2_fetched"] += 1
                    authors = _parse_s2_authors(payload or {})
                    if not authors:
                        stats["papers_no_authors"] += 1
                        outcome = "no authors"
                    elif not any(a["affiliation_strings"] for a in authors):
                        stats["papers_no_affiliations"] += 1
                        outcome = "no affs"
                    else:
                        from openalex.commands.enrich_crossref import _load_target_rows
                        targets = _load_target_rows(con, paper_id)
                        if not targets:
                            outcome = "already filled"
                        else:
                            matched, match_stats = _match_authors(authors, targets)
                            stats["matched_by_position"] += match_stats["matched_by_position"]
                            stats["author_count_mismatch"] += match_stats["author_count_mismatch"]

                            writable = [
                                (t, s2_a, m) for (t, s2_a, m) in matched
                                if s2_a.get("affiliation_strings") and not t["has_raw_aff"]
                            ]
                            if not writable:
                                outcome = "no writable rows"
                            elif dry_run:
                                stats["raw_aff_populated"] += len(writable)
                                stats["papers_enriched"] += 1
                                outcome = f"[green]dry: +{len(writable)} rows[/green]"
                            else:
                                written = _flush_paper(con, paper_id, writable)
                                stats["raw_aff_populated"] += written
                                if written:
                                    stats["papers_enriched"] += 1
                                outcome = f"[green]+{written} rows[/green]"

                bar.log(f"{paper_id} → {outcome}")
                bar.advance()

    return stats


def _print_summary(stats: Dict[str, int], dry_run: bool) -> None:
    table = Table(title="enrich-semanticscholar", show_lines=False)
    table.add_column("Metric", style="white")
    table.add_column("Count", justify="right", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " "), f"{v:,}")
    table.add_row("mode", "dry-run" if dry_run else "apply")
    console.print(table)
