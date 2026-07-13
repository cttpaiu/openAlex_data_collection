"""Command: enrich-openalex — re-query OpenAlex for existing DB stubs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from openalex.api_client import AsyncOpenAlexClient
from openalex.commands.database import OpenAlexLoader
from openalex.commands.impute_affiliation import _resolve_db_path
from openalex.config import load_config
from openalex.wos import normalize_doi

console = Console()


def _ensure_columns(con) -> None:
    """Add source and from_wos columns if missing."""
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS source VARCHAR")
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS from_wos BOOLEAN DEFAULT FALSE")
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS wos_accession_number VARCHAR")


def _load_eligible_papers(con, limit: int | None) -> List[tuple[str | None, str | None]]:
    """Papers that are WoS stubs or missing institution_id on contributions."""
    _ensure_columns(con)
    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""
    # We prioritize papers tagged as source='web_of_science' or from_wos=TRUE
    # OR any paper that has a DOI but still has contributions with institution_id IS NULL
    rows = con.execute(
        f"""
        SELECT DISTINCT p.doi, p.title
        FROM papers p
        JOIN contributions c ON c.paper_id = p.id
        WHERE c.institution_id IS NULL
          AND (
            (p.doi IS NOT NULL AND TRIM(p.doi) != '')
            OR (p.title IS NOT NULL AND TRIM(p.title) != '' AND p.title != 'Untitled')
          )
        {limit_sql}
        """
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


@click.command("openalex")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
@click.option("--dry-run", is_flag=True, help="Preview without DB writes")
@click.option("--limit", type=int, default=None, help="Cap eligible papers")
@click.option("--concurrency", type=int, default=None, help="Override concurrent requests")
def enrich_openalex_command(
    config_path: str,
    db_path: str | None,
    dry_run: bool,
    limit: int | None,
    concurrency: int | None,
) -> None:
    """Re-query OpenAlex for papers marked as Web of Science stubs."""
    cfg = load_config(config_path)
    final_db_path = _resolve_db_path(cfg, db_path)
    if not Path(final_db_path).exists():
        console.print(f"[bold red]✗ Database not found:[/bold red] {final_db_path}")
        raise SystemExit(1)

    loader = OpenAlexLoader(final_db_path)
    con = loader.con

    try:
        papers = _load_eligible_papers(con, limit)
        eligible = len(papers)
        if eligible == 0:
            console.print("[dim]enrich-openalex: no eligible papers found.[/dim]")
            return

        console.print(
            f"[bold cyan]enrich-openalex[/bold cyan] — fetching {eligible:,} papers from OpenAlex"
        )

        if dry_run:
            console.print("[yellow]Dry run: exiting.[/yellow]")
            return

        stats = asyncio.run(_run_enrich(cfg, loader, papers, concurrency))
        _print_summary(stats)
    finally:
        loader.close()


async def _run_enrich(cfg, loader, papers: List[tuple[str|None, str|None]], concurrency: int | None) -> Dict[str, int]:
    stats = {"eligible": len(papers), "found": 0, "not_found": 0, "errors": 0}
    con = loader.con
    effective_concurrency = concurrency or cfg.concurrent_requests

    async with AsyncOpenAlexClient(
        api_keys=cfg.api_keys,
        email=cfg.email,
        concurrent_requests=effective_concurrency,
    ) as client:
        # Separate into DOI papers and Title-only papers
        doi_papers = [p for p in papers if p[0]]
        title_only_papers = [p for p in papers if not p[0] and p[1]]

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[green]{task.completed:,}[/green]/[dim]{task.total:,}[/dim]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching from OpenAlex...", total=len(papers))

            # DB writes stay on the event-loop thread (single-threaded), so the
            # callbacks below mutate `con`/`stats` sequentially even though many
            # network requests are in flight. The client semaphore caps in-flight
            # requests at `concurrent_requests`.

            # 1. DOI batch fetch — one OpenAlex query per 50 DOIs, all batches
            #    issued concurrently (previously awaited serially → no concurrency).
            batch_size = 50
            doi_batches = [
                doi_papers[i : i + batch_size]
                for i in range(0, len(doi_papers), batch_size)
            ]

            async def handle_doi_batch(batch: List[tuple[str | None, str | None]]) -> None:
                normalized_dois = [d for d in (normalize_doi(p[0]) for p in batch) if d]
                if normalized_dois:
                    doi_filter = f"doi:{'|'.join(normalized_dois)}"
                    try:
                        fetched = await client.fetch_all_pages(doi_filter)
                        fetched_dois = {normalize_doi(p.get("doi")) for p in fetched if p.get("doi")}
                        for record in fetched:
                            loader.process_record(record)
                            pid = loader._extract_id(record.get("id"))
                            if pid:
                                con.execute("UPDATE papers SET source = 'openalex' WHERE id = ?", [pid])
                                stats["found"] += 1
                        stats["not_found"] += len(normalized_dois) - len(fetched_dois)
                    except Exception as e:
                        console.print(f"[red]Batch DOI error: {e}[/red]")
                        stats["errors"] += len(batch)
                progress.advance(task, len(batch))

            if doi_batches:
                await asyncio.gather(*(handle_doi_batch(b) for b in doi_batches))
                con.commit()

            # 2. Title-only search — single page (top match) per title, concurrent.
            #    Was fetch_all_pages, which cursor-paginated EVERY search hit just
            #    to take results[0]; broad titles meant dozens of wasted requests.
            async def handle_title(title: str | None) -> None:
                try:
                    clean_title = (title or "").replace('"', '')
                    query = f'display_name.search:"{clean_title}"'
                    data = await client.fetch_page(query, extra_params={"per_page": 1})
                    fetched = (data or {}).get("results", [])
                    if fetched:
                        record = fetched[0]
                        loader.process_record(record)
                        pid = loader._extract_id(record.get("id"))
                        if pid:
                            con.execute("UPDATE papers SET source = 'openalex' WHERE id = ?", [pid])
                            stats["found"] += 1
                        else:
                            stats["not_found"] += 1
                    else:
                        stats["not_found"] += 1
                except Exception:
                    stats["errors"] += 1
                progress.advance(task, 1)

            if title_only_papers:
                await asyncio.gather(*(handle_title(t) for _, t in title_only_papers))
                con.commit()

            con.commit()

    return stats


def _print_summary(stats: Dict[str, int]) -> None:
    table = Table(title="enrich-openalex", show_lines=False)
    table.add_column("Metric", style="white")
    table.add_column("Count", justify="right", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " "), f"{v:,}")
    console.print(table)
