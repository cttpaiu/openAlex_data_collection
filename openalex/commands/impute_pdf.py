"""Command: ``openalex impute pdf`` — recover institution + country from paper PDFs.

Pipeline (per paper): DOI → Unpaywall/arXiv URL → download PDF → first-page
text → langchain structured-output LLM → match authors to ``contributions``
rows → write ``institution_id`` + ``country_code`` per matched row, plus
audit. Per-paper commit makes the run resumable.

This commit (P3) only implements the URL+download+text-extract pass and a
dry-run coverage report; the LLM extraction + DB writes ship in P4.
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
    _StageProgress,
    _ensure_audit_table,
    _resolve_db_path,
)
from openalex.config import load_config
from openalex.pdf import (
    download_pdf,
    extract_first_page_text,
    find_pdf_url,
    text_metrics,
)

console = Console()

DEFAULT_PDF_CACHE = Path("data/cache/pdfs")


def _parse_sources(raw: str) -> tuple[str, ...]:
    items = [s.strip().lower() for s in raw.split(",") if s.strip()]
    allowed = {"unpaywall", "arxiv"}
    bad = [s for s in items if s not in allowed]
    if bad:
        raise click.BadParameter(f"unknown source(s): {bad}; allowed: {sorted(allowed)}")
    return tuple(items) or ("unpaywall", "arxiv")


def _load_eligible_papers(con, limit: int | None) -> list[tuple[str, str]]:
    """Papers with a DOI and at least one contribution row lacking institution_id."""
    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""
    rows = con.execute(
        f"""
        SELECT DISTINCT p.id, p.doi
        FROM papers p
        JOIN contributions c ON c.paper_id = p.id
        WHERE p.doi IS NOT NULL AND TRIM(p.doi) != ''
          AND c.institution_id IS NULL
        {limit_sql}
        """
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


@click.command("pdf")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
@click.option("--dry-run", is_flag=True, help="Preview without DB writes (P3 only supports dry-run)")
@click.option("--limit", type=int, default=None, help="Cap eligible papers")
@click.option("--source", "source_csv", default="unpaywall,arxiv", show_default=True,
              help="Comma-separated PDF source order (unpaywall, arxiv)")
@click.option("--concurrent-downloads", type=int, default=4, show_default=True)
@click.option("--max-pdf-mb", type=int, default=30, show_default=True)
@click.option("--cache-dir", default=str(DEFAULT_PDF_CACHE), show_default=True)
@click.option("--skip-cache", is_flag=True, help="Force re-download even if cached")
@click.option("--request-timeout", type=int, default=60, show_default=True,
              help="HTTP timeout per request (seconds)")
def impute_pdf_command(
    config_path: str,
    db_path: str | None,
    dry_run: bool,
    limit: int | None,
    source_csv: str,
    concurrent_downloads: int,
    max_pdf_mb: int,
    cache_dir: str,
    skip_cache: bool,
    request_timeout: int,
) -> None:
    """Recover affiliations from PDF text (URL coverage report stage — P3).

    This command currently runs the URL resolver, downloads PDFs to a
    content-addressed cache, and extracts page-1 text. The LLM extraction
    and DB writes land in P4.
    """
    cfg = load_config(config_path)
    final_db_path = _resolve_db_path(cfg, db_path)
    if not Path(final_db_path).exists():
        console.print(f"[bold red]✗ Database not found:[/bold red] {final_db_path}")
        raise SystemExit(1)
    if not cfg.email or "@" not in cfg.email:
        console.print(
            "[bold red]✗ Set api.email in config/collection.yml for Unpaywall polite pool.[/bold red]"
        )
        raise SystemExit(1)

    sources = _parse_sources(source_csv)
    cache_path = Path(cache_dir)

    import duckdb

    con = duckdb.connect(final_db_path, read_only=False)
    try:
        _ensure_audit_table(con)
        candidates = _load_eligible_papers(con, limit)
        eligible = len(candidates)
        if eligible == 0:
            console.print("[dim]impute pdf: nothing eligible.[/dim]")
            return

        console.print(
            f"[bold cyan]impute pdf[/bold cyan] — {eligible:,} papers "
            f"(sources={sources}, concurrency={concurrent_downloads}, "
            f"cache={cache_path})"
        )

        stats = asyncio.run(
            _run_pdf_pass(
                candidates=candidates,
                email=cfg.email,
                sources=sources,
                concurrency=concurrent_downloads,
                cache_dir=cache_path,
                max_bytes=max_pdf_mb * 1024 * 1024,
                skip_cache=skip_cache,
                request_timeout=request_timeout,
            )
        )
        _print_summary(stats, dry_run=True)
        if not dry_run:
            console.print(
                "[yellow]Note:[/yellow] P3 only does URL/download/text. "
                "DB writes land in P4."
            )
    finally:
        con.close()


async def _run_pdf_pass(
    candidates: list[tuple[str, str]],
    email: str,
    sources: tuple[str, ...],
    concurrency: int,
    cache_dir: Path,
    max_bytes: int,
    skip_cache: bool,
    request_timeout: int,
) -> dict[str, int]:
    stats: dict[str, int] = {
        "papers_eligible": len(candidates),
        "url_found_unpaywall": 0,
        "url_found_arxiv": 0,
        "url_missing": 0,
        "downloaded": 0,
        "download_cached_hit": 0,
        "download_failed_http": 0,
        "download_failed_network": 0,
        "download_too_large": 0,
        "download_not_pdf": 0,
        "text_extracted": 0,
        "text_empty": 0,
        "chars_total": 0,
    }

    headers = {
        "User-Agent": f"openalex-data-collection/0.1.0 (mailto:{email})",
        "Accept": "application/json, application/pdf, */*",
    }
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    sem = asyncio.Semaphore(max(1, concurrency))

    cache_dir.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(
        headers=headers, timeout=timeout, connector=connector
    ) as session:
        async def _one(paper_id: str, doi: str) -> tuple[str, dict[str, Any]]:
            outcome: dict[str, Any] = {"paper_id": paper_id, "doi": doi}
            async with sem:
                url, src = await find_pdf_url(session, doi, email, sources)
                outcome["url"], outcome["source"] = url, src
                if not url:
                    return paper_id, outcome
                # Check cache hit before downloading.
                from openalex.pdf import _cache_path
                cached = _cache_path(doi, cache_dir)
                cached_before = cached.exists() and not skip_cache
                path, err = await download_pdf(
                    session, url, doi, cache_dir,
                    max_bytes=max_bytes, skip_cache=skip_cache,
                )
                outcome["path"] = str(path) if path else None
                outcome["err"] = err
                outcome["cache_hit"] = cached_before and path is not None and err is None
                if path and not err:
                    text = extract_first_page_text(path)
                    outcome["text_chars"] = len(text)
                    outcome["text_empty"] = not bool(text)
                return paper_id, outcome

        with _StageProgress(
            f"impute pdf — URL+download+text ({len(candidates):,} papers)",
            len(candidates),
            unit="paper",
        ) as bar:
            tasks = [
                asyncio.create_task(_one(pid, doi)) for pid, doi in candidates
            ]
            for fut in asyncio.as_completed(tasks):
                paper_id, outcome = await fut
                src = outcome.get("source")
                if src == "unpaywall":
                    stats["url_found_unpaywall"] += 1
                elif src == "arxiv":
                    stats["url_found_arxiv"] += 1
                else:
                    stats["url_missing"] += 1
                    bar.log(f"{paper_id} → no PDF URL")
                    bar.advance()
                    continue

                err = outcome.get("err")
                if err == "too_large":
                    stats["download_too_large"] += 1
                    bar.log(f"{paper_id} → [yellow]too large[/yellow]")
                    bar.advance(); continue
                if err == "not_pdf":
                    stats["download_not_pdf"] += 1
                    bar.log(f"{paper_id} → [yellow]not PDF[/yellow]")
                    bar.advance(); continue
                if err and err.startswith("http_"):
                    stats["download_failed_http"] += 1
                    bar.log(f"{paper_id} → [red]{err}[/red]")
                    bar.advance(); continue
                if err == "network":
                    stats["download_failed_network"] += 1
                    bar.log(f"{paper_id} → [red]network[/red]")
                    bar.advance(); continue

                if outcome.get("cache_hit"):
                    stats["download_cached_hit"] += 1
                else:
                    stats["downloaded"] += 1

                chars = int(outcome.get("text_chars", 0))
                if outcome.get("text_empty"):
                    stats["text_empty"] += 1
                    bar.log(f"{paper_id} → [yellow]no text layer[/yellow]")
                else:
                    stats["text_extracted"] += 1
                    stats["chars_total"] += chars
                    bar.log(
                        f"{paper_id} → {src}, {chars} chars"
                    )
                bar.advance()

    return stats


def _print_summary(stats: dict[str, int], dry_run: bool) -> None:
    table = Table(title="impute pdf — URL+download+text coverage", show_lines=False)
    table.add_column("Metric", style="white")
    table.add_column("Count", justify="right", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " "), f"{v:,}")
    if stats["text_extracted"]:
        avg = stats["chars_total"] // stats["text_extracted"]
        table.add_row("avg chars per page-1", f"{avg:,}")
    table.add_row("mode", "dry-run (P3)")
    console.print(table)
