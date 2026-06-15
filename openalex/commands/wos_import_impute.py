"""Command: wos-import-impute — single-command WoS CSV ingest + OpenAlex enrich + impute.

Flow:

1. Interactive (or `--wos-csv` / `--db`) prompts for CSV + DuckDB paths.
2. Skip DOIs already present in `papers`.
3. Batch-fetch new DOIs from OpenAlex (50/batch) using `AsyncOpenAlexClient.fetch_all_pages`.
4. For each fetched record: `OpenAlexLoader.process_record` (writes paper, authors,
   institutions, contributions, abstract); tag `papers.source = 'openalex'`.
5. For each batch row missing from the OpenAlex response: insert a WoS stub row
   (title + DOI + accession + journal); tag `papers.source = 'web_of_science'`.
6. Unless `--no-impute`: invoke `impute-affiliation` in-process so newly imported
   contributions get institution_id / country_code filled by the existing pipeline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Set

import click
import polars as pl
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
from openalex.commands.impute_affiliation import impute_affiliation_command
from openalex.config import load_config
from openalex.wos import normalize_doi

console = Console()


def _prompt_path(message: str, default: str) -> str:
    import questionary

    answer = questionary.text(message, default=default).ask()
    return (answer or default).strip()


def _ensure_source_column(con) -> None:
    """Add `source` column and backfill existing rows to 'openalex'."""
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS source VARCHAR")
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS from_wos BOOLEAN DEFAULT FALSE")
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS wos_accession_number VARCHAR")
    con.execute("UPDATE papers SET source = 'openalex' WHERE source IS NULL")


def _existing_dois(con) -> Set[str]:
    rows = con.execute("SELECT doi FROM papers WHERE doi IS NOT NULL").fetchall()
    return {n for r in rows if (n := normalize_doi(r[0])) is not None}


def _insert_wos_stub(con, row: Dict[str, Any]) -> None:
    """Insert a minimal paper row from WoS metadata. Tags source='web_of_science'."""
    doi = normalize_doi(row.get("DOI"))
    title = row.get("Article Title") or "Untitled"
    source_journal = row.get("Source")
    year = row.get("Publication Date")
    cited = row.get("Times Cited")
    doc_type = row.get("Document Type")
    accession = row.get("Accession Number")

    if accession:
        paper_id = f"WOS_{str(accession).replace('WOS:', '').strip()}"
    else:
        import hashlib

        paper_id = f"WOS_T{hashlib.sha1(str(title).lower().encode()).hexdigest()[:12]}"

    con.execute(
        """
        INSERT INTO papers (
            id, doi, title, publication_year, type, journal_name,
            cited_by_count, source, from_wos, wos_accession_number
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'web_of_science', TRUE, ?)
        ON CONFLICT DO NOTHING
        """,
        [paper_id, doi, title, year, doc_type, source_journal, cited, accession],
    )


async def _ingest(
    cfg,
    csv_path: str,
    db_path: str,
    limit: int | None,
    dry_run: bool,
    concurrency: int | None = None,
) -> Dict[str, int]:
    loader = OpenAlexLoader(db_path, csv_path)
    loader.load_existing_caches()
    con = loader.con

    _ensure_source_column(con)

    existing = _existing_dois(con)
    console.print(f"[dim]Existing DOIs in DB: {len(existing):,}[/dim]")

    df = pl.read_csv(csv_path, null_values=["n/a", "N/A", "na", "null", ""], infer_schema_length=10000)
    if "DOI" not in df.columns:
        loader.close()
        raise click.ClickException(f"CSV missing 'DOI' column. Found: {df.columns}")
    if limit:
        df = df.head(limit)

    new_records: List[Dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        doi = normalize_doi(row.get("DOI"))
        if not doi or doi not in existing:
            new_records.append(row)

    stats: Dict[str, Any] = {
        "candidates": len(new_records),
        "with_doi": 0,            # CSV rows that had a normalisable DOI
        "found_in_openalex": 0,   # rows whose DOI was returned by OpenAlex
        "not_found_in_openalex": 0,  # rows with a DOI but no OpenAlex match
        "no_doi": 0,              # rows without any DOI
        "openalex_records": 0,    # papers ingested via loader.process_record
        "wos_stubs": 0,           # stub rows written (= not_found + no_doi)
        "aborted": False,
        "abort_reason": None,
    }

    if not new_records:
        console.print("[green]No new records to import.[/green]")
        loader.close()
        return stats

    console.print(f"Found [bold]{stats['candidates']:,}[/bold] new records to import.")

    if dry_run:
        console.print("[yellow]Dry run — exiting without fetching or inserting.[/yellow]")
        loader.close()
        return stats

    effective_concurrency = concurrency if concurrency and concurrency > 0 else cfg.concurrent_requests
    console.print(f"[dim]Fetching with {effective_concurrency}-way concurrency, batch_size=50.[/dim]")

    try:
        async with AsyncOpenAlexClient(
            api_keys=cfg.api_keys,
            email=cfg.email,
            concurrent_requests=effective_concurrency,
        ) as client:
            batch_size = 50
            batches = [new_records[i:i + batch_size] for i in range(0, len(new_records), batch_size)]

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[green]{task.completed:,}[/green]/[dim]{task.total:,}[/dim]"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                bar = progress.add_task("Fetching OpenAlex by DOI...", total=len(new_records))

                # DuckDB connection is single-threaded — serialize all writes through
                # this lock. Fetches still run concurrently via the client's internal
                # semaphore (cfg.concurrent_requests).
                db_lock = asyncio.Lock()
                # Set on first fatal upstream error (budget exhausted, all keys
                # rejected). Other in-flight tasks check this and short-circuit
                # so we land cleanly in the loader.close() path.
                stop_event = asyncio.Event()

                async def handle_batch(batch: List[Dict[str, Any]]) -> None:
                    if stop_event.is_set():
                        return

                    batch_dois = [
                        d for r in batch
                        if (d := normalize_doi(r.get("DOI"))) is not None
                    ]
                    fetched_papers: list = []
                    if batch_dois:
                        doi_filter = f"doi:{'|'.join(batch_dois)}"
                        try:
                            fetched_papers = await client.fetch_all_pages(doi_filter)
                        except (PermissionError, RuntimeError) as e:
                            # Budget exhausted / all keys rejected / non-JSON 4xx.
                            # Record reason once, signal siblings to stop, return
                            # so already-fetched batches still drain their writes.
                            if not stop_event.is_set():
                                stats["aborted"] = True
                                stats["abort_reason"] = str(e)
                                stop_event.set()
                            return

                    fetched_dois = {
                        normalize_doi(p.get("doi")) for p in fetched_papers if p.get("doi")
                    }

                    async with db_lock:
                        for paper in fetched_papers:
                            loader.process_record(paper)
                            pid = loader._extract_id(paper.get("id"))
                            if pid:
                                con.execute(
                                    "UPDATE papers SET source = 'openalex', from_wos = TRUE WHERE id = ?",
                                    [pid],
                                )
                                stats["openalex_records"] += 1

                        for row in batch:
                            doi = normalize_doi(row.get("DOI"))
                            if not doi:
                                stats["no_doi"] += 1
                                _insert_wos_stub(con, row)
                                stats["wos_stubs"] += 1
                            else:
                                stats["with_doi"] += 1
                                if doi in fetched_dois:
                                    stats["found_in_openalex"] += 1
                                else:
                                    stats["not_found_in_openalex"] += 1
                                    _insert_wos_stub(con, row)
                                    stats["wos_stubs"] += 1

                        progress.advance(bar, len(batch))

                await asyncio.gather(*(handle_batch(b) for b in batches))
    except KeyboardInterrupt:
        stats["aborted"] = True
        stats["abort_reason"] = "KeyboardInterrupt (Ctrl+C)"
    finally:
        # Always close the loader so DuckDB flushes anything still buffered.
        # This is what makes resume work after a crash / Ctrl+C.
        loader.close()

    return stats


def _print_summary(stats: Dict[str, Any]) -> None:
    candidates = stats.get("candidates", 0)
    found = stats.get("found_in_openalex", 0)
    not_found = stats.get("not_found_in_openalex", 0)
    no_doi = stats.get("no_doi", 0)
    with_doi = stats.get("with_doi", 0)
    openalex_records = stats.get("openalex_records", 0)
    wos_stubs = stats.get("wos_stubs", not_found + no_doi)

    pct_overall = (found / candidates * 100) if candidates else 0.0
    pct_with_doi = (found / with_doi * 100) if with_doi else 0.0

    table = Table(title="WoS Import Summary", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Note", style="dim")

    table.add_row("New candidate rows", f"{candidates:,}", "")
    table.add_row(
        "  ✓ Found in OpenAlex",
        f"{found:,}",
        f"{pct_overall:.1f}% of all  /  {pct_with_doi:.1f}% of rows with DOI",
    )
    table.add_row(
        "  ✗ DOI not in OpenAlex",
        f"{not_found:,}",
        "stubbed from WoS metadata",
    )
    table.add_row(
        "  – No DOI in CSV row",
        f"{no_doi:,}",
        "stubbed from WoS metadata",
    )
    table.add_row("WoS stubs written", f"{wos_stubs:,}", "")
    table.add_row(
        "OpenAlex records inserted",
        f"{openalex_records:,}",
        "papers + authors + contributions",
    )
    if stats.get("aborted"):
        table.add_row("Status", "[bold yellow]ABORTED[/bold yellow]", "")
    console.print(table)

    snap = AsyncOpenAlexClient.pool_snapshot()
    if snap["total"]:
        usage_str = ", ".join(
            f"#{i+1}:{u:,}" for i, u in enumerate(snap["usage"])
        )
        console.print(
            f"[dim]API keys — active:{snap['active']} "
            f"exhausted:{snap['exhausted']} rejected:{snap['rejected']} | "
            f"requests served: {usage_str}[/dim]"
        )

    if stats.get("aborted"):
        reason = stats.get("abort_reason") or "unknown"
        console.print(
            f"\n[bold yellow]⚠ Run aborted: {reason}[/bold yellow]"
        )
        console.print(
            "[dim]Progress was saved to the DuckDB file. "
            "Re-run the same command with the same CSV + DB to resume — "
            "already-imported DOIs are skipped automatically.[/dim]"
        )


@click.command("wos-import-impute")
@click.option("--wos-csv", "wos_csv", default=None, help="Path to WoS CSV (prompts if omitted)")
@click.option("--db", "db_path", default=None, help="Path to DuckDB file (prompts if omitted)")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--limit", type=int, default=None, help="Limit rows to process")
@click.option("--concurrency", type=int, default=None, help="Override concurrent OpenAlex requests (default: cfg.concurrent_requests)")
@click.option("--dry-run", is_flag=True, help="Preview only — no inserts, no imputation")
@click.option("--no-impute", is_flag=True, help="Skip the chained impute-affiliation pass")
@click.option("--llm-provider", default=None, help="LLM provider for impute step (groq|ollama)")
@click.option("--groq-api-key", default=None, help="Groq API key for impute step")
@click.option("--stages", default="1,2,3", show_default=True, help="Imputation stages to run")
@click.pass_context
def wos_import_impute_command(
    ctx: click.Context,
    wos_csv: str | None,
    db_path: str | None,
    config_path: str,
    limit: int | None,
    concurrency: int | None,
    dry_run: bool,
    no_impute: bool,
    llm_provider: str | None,
    groq_api_key: str | None,
    stages: str,
) -> None:
    """One-shot: import a WoS CSV, enrich each new DOI from OpenAlex, then run impute-affiliation."""
    cfg = load_config(config_path)

    if not wos_csv:
        wos_csv = _prompt_path("Path to WoS CSV:", default="data/")
    if not Path(wos_csv).exists():
        raise click.ClickException(f"WoS CSV not found: {wos_csv}")

    if not db_path:
        default_db = str(Path(cfg.db_dir) / "quantum_papers.duckdb")
        db_path = _prompt_path("Path to DuckDB file:", default=default_db)

    console.print(f"[dim]CSV: {wos_csv}[/dim]")
    console.print(f"[dim]DB:  {db_path}[/dim]")

    stats = asyncio.run(_ingest(cfg, wos_csv, db_path, limit, dry_run, concurrency))
    _print_summary(stats)

    if stats.get("aborted"):
        # Don't chain impute on a partial run — same budget/keys would just fail
        # again. User re-runs the import command first to resume.
        console.print("[yellow]Skipped impute-affiliation because import was aborted.[/yellow]")
        return

    if dry_run or no_impute:
        if no_impute and not dry_run:
            console.print("[yellow]Skipped impute-affiliation (--no-impute).[/yellow]")
        return

    if stats.get("openalex_records", 0) == 0:
        console.print("[yellow]No OpenAlex rows added — skipping impute-affiliation.[/yellow]")
        return

    console.print("\n[bold cyan]Running impute-affiliation against same DB...[/bold cyan]")
    ctx.invoke(
        impute_affiliation_command,
        config_path=config_path,
        db_path=db_path,
        dry_run=False,
        limit=None,
        stages=stages,
        llm_fallback=True,
        groq_api_key=groq_api_key,
        llm_provider=llm_provider,
        llm_model=None,
        llm_base_url=None,
        llm_batch_size=20,
        llm_concurrency=2,
        llm_min_confidence=0.8,
        llm_max_tokens_stage1=1200,
        llm_max_tokens_stage2=400,
        llm_max_tokens_stage3=300,
        match_threshold=0.78,
    )
