"""Command: import-wos-csv — import records from a WoS CSV file."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Set

import click
import polars as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from openalex.api_client import AsyncOpenAlexClient
from openalex.commands.database import OpenAlexLoader
from openalex.commands.impute_affiliation import _resolve_db_path
from openalex.config import load_config
from openalex.wos import normalize_doi

console = Console()


def _get_existing_dois(con) -> Set[str]:
    """Retrieve all existing normalized DOIs from the database."""
    rows = con.execute("SELECT doi FROM papers WHERE doi IS NOT NULL").fetchall()
    return {normalize_doi(r[0]) for r in rows if r[0]}


def _insert_stub_record(con, row: Dict[str, Any]) -> None:
    """Insert a minimal paper record from WoS metadata when OpenAlex fetch fails."""
    doi = row.get("DOI")
    title = row.get("Article Title") or "Untitled"
    authors = row.get("Authors")
    source = row.get("Source")
    year = row.get("Publication Date")
    cited = row.get("Times Cited")
    doc_type = row.get("Document Type")
    accession = row.get("Accession Number")

    # Generate a stable ID for WoS-only papers if no DOI
    if accession:
        paper_id = f"WOS_{accession.replace('WOS:', '').strip()}"
    else:
        import hashlib
        paper_id = f"WOS_T{hashlib.sha1(title.lower().encode()).hexdigest()[:12]}"

    con.execute(
        """
        INSERT INTO papers (
            id, doi, title, publication_year, type, journal_name,
            cited_by_count, from_wos, wos_accession_number
        ) VALUES (?, ?, ?, ?, ?, ?, ?, TRUE, ?)
        ON CONFLICT DO NOTHING
        """,
        [paper_id, doi, title, year, doc_type, source, cited, accession]
    )

    # If authors are provided, we could potentially add them to authors/contributions,
    # but the user said "only the DOI title and abstract" and "impute the data",
    # implying OpenAlex is the primary source for authors.
    # For a stub, we might just leave authors out of the relational tables for now
    # to avoid creating duplicate author entries without IDs.


async def _import_csv(config_path: str, csv_path: str, final_db_path: str, limit: int | None, dry_run: bool) -> None:
    cfg = load_config(config_path)

    if not Path(csv_path).exists():
        console.print(f"[bold red]✗ CSV file not found:[/bold red] {csv_path}")
        return

    # Prepare OpenAlex loader (and ensure schema exists)
    loader = OpenAlexLoader(final_db_path, csv_path)
    loader.load_existing_caches()

    con = loader.con
    existing_dois = _get_existing_dois(con)

    df = pl.read_csv(csv_path, null_values="n/a", infer_schema_length=10000)
    if limit:
        df = df.head(limit)

    new_records = []
    for row in df.iter_rows(named=True):
        doi = normalize_doi(row.get("DOI"))
        if not doi or doi not in existing_dois:
            new_records.append(row)

    if not new_records:
        console.print("[green]No new records to import.[/green]")
        loader.close()
        return

    console.print(f"Found [bold]{len(new_records)}[/bold] new records in CSV.")

    if dry_run:
        console.print("[yellow]Dry run: exiting without fetching or inserting.[/yellow]")
        loader.close()
        return

    # Ensure papers has from_wos column
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS from_wos BOOLEAN DEFAULT FALSE")
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS wos_accession_number VARCHAR")

    async with AsyncOpenAlexClient(
        api_keys=cfg.api_keys,
        email=cfg.email,
        concurrent_requests=cfg.concurrent_requests,
    ) as client:

        # Batch DOIs
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
            task = progress.add_task("Importing records...", total=len(new_records))

            for batch in batches:
                batch_dois = [normalize_doi(r.get("DOI")) for r in batch if normalize_doi(r.get("DOI"))]
                fetched_papers = []
                if batch_dois:
                    doi_filter = f"doi:{'|'.join(batch_dois)}"
                    fetched_papers = await client.fetch_all_pages(doi_filter)

                fetched_dois = {normalize_doi(p.get("doi")) for p in fetched_papers if p.get("doi")}

                # Insert fetched papers
                for paper in fetched_papers:
                    loader.process_record(paper)
                    # Mark as from WoS
                    pid = loader._extract_id(paper.get("id"))
                    if pid:
                        con.execute("UPDATE papers SET from_wos = TRUE WHERE id = ?", [pid])

                # Handle papers not found in OpenAlex
                for row in batch:
                    doi = normalize_doi(row.get("DOI"))
                    if not doi or doi not in fetched_dois:
                        _insert_stub_record(con, row)

                progress.advance(task, len(batch))

    loader.close()
    console.print(f"[bold green]✓ Import complete.[/bold green]")


@click.command("import-wos-csv")
@click.argument("input_csv", type=click.Path(exists=True))
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
@click.option("--limit", type=int, default=None, help="Limit number of rows to process")
@click.option("--dry-run", is_flag=True, help="Preview new records without inserting")
def import_wos_csv_command(input_csv: str, config_path: str, db_path: str | None, limit: int | None, dry_run: bool) -> None:
    """Import Web of Science records from a CSV file, fetching full data from OpenAlex."""
    cfg = load_config(config_path)
    final_db_path = _resolve_db_path(cfg, db_path)
    asyncio.run(_import_csv(config_path, input_csv, final_db_path, limit, dry_run))
