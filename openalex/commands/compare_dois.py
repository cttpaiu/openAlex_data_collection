"""Command: compare-dois — overlap analysis between DuckDB papers.doi and a WoS CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Set

import click
from rich.console import Console
from rich.table import Table

from openalex.config import load_config
from openalex.wos import normalize_doi

console = Console()


def _prompt_path(message: str, default: str | None = None) -> str:
    import questionary

    answer = questionary.text(message, default=default or "").ask()
    return (answer or "").strip()


def _db_dois(db_path: str) -> Set[str]:
    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT DISTINCT doi FROM papers WHERE doi IS NOT NULL AND doi != ''"
        ).fetchall()
    finally:
        con.close()
    return {n for r in rows if (n := normalize_doi(r[0])) is not None}


def _wos_csv_dois(csv_path: str) -> tuple[Set[str], "object"]:
    import polars as pl

    df = pl.read_csv(
        csv_path,
        null_values=["n/a", "N/A", "na", "null", ""],
        infer_schema_length=10000,
    )
    if "DOI" not in df.columns:
        raise click.ClickException(f"CSV missing required 'DOI' column. Found: {df.columns}")
    dois = {
        n for raw in df["DOI"].drop_nulls().to_list()
        if (n := normalize_doi(str(raw))) is not None
    }
    return dois, df


def _render(common: Set[str], only_db: Set[str], only_wos: Set[str], n_db: int, n_wos: int) -> None:
    def pct(n: int, d: int) -> str:
        return f"{n / d * 100:.1f}%" if d else "-"

    table = Table(title="DOI Overlap: DuckDB vs Web of Science CSV", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("% of WoS", justify="right", style="dim")

    table.add_row("DOIs in DuckDB (normalized)", f"{n_db:,}", "")
    table.add_row("DOIs in WoS CSV (normalized)", f"{n_wos:,}", "100%")
    table.add_section()
    table.add_row("Common to both", f"{len(common):,}", pct(len(common), n_wos))
    table.add_row("Only in DuckDB", f"{len(only_db):,}", "")
    table.add_row("Only in WoS CSV", f"{len(only_wos):,}", pct(len(only_wos), n_wos))
    console.print(table)


def _show_samples(only_wos: Set[str], only_db: Set[str], wos_df) -> None:
    import polars as pl

    if only_wos:
        console.print("\n[bold]Sample DOIs only in WoS (missing from DB):[/bold]")
        sample = (
            wos_df.with_columns(
                pl.col("DOI")
                .cast(pl.String)
                .map_elements(normalize_doi, return_dtype=pl.String)
                .alias("__doi_norm__")
            )
            .filter(pl.col("__doi_norm__").is_in(list(only_wos)))
            .select([c for c in ("DOI", "Article Title") if c in wos_df.columns])
            .head(3)
        )
        for row in sample.iter_rows(named=True):
            title = str(row.get("Article Title", ""))[:80]
            console.print(f"  • [yellow]{normalize_doi(row['DOI'])}[/yellow]\n    → {title}")

    if only_db:
        console.print(f"\n[bold]Sample DOIs only in DuckDB:[/bold] {len(only_db):,} total")
        for doi in list(only_db)[:3]:
            console.print(f"  • [yellow]{doi}[/yellow]")


def _export(only_wos: Set[str], only_db: Set[str], wos_df, out_dir: Path) -> None:
    import polars as pl

    out_dir.mkdir(parents=True, exist_ok=True)

    if only_wos:
        wos_path = out_dir / "wos_dois_missing_from_db.csv"
        missing = (
            wos_df.with_columns(
                pl.col("DOI")
                .cast(pl.String)
                .map_elements(normalize_doi, return_dtype=pl.String)
                .alias("__doi_norm__")
            )
            .filter(pl.col("__doi_norm__").is_in(list(only_wos)))
            .drop("__doi_norm__")
        )
        missing.write_csv(wos_path)
        console.print(f"[green]✓ Exported {len(missing):,} rows → {wos_path}[/green]")

    if only_db:
        db_path = out_dir / "db_dois_missing_from_wos.csv"
        pl.DataFrame({"doi": sorted(only_db)}).write_csv(db_path)
        console.print(f"[green]✓ Exported {len(only_db):,} DOIs → {db_path}[/green]")


@click.command("compare-dois")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file (asked interactively if omitted)")
@click.option("--wos-csv", "wos_csv", default=None, help="Path to WoS CSV (asked interactively if omitted)")
@click.option("--export-dir", default=None, help="If set, write unmatched DOI CSVs here")
@click.option("--no-samples", is_flag=True, help="Skip printing example DOIs")
def compare_dois_command(
    config_path: str,
    db_path: str | None,
    wos_csv: str | None,
    export_dir: str | None,
    no_samples: bool,
) -> None:
    """Compare DOIs between the DuckDB papers table and a Web of Science CSV export."""
    cfg = load_config(config_path)

    if not db_path:
        default_db = str(Path(cfg.db_dir) / "quantum_papers.duckdb")
        db_path = _prompt_path("Path to DuckDB file:", default=default_db)
    if not Path(db_path).exists():
        raise click.ClickException(f"Database not found: {db_path}")

    if not wos_csv:
        wos_csv = _prompt_path("Path to WoS CSV:", default="data/")
    if not Path(wos_csv).exists():
        raise click.ClickException(f"WoS CSV not found: {wos_csv}")

    console.print(f"[dim]Reading DOIs from DuckDB: {db_path}[/dim]")
    db_dois = _db_dois(db_path)
    console.print(f"[dim]Reading DOIs from CSV:    {wos_csv}[/dim]")
    wos_dois, wos_df = _wos_csv_dois(wos_csv)

    common = db_dois & wos_dois
    only_db = db_dois - wos_dois
    only_wos = wos_dois - db_dois

    _render(common, only_db, only_wos, len(db_dois), len(wos_dois))

    if not no_samples:
        _show_samples(only_wos, only_db, wos_df)

    if export_dir:
        _export(only_wos, only_db, wos_df, Path(export_dir))
