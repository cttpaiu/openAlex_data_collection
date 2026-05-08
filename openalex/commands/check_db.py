"""Command: check-db — run completeness checks on a DuckDB database."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.command("check-db")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
def check_db_command(config_path: str, db_path: str) -> None:
    """Run completeness checks on a DuckDB database and display a health report."""
    from openalex.config import load_config
    cfg = load_config(config_path)

    if not db_path:
        import questionary
        default = str(Path(cfg.db_dir) / "quantum_papers.duckdb")
        db_path = questionary.text("Path to database:", default=default).ask() or default

    if not Path(db_path).exists():
        console.print(f"[bold red]✗ Database not found:[/bold red] {db_path}")
        raise SystemExit(1)

    import duckdb
    con = duckdb.connect(db_path, read_only=True)

    try:
        _print_health_report(con, db_path)
    finally:
        con.close()


def _q(con, sql: str, default=0):
    try:
        r = con.execute(sql).fetchone()
        return r[0] if r else default
    except Exception:
        return default


def _print_health_report(con, db_path: str) -> None:
    total = _q(con, "SELECT COUNT(*) FROM papers")
    year_min = _q(con, "SELECT MIN(publication_year) FROM papers", "—")
    year_max = _q(con, "SELECT MAX(publication_year) FROM papers", "—")
    with_abstract = _q(con, "SELECT COUNT(*) FROM papers WHERE abstract_text IS NOT NULL AND abstract_text != ''")
    author_count = _q(con, "SELECT COUNT(*) FROM authors")
    inst_count = _q(con, "SELECT COUNT(*) FROM institutions")
    country_count = _q(con, "SELECT COUNT(*) FROM countries")
    contrib_count = _q(con, "SELECT COUNT(*) FROM contributions")

    # Papers with any country info
    with_country = _q(con, """
        SELECT COUNT(DISTINCT paper_id) FROM contributions WHERE country_code IS NOT NULL
    """)

    # Tier classification
    tier_sql = """
        SELECT
            SUM(CASE WHEN has_country AND has_inst THEN 1 ELSE 0 END),
            SUM(CASE WHEN (has_country OR has_inst) AND NOT (has_country AND has_inst) THEN 1 ELSE 0 END),
            SUM(CASE WHEN NOT has_country AND NOT has_inst THEN 1 ELSE 0 END)
        FROM (
            SELECT p.id,
                MAX(c.country_code IS NOT NULL)::BOOLEAN as has_country,
                MAX(c.institution_id IS NOT NULL)::BOOLEAN as has_inst
            FROM papers p
            LEFT JOIN contributions c ON p.id = c.paper_id
            GROUP BY p.id
        ) sub
    """
    try:
        tier_row = con.execute(tier_sql).fetchone()
        tier1 = tier_row[0] or 0
        tier2 = tier_row[1] or 0
        tier3 = tier_row[2] or 0
    except Exception:
        tier1 = tier2 = tier3 = 0

    def pct(n):
        return f"{n / total * 100:.1f}%" if total else "-"

    table = Table(title=f"Database Health Report — {db_path}", show_lines=False)
    table.add_column("Metric", style="white", min_width=35)
    table.add_column("Count", justify="right", style="green")
    table.add_column("%", justify="right", style="dim")

    table.add_row("Total papers", f"{total:,}", "100%")
    table.add_row("Year range", f"{year_min} – {year_max}", "")
    table.add_row("Authors", f"{author_count:,}", "")
    table.add_row("Institutions", f"{inst_count:,}", "")
    table.add_row("Countries", f"{country_count:,}", "")
    table.add_row("Contributions (authorships)", f"{contrib_count:,}", "")
    table.add_section()
    table.add_row("Papers with abstracts", f"{with_abstract:,}", pct(with_abstract))
    table.add_row("Papers with country info", f"{with_country:,}", pct(with_country))
    table.add_section()
    table.add_row("[bold]Tier 1[/bold] — Complete metadata", f"{tier1:,}", pct(tier1))
    table.add_row("Tier 2 — Partial metadata", f"{tier2:,}", pct(tier2))
    table.add_row("[dim]Tier 3 — Ghost (no location)[/dim]", f"{tier3:,}", pct(tier3))

    # Country imputation readiness
    missing_country = _q(con, "SELECT COUNT(*) FROM contributions WHERE country_code IS NULL")
    imputable = _q(con, """
        SELECT COUNT(*) FROM contributions
        WHERE country_code IS NULL
          AND raw_affiliation_string IS NOT NULL
          AND TRIM(raw_affiliation_string) != ''
    """)
    dead = _q(con, """
        SELECT COUNT(*) FROM contributions
        WHERE country_code IS NULL
          AND (raw_affiliation_string IS NULL OR TRIM(raw_affiliation_string) = '')
    """)
    imputable_papers = _q(con, """
        SELECT COUNT(DISTINCT paper_id) FROM contributions
        WHERE country_code IS NULL
          AND raw_affiliation_string IS NOT NULL
          AND TRIM(raw_affiliation_string) != ''
    """)

    def pct_of(n, d):
        return f"{n / d * 100:.1f}%" if d else "-"

    table.add_section()
    table.add_row("Contributions missing country_code", f"{missing_country:,}", "")
    table.add_row("  → have raw_affiliation (imputable)", f"{imputable:,}", pct_of(imputable, missing_country))
    table.add_row("  → no raw_affiliation (dead end)", f"{dead:,}", pct_of(dead, missing_country))
    table.add_row("Distinct papers imputable", f"{imputable_papers:,}", "")

    console.print(table)

    if imputable > 0:
        console.print(
            f"\n[bold yellow]→ {imputable:,} rows can be imputed.[/bold yellow] "
            "Run: [cyan]uv run openalex impute-country --dry-run[/cyan]"
        )

    # Top 5 topics
    try:
        top_topics = con.execute("""
            SELECT primary_topic_name, COUNT(*) as n
            FROM papers
            WHERE primary_topic_name IS NOT NULL
            GROUP BY primary_topic_name ORDER BY n DESC LIMIT 5
        """).fetchall()
        if top_topics:
            console.print("\n[bold]Top 5 topics:[/bold]")
            for name, n in top_topics:
                console.print(f"  [cyan]{name[:60]}[/cyan]  [green]{n:,}[/green]")
    except Exception:
        pass
