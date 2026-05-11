"""Command: check-db — run completeness checks on a DuckDB database."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
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

    _add_paper_coverage_rows(con, table, total, pct)

    console.print(table)

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


def _add_paper_coverage_rows(con, table, total: int, pct) -> None:
    orphan = _q(con, """
        SELECT COUNT(*) FROM papers p
        WHERE NOT EXISTS (SELECT 1 FROM contributions c WHERE c.paper_id = p.id)
    """)
    all_null = _q(con, """
        SELECT COUNT(*) FROM papers p
        WHERE EXISTS (SELECT 1 FROM contributions c WHERE c.paper_id = p.id)
          AND NOT EXISTS (
              SELECT 1 FROM contributions c
              WHERE c.paper_id = p.id AND c.institution_id IS NOT NULL
          )
    """)
    zero_total = orphan + all_null
    contribs_for_zero = _q(con, """
        SELECT COUNT(*) FROM contributions c
        WHERE NOT EXISTS (
            SELECT 1 FROM contributions c2
            WHERE c2.paper_id = c.paper_id AND c2.institution_id IS NOT NULL
        )
    """)

    table.add_section()
    table.add_row("[bold]Papers with zero institution connection[/bold]", f"{zero_total:,}", pct(zero_total))
    table.add_row("  → orphan (no contribution rows)", f"{orphan:,}", pct(orphan))
    table.add_row("  → all contributions missing institution_id", f"{all_null:,}", pct(all_null))
    table.add_row("  → contribution rows for those papers (expanded)", f"{contribs_for_zero:,}", "")

    buckets = _partial_coverage_buckets(con)
    partial_total = sum(buckets.values())
    table.add_row("[bold]Papers with partial institution coverage[/bold]", f"{partial_total:,}", pct(partial_total))
    for k in (1, 2, 3, 4, 5):
        label = "5+ missing" if k == 5 else f"{k} missing"
        table.add_row(f"  → {label}", f"{buckets.get(k, 0):,}", "")


def _partial_coverage_buckets(con) -> dict[int, int]:
    """Return {bucket: paper_count} where bucket 5 collapses 5+ missing."""
    try:
        rows = con.execute("""
            WITH paper_stats AS (
                SELECT paper_id,
                    SUM(CASE WHEN institution_id IS NULL THEN 1 ELSE 0 END) AS missing,
                    SUM(CASE WHEN institution_id IS NOT NULL THEN 1 ELSE 0 END) AS filled
                FROM contributions
                GROUP BY paper_id
            )
            SELECT
                CASE WHEN missing >= 5 THEN 5 ELSE missing END AS bucket,
                COUNT(*) AS papers
            FROM paper_stats
            WHERE missing > 0 AND filled > 0
            GROUP BY bucket
            ORDER BY bucket
        """).fetchall()
        return {int(b): int(n) for b, n in rows}
    except Exception:
        return {}
