"""Command: impute-coauthor — fill missing institution_id via author dominance.

For every contribution row with ``institution_id IS NULL`` and a known
author_id, look up that author's most-used institution across *other*
papers in the DB. If the dominant institution covers at least
``--threshold`` of the author's other appearances, apply it to the
NULL row. Country_code cascades from the matched institution.

Pure SQL pass — no LLM, no network. Designed to run first, before
``impute-affiliation``, since it's free and idempotent.

The same idempotent ``COALESCE`` UPDATE pattern used by impute-affiliation
applies here: re-running the command only touches still-NULL rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from openalex.commands.impute_affiliation import (
    _ensure_audit_table,
    _ensure_country,
    _insert_audit,
    _resolve_db_path,
)
from openalex.config import load_config
from openalex.imputation import normalize_country_code

console = Console()


@click.command("impute-coauthor")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
@click.option("--dry-run", is_flag=True, help="Show what would be updated without writing changes")
@click.option(
    "--threshold", type=float, default=0.6, show_default=True,
    help="Min fraction of an author's other contributions sharing the same institution",
)
@click.option(
    "--min-other-papers", type=int, default=2, show_default=True,
    help="Author must have at least this many other contributions with a known institution",
)
@click.option("--limit", type=int, default=None, help="Cap eligible rows (debug)")
def impute_coauthor_command(
    config_path: str,
    db_path: str | None,
    dry_run: bool,
    threshold: float,
    min_other_papers: int,
    limit: int | None,
) -> None:
    """Impute missing institution_id from each author's dominant other-paper institution."""
    cfg = load_config(config_path)
    final_db_path = _resolve_db_path(cfg, db_path)
    if not Path(final_db_path).exists():
        console.print(f"[bold red]✗ Database not found:[/bold red] {final_db_path}")
        raise SystemExit(1)

    import duckdb

    con = duckdb.connect(final_db_path, read_only=False)
    try:
        _ensure_audit_table(con)
        _run_coauthor_imputation(con, dry_run, threshold, min_other_papers, limit)
    finally:
        con.close()


def _run_coauthor_imputation(
    con,
    dry_run: bool,
    threshold: float,
    min_other_papers: int,
    limit: int | None,
) -> None:
    console.print(
        f"[bold cyan]Coauthor imputation[/bold cyan] "
        f"(threshold={threshold}, min_other_papers={min_other_papers})"
    )

    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""
    candidates = con.execute(
        f"""
        WITH author_inst_freq AS (
            SELECT author_id, institution_id, COUNT(*) AS n
            FROM contributions
            WHERE author_id IS NOT NULL
              AND institution_id IS NOT NULL
            GROUP BY author_id, institution_id
        ),
        author_total AS (
            SELECT author_id, SUM(n) AS total
            FROM author_inst_freq
            GROUP BY author_id
        ),
        author_top AS (
            SELECT
                f.author_id,
                f.institution_id,
                f.n,
                t.total,
                CAST(f.n AS DOUBLE) / NULLIF(t.total, 0) AS confidence,
                ROW_NUMBER() OVER (PARTITION BY f.author_id ORDER BY f.n DESC) AS rnk
            FROM author_inst_freq f
            JOIN author_total t USING (author_id)
        )
        SELECT
            c.row_id,
            c.paper_id,
            c.author_id,
            a.institution_id,
            a.confidence,
            a.n,
            a.total,
            i.country_code,
            i.display_name
        FROM contributions c
        JOIN author_top a ON c.author_id = a.author_id AND a.rnk = 1
        LEFT JOIN institutions i ON i.id = a.institution_id
        WHERE c.institution_id IS NULL
          AND c.author_id IS NOT NULL
          AND a.total >= ?
          AND a.confidence >= ?
        {limit_sql}
        """,
        [int(min_other_papers), float(threshold)],
    ).fetchall()

    eligible_null_authored = _q(con, """
        SELECT COUNT(*) FROM contributions
        WHERE institution_id IS NULL AND author_id IS NOT NULL
    """)

    stats = {
        "null_with_author": eligible_null_authored,
        "candidates_meeting_threshold": len(candidates),
        "applied": 0,
        "papers_recovered": 0,
        "papers_partial_recovered": 0,
    }

    if not candidates:
        console.print("[dim]Coauthor: no candidates meeting threshold.[/dim]")
        _print_summary(stats, dry_run)
        return

    paper_ids_before = _zero_inst_paper_ids(con)

    updates: list[dict[str, Any]] = []
    for row_id, paper_id, _author_id, inst_id, confidence, n, total, country_code, display_name in candidates:
        cc = normalize_country_code(country_code)
        updates.append({
            "row_id": int(row_id),
            "paper_id": paper_id,
            "institution_id": inst_id,
            "country_code": cc,
            "raw_affiliation_string": None,
            "matched_terms": (
                f"coauthor: {display_name or inst_id} "
                f"({n}/{total} = {float(confidence):.2f})"
            ),
            "source": "coauthor",
            "confidence": float(confidence),
        })

    if not dry_run:
        for u in updates:
            cc = _ensure_country(con, u["country_code"]) if u["country_code"] else None
            con.execute(
                """
                UPDATE contributions
                SET institution_id = COALESCE(institution_id, ?),
                    country_code = COALESCE(country_code, ?)
                WHERE row_id = ?
                """,
                [u["institution_id"], cc, u["row_id"]],
            )
        _insert_audit(con, updates, stage="coauthor")
        con.commit()
        stats["applied"] = len(updates)

        paper_ids_after = _zero_inst_paper_ids(con)
        recovered = paper_ids_before - paper_ids_after
        stats["papers_recovered"] = len(recovered)
        stats["papers_partial_recovered"] = (
            len({u["paper_id"] for u in updates}) - len(recovered)
        )
    else:
        stats["applied"] = len(updates)

    _print_summary(stats, dry_run)


def _zero_inst_paper_ids(con) -> set[str]:
    rows = con.execute(
        """
        SELECT p.id FROM papers p
        WHERE NOT EXISTS (
            SELECT 1 FROM contributions c
            WHERE c.paper_id = p.id AND c.institution_id IS NOT NULL
        )
        """
    ).fetchall()
    return {r[0] for r in rows}


def _q(con, sql: str, default=0):
    try:
        r = con.execute(sql).fetchone()
        return r[0] if r else default
    except Exception:
        return default


def _print_summary(stats: dict[str, Any], dry_run: bool) -> None:
    table = Table(title="Coauthor imputation", show_lines=False)
    table.add_column("Metric", style="white")
    table.add_column("Count", justify="right", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " "), f"{v:,}" if isinstance(v, int) else str(v))
    table.add_row("mode", "dry-run" if dry_run else "apply")
    console.print(table)
