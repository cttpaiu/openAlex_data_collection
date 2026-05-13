"""Command: ``openalex import-wos`` — load WoS country exports into DuckDB.

Per Excel file in ``data/QSM/``:

1. Resolve the filename to an ISO-3166-1 alpha-2 country code.
2. For each row:
   * Try DOI match against an in-memory index built from
     ``papers.doi``.
   * If no DOI hit, try exact normalised-title match.
   * If still no hit, try fuzzy normalised-title match
     (``rapidfuzz.fuzz.token_set_ratio`` ≥ threshold).
   * If still no match, insert a new ``papers`` row carrying the WoS
     metadata (no author, no institution).
3. Apply OR-semantics for ``is_top_1_percent`` / ``is_top_10_percent``
   on the matched/inserted paper, set ``from_wos = TRUE``, and remember
   the WoS accession number.
4. Insert one row in ``contributions`` (paper_id, country_code,
   source='wos') with ``author_id`` and ``institution_id`` NULL — this
   is a paper-level country marker, distinct from OpenAlex's per-author
   contribution rows.

Per-file ``con.commit()`` makes the run resumable: idempotent SQL means
re-running picks up only rows that weren't already imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from openalex.commands.impute_affiliation import (
    _StageProgress,
    _ensure_country,
    _resolve_db_path,
)
from openalex.config import load_config
from openalex.imputation import normalize_country_code
from openalex.wos import (
    AGGREGATE_FILENAMES,
    WosRecord,
    country_code_for_file,
    normalize_doi,
    normalize_title,
    read_wos_country_file,
)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Schema migration
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_wos_schema(con) -> None:
    """Add the columns the importer needs. Safe to call repeatedly."""
    con.execute(
        "ALTER TABLE papers ADD COLUMN IF NOT EXISTS from_wos BOOLEAN DEFAULT FALSE"
    )
    con.execute(
        "ALTER TABLE papers ADD COLUMN IF NOT EXISTS wos_accession_number VARCHAR"
    )
    con.execute(
        "ALTER TABLE contributions ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'openalex'"
    )
    # Backfill any pre-existing NULL source values produced before the column existed.
    con.execute(
        "UPDATE contributions SET source = 'openalex' WHERE source IS NULL"
    )
    con.commit()


# ─────────────────────────────────────────────────────────────────────────────
# In-memory matching indexes
# ─────────────────────────────────────────────────────────────────────────────


def _build_doi_index(con) -> dict[str, str]:
    rows = con.execute(
        "SELECT id, doi FROM papers WHERE doi IS NOT NULL AND TRIM(doi) != ''"
    ).fetchall()
    out: dict[str, str] = {}
    for paper_id, doi in rows:
        nd = normalize_doi(doi)
        if nd:
            out[nd] = paper_id
    return out


def _build_title_index(con) -> dict[str, str]:
    rows = con.execute(
        "SELECT id, title FROM papers WHERE title IS NOT NULL AND TRIM(title) != ''"
    ).fetchall()
    out: dict[str, str] = {}
    for paper_id, title in rows:
        nt = normalize_title(title)
        if nt and nt not in out:  # first writer wins
            out[nt] = paper_id
    return out


def _match_wos_to_paper(
    rec: WosRecord,
    doi_index: dict[str, str],
    title_index: dict[str, str],
    fuzzy_threshold: float,
    use_fuzzy: bool,
) -> tuple[str | None, str | None]:
    """Resolve a WoS record to an existing paper_id, or (None, None).

    Match path: DOI → exact-normalised title → fuzzy normalised title
    (rapidfuzz token_set_ratio above threshold). The returned method
    string carries the match path for audit / stats.
    """
    if rec.doi_normalized:
        pid = doi_index.get(rec.doi_normalized)
        if pid:
            return pid, "doi"

    if rec.title_normalized:
        pid = title_index.get(rec.title_normalized)
        if pid:
            return pid, "title_exact"

        if use_fuzzy and title_index:
            from rapidfuzz import fuzz, process

            cutoff = int(fuzzy_threshold * 100)
            match = process.extractOne(
                rec.title_normalized,
                title_index.keys(),
                scorer=fuzz.token_set_ratio,
                score_cutoff=cutoff,
            )
            if match is not None:
                matched_title, score, _ = match
                return title_index[matched_title], f"title_fuzzy:{int(score)}"

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Paper write paths
# ─────────────────────────────────────────────────────────────────────────────


def _paper_exists_top_flags(con, paper_id: str) -> tuple[bool, bool]:
    row = con.execute(
        "SELECT is_top_1_percent, is_top_10_percent FROM papers WHERE id = ?",
        [paper_id],
    ).fetchone()
    if not row:
        return False, False
    return bool(row[0]), bool(row[1])


def _update_existing_paper(con, paper_id: str, rec: WosRecord) -> None:
    """Apply OR-semantics for top-percentile flags and set from_wos.

    Stores the WoS accession number when the paper currently has none.
    Existing OpenAlex fields are otherwise untouched.
    """
    cur_top1, cur_top10 = _paper_exists_top_flags(con, paper_id)
    new_top1 = cur_top1 or rec.top_1_percent
    new_top10 = cur_top10 or rec.top_10_percent
    con.execute(
        """
        UPDATE papers
        SET is_top_1_percent = ?,
            is_top_10_percent = ?,
            from_wos = TRUE,
            wos_accession_number = COALESCE(wos_accession_number, ?)
        WHERE id = ?
        """,
        [new_top1, new_top10, rec.accession_number, paper_id],
    )


def _wos_synthetic_paper_id(rec: WosRecord) -> str:
    """Stable id for WoS-only papers: WOS_<accession-suffix>.

    Accession numbers are unique within WoS; use them directly so re-runs
    converge on the same id.
    """
    if rec.accession_number:
        suffix = rec.accession_number.replace("WOS:", "").strip()
        return f"WOS_{suffix}"
    # Fallback: hash the title — only used when WoS itself didn't ship an
    # accession number (very rare).
    import hashlib
    blob = (rec.title or "").strip().lower().encode("utf-8")
    return f"WOS_T{hashlib.sha1(blob).hexdigest()[:12]}"


def _insert_new_paper(con, rec: WosRecord) -> str:
    """Insert a WoS-only paper. Returns the new paper_id."""
    paper_id = _wos_synthetic_paper_id(rec)
    con.execute(
        """
        INSERT INTO papers (
            id, doi, title, publication_year, publication_date, type,
            journal_name, cited_by_count,
            is_top_1_percent, is_top_10_percent,
            from_wos, wos_accession_number
        ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, TRUE, ?)
        ON CONFLICT DO NOTHING
        """,
        [
            paper_id,
            rec.doi,
            rec.title or "Untitled",
            rec.publication_year,
            rec.document_type,
            rec.source,
            rec.times_cited,
            rec.top_1_percent,
            rec.top_10_percent,
            rec.accession_number,
        ],
    )
    return paper_id


def _insert_wos_contribution(con, paper_id: str, country_code: str) -> bool:
    """Insert paper-level country marker; return True if a row was added."""
    cc = normalize_country_code(country_code)
    if not cc:
        return False
    _ensure_country(con, cc)
    exists = con.execute(
        """
        SELECT 1 FROM contributions
        WHERE paper_id = ? AND country_code = ? AND source = 'wos'
        LIMIT 1
        """,
        [paper_id, cc],
    ).fetchone()
    if exists:
        return False
    con.execute(
        """
        INSERT INTO contributions (
            row_id, paper_id, author_id, institution_id, country_code,
            author_name, author_position, is_corresponding,
            raw_affiliation_string, source
        ) VALUES (nextval('seq_contrib'), ?, NULL, NULL, ?, NULL, NULL,
                  FALSE, NULL, 'wos')
        """,
        [paper_id, cc],
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Per-file driver
# ─────────────────────────────────────────────────────────────────────────────


def _process_file(
    con,
    path: Path,
    country_code: str,
    doi_index: dict[str, str],
    title_index: dict[str, str],
    fuzzy_threshold: float,
    use_fuzzy: bool,
    limit_rows: int | None,
    dry_run: bool,
    stats: dict[str, int],
) -> None:
    records = list(read_wos_country_file(path))
    if limit_rows is not None:
        records = records[:limit_rows]

    total = len(records)
    stats["rows_total"] += total

    with _StageProgress(
        f"import-wos: {path.name} ({total:,} rows → {country_code})",
        total,
        unit="row",
    ) as bar:
        for idx, rec in enumerate(records, start=1):
            if not rec.doi_normalized and not rec.title_normalized:
                stats["skipped_empty"] += 1
                bar.advance()
                continue

            paper_id, method = _match_wos_to_paper(
                rec, doi_index, title_index, fuzzy_threshold, use_fuzzy,
            )

            if paper_id is None:
                if dry_run:
                    paper_id = _wos_synthetic_paper_id(rec)
                else:
                    paper_id = _insert_new_paper(con, rec)
                stats["new_papers_inserted"] += 1
                method = "new"
                # Register in indexes so later rows in the same file can hit.
                if rec.doi_normalized:
                    doi_index[rec.doi_normalized] = paper_id
                if rec.title_normalized:
                    title_index.setdefault(rec.title_normalized, paper_id)
            else:
                if method == "doi":
                    stats["matched_by_doi"] += 1
                elif method == "title_exact":
                    stats["matched_by_title_exact"] += 1
                elif method and method.startswith("title_fuzzy"):
                    stats["matched_by_title_fuzzy"] += 1
                if not dry_run:
                    cur_top1, cur_top10 = _paper_exists_top_flags(con, paper_id)
                    if rec.top_1_percent and not cur_top1:
                        stats["top1_or_set"] += 1
                    if rec.top_10_percent and not cur_top10:
                        stats["top10_or_set"] += 1
                    _update_existing_paper(con, paper_id, rec)

            if not dry_run:
                if _insert_wos_contribution(con, paper_id, country_code):
                    stats["contribs_inserted_wos"] += 1
            else:
                stats["contribs_inserted_wos"] += 1

            if idx % 500 == 0 or idx == total:
                bar.log(
                    f"row {idx:,}/{total:,} → "
                    f"doi={stats['matched_by_doi']:,} "
                    f"title={stats['matched_by_title_exact'] + stats['matched_by_title_fuzzy']:,} "
                    f"new={stats['new_papers_inserted']:,}"
                )
            bar.advance()

    if not dry_run:
        con.commit()
    stats["files_processed"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────────


@click.command("import-wos")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
@click.option("--dir", "qsm_dir", default="data/QSM", show_default=True,
              help="Directory containing per-country *.xlsx exports")
@click.option("--dry-run", is_flag=True, help="Preview without DB writes")
@click.option("--limit-files", type=int, default=None,
              help="Cap number of country files (debug)")
@click.option("--limit-rows-per-file", type=int, default=None,
              help="Cap rows per file (debug)")
@click.option("--fuzzy-threshold", type=float, default=0.92, show_default=True,
              help="Min token_set_ratio (0-1) for fuzzy title match")
@click.option("--no-fuzzy", is_flag=True,
              help="Disable fuzzy title matching; DOI + exact title only")
@click.option("--only", "only_filename", default=None,
              help="Process a single filename (debug)")
def import_wos_command(
    config_path: str,
    db_path: str | None,
    qsm_dir: str,
    dry_run: bool,
    limit_files: int | None,
    limit_rows_per_file: int | None,
    fuzzy_threshold: float,
    no_fuzzy: bool,
    only_filename: str | None,
) -> None:
    """Import WoS country-level Excel exports into the DuckDB."""
    cfg = load_config(config_path)
    final_db_path = _resolve_db_path(cfg, db_path)
    if not Path(final_db_path).exists():
        console.print(f"[bold red]✗ Database not found:[/bold red] {final_db_path}")
        raise SystemExit(1)

    qsm_path = Path(qsm_dir)
    if not qsm_path.is_dir():
        console.print(f"[bold red]✗ QSM directory not found:[/bold red] {qsm_path}")
        raise SystemExit(1)

    candidates: list[tuple[Path, str]] = []
    for path in sorted(qsm_path.glob("*.xlsx")):
        if path.name in AGGREGATE_FILENAMES:
            continue
        if only_filename and path.name != only_filename:
            continue
        cc = country_code_for_file(path.name)
        if cc is None:
            console.print(
                f"[yellow]⚠ skipping {path.name}: no country mapping[/yellow]"
            )
            continue
        candidates.append((path, cc))

    if limit_files is not None:
        candidates = candidates[:limit_files]

    if not candidates:
        console.print("[dim]import-wos: no candidate files.[/dim]")
        return

    import duckdb

    con = duckdb.connect(final_db_path, read_only=False)
    try:
        _ensure_wos_schema(con)

        console.print(
            f"[bold cyan]import-wos[/bold cyan] — {len(candidates)} country files "
            f"(fuzzy={'off' if no_fuzzy else fuzzy_threshold})"
        )

        doi_index = _build_doi_index(con)
        title_index = _build_title_index(con)
        console.print(
            f"  indexed {len(doi_index):,} existing DOIs, "
            f"{len(title_index):,} titles"
        )

        stats: dict[str, int] = {
            "files_processed": 0,
            "rows_total": 0,
            "matched_by_doi": 0,
            "matched_by_title_exact": 0,
            "matched_by_title_fuzzy": 0,
            "new_papers_inserted": 0,
            "contribs_inserted_wos": 0,
            "top1_or_set": 0,
            "top10_or_set": 0,
            "skipped_empty": 0,
        }

        for path, cc in candidates:
            _process_file(
                con, path, cc, doi_index, title_index,
                fuzzy_threshold=fuzzy_threshold,
                use_fuzzy=not no_fuzzy,
                limit_rows=limit_rows_per_file,
                dry_run=dry_run,
                stats=stats,
            )

        _print_summary(stats, dry_run=dry_run)
    finally:
        con.close()


def _print_summary(stats: dict[str, int], dry_run: bool) -> None:
    table = Table(title="import-wos summary", show_lines=False)
    table.add_column("Metric", style="white")
    table.add_column("Count", justify="right", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " "), f"{v:,}")
    table.add_row("mode", "dry-run" if dry_run else "apply")
    console.print(table)
