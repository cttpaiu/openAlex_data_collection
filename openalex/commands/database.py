"""Command: convert-to-db — load JSONL into DuckDB (ported from load_openalex_data.py)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Set

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from openalex.imputation import normalize_country_code

console = Console()


@click.command("convert-to-db")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--input", "-i", "jsonl_path", default=None, help="Path to JSONL file")
@click.option("--output", "-o", "db_name", default=None, help="Output database filename")
@click.option("--db-dir", default=None, help="Output directory (default: from config)")
def convert_to_db_command(config_path: str, jsonl_path: str, db_name: str, db_dir: str) -> None:
    """Convert a JSONL file into a normalised DuckDB database."""
    from openalex.config import load_config
    cfg = load_config(config_path)

    import questionary

    if not jsonl_path:
        default = "data/raw/openalex.jsonl"
        jsonl_path = questionary.text("Path to JSONL file:", default=default).ask() or default

    if not Path(jsonl_path).exists():
        console.print(f"[bold red]✗ JSONL file not found:[/bold red] {jsonl_path}")
        raise SystemExit(1)

    if not db_name:
        db_name = questionary.text("Output database name:", default="quantum_papers.duckdb").ask() or "quantum_papers.duckdb"

    out_dir = Path(db_dir or cfg.db_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(out_dir / db_name)

    console.print(f"\n[bold]Loading:[/bold] [cyan]{jsonl_path}[/cyan] → [cyan]{db_path}[/cyan]\n")

    start = time.time()
    loader = OpenAlexLoader(db_path, jsonl_path)
    try:
        loader.load_existing_caches()
        loader.load_jsonl()
        elapsed = time.time() - start
        loader.print_stats(elapsed)
    finally:
        loader.close()


class OpenAlexLoader:
    """Load OpenAlex JSONL data into a normalised DuckDB database."""

    def __init__(self, db_path: str, jsonl_path: str):
        import duckdb
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self.con = duckdb.connect(db_path)
        self.author_cache: Set[str] = set()
        self.institution_cache: Set[str] = set()
        self.country_cache: Set[str] = set()
        self.stats = {
            "papers": 0, "authors": 0, "institutions": 0,
            "contributions": 0, "countries_added": 0, "errors": 0, "skipped": 0,
        }
        self._create_schema()
        self._init_contribution_sequence()
        self._load_country_cache()

    def _create_schema(self) -> None:
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id VARCHAR PRIMARY KEY, doi VARCHAR, title TEXT,
                publication_year INTEGER, publication_date VARCHAR, type VARCHAR,
                journal_name VARCHAR, journal_issn VARCHAR, is_core_journal BOOLEAN,
                publisher VARCHAR, is_oa BOOLEAN, oa_status VARCHAR, oa_url VARCHAR,
                cited_by_count INTEGER, citation_percentile DOUBLE,
                is_top_1_percent BOOLEAN, is_top_10_percent BOOLEAN, fwci DOUBLE,
                primary_topic_id VARCHAR, primary_topic_name VARCHAR, primary_topic_score DOUBLE,
                primary_topic_field VARCHAR, primary_topic_subfield VARCHAR, primary_topic_domain VARCHAR,
                institutions_distinct_count INTEGER, countries_distinct_count INTEGER,
                is_international BOOLEAN, abstract_text TEXT, updated_date VARCHAR
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                id VARCHAR PRIMARY KEY, display_name VARCHAR, orcid VARCHAR
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS institutions (
                id VARCHAR PRIMARY KEY, display_name VARCHAR,
                country_code VARCHAR, type VARCHAR, ror_id VARCHAR,
                is_synthetic BOOLEAN DEFAULT FALSE
            )
        """)
        try:
            self.con.execute(
                "ALTER TABLE institutions ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN DEFAULT FALSE"
            )
        except Exception:
            pass
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS countries (
                id INTEGER PRIMARY KEY, country_name VARCHAR, country_code VARCHAR UNIQUE, status INTEGER
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS contributions (
                row_id INTEGER PRIMARY KEY, paper_id VARCHAR, author_id VARCHAR,
                institution_id VARCHAR, country_code VARCHAR, author_name VARCHAR,
                author_position VARCHAR, is_corresponding BOOLEAN, raw_affiliation_string VARCHAR
            )
        """)

    def _init_contribution_sequence(self) -> None:
        try:
            result = self.con.execute("SELECT COALESCE(MAX(row_id), 0) FROM contributions").fetchone()
            max_id = result[0] if result else 0
            self.con.execute("DROP SEQUENCE IF EXISTS seq_contrib")
            self.con.execute(f"CREATE SEQUENCE seq_contrib START {max_id + 1}")
        except Exception as e:
            console.print(f"[yellow]⚠ Sequence init: {e}[/yellow]")

    def _load_country_cache(self) -> None:
        try:
            rows = self.con.execute("SELECT country_code FROM countries WHERE country_code IS NOT NULL").fetchall()
            self.country_cache = {r[0] for r in rows}
        except Exception:
            self.country_cache = set()

    def load_existing_caches(self) -> None:
        console.print("[dim]Loading existing data into cache...[/dim]")
        try:
            self.author_cache = {r[0] for r in self.con.execute("SELECT id FROM authors").fetchall()}
            self.institution_cache = {r[0] for r in self.con.execute("SELECT id FROM institutions").fetchall()}
            console.print(f"  • {len(self.author_cache):,} authors  • {len(self.institution_cache):,} institutions")
        except Exception:
            pass

    def _extract_id(self, url: str | None) -> str | None:
        if not url:
            return None
        return url.split("/")[-1] if "/" in url else url

    def _ensure_country(self, code: str | None) -> str | None:
        norm = normalize_country_code(code)
        if not norm or norm in self.country_cache:
            return norm
        try:
            max_id = self.con.execute("SELECT COALESCE(MAX(id), 0) FROM countries").fetchone()[0]
            self.con.execute(
                "INSERT INTO countries (id, country_name, country_code, status) VALUES (?, ?, ?, 1) ON CONFLICT DO NOTHING",
                [max_id + 1, f"[{norm}]", norm],
            )
            self.country_cache.add(norm)
            self.stats["countries_added"] += 1
        except Exception:
            pass
        return norm

    def _insert_author(self, data: Dict) -> None:
        aid = self._extract_id(data.get("id"))
        if not aid or aid in self.author_cache:
            return
        try:
            self.con.execute(
                "INSERT INTO authors (id, display_name, orcid) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                [aid, data.get("display_name") or "Unknown", data.get("orcid")],
            )
            self.author_cache.add(aid)
            self.stats["authors"] += 1
        except Exception:
            pass

    def _insert_institution(self, data: Dict) -> None:
        iid = self._extract_id(data.get("id"))
        if not iid or iid in self.institution_cache:
            return
        cc = normalize_country_code(data.get("country_code"))
        if cc:
            self._ensure_country(cc)
        try:
            self.con.execute(
                "INSERT INTO institutions (id, display_name, country_code, type, ror_id, is_synthetic) VALUES (?, ?, ?, ?, ?, FALSE) ON CONFLICT DO NOTHING",
                [iid, data.get("display_name") or "Unknown", cc, data.get("type"), data.get("ror")],
            )
            self.institution_cache.add(iid)
            self.stats["institutions"] += 1
        except Exception:
            pass

    def _reconstruct_abstract(self, inv: Dict | None) -> str | None:
        if not inv:
            return None
        try:
            words = {}
            for w, positions in inv.items():
                for p in positions:
                    words[p] = w
            return " ".join(words[i] for i in sorted(words))
        except Exception:
            return None

    def _first_nonempty_affiliation(self, raw_affs: list[str]) -> str | None:
        for raw in raw_affs:
            if raw and raw.strip():
                return raw.strip()
        return None

    def process_record(self, record: Dict) -> None:
        paper_id = self._extract_id(record.get("id"))
        if not paper_id:
            self.stats["skipped"] += 1
            return

        exists = self.con.execute("SELECT 1 FROM papers WHERE id = ?", [paper_id]).fetchone()
        if exists:
            self.stats["skipped"] += 1
            return

        try:
            pl = record.get("primary_location") or {}
            src = pl.get("source") or {}
            oa = record.get("open_access") or {}
            cp = record.get("citation_normalized_percentile") or {}
            pt = record.get("primary_topic") or {}
            sf = pt.get("subfield") or {}
            fd = pt.get("field") or {}
            dm = pt.get("domain") or {}
            cc = record.get("countries_distinct_count")

            self.con.execute(
                """INSERT INTO papers (
                    id, doi, title, publication_year, publication_date, type,
                    journal_name, journal_issn, is_core_journal, publisher,
                    is_oa, oa_status, oa_url, cited_by_count, citation_percentile,
                    is_top_1_percent, is_top_10_percent, fwci,
                    primary_topic_id, primary_topic_name, primary_topic_score,
                    primary_topic_field, primary_topic_subfield, primary_topic_domain,
                    institutions_distinct_count, countries_distinct_count, is_international,
                    abstract_text, updated_date
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    paper_id, record.get("doi"), record.get("title") or "Untitled",
                    record.get("publication_year"), record.get("publication_date"), record.get("type"),
                    src.get("display_name"), src.get("issn_l"), src.get("is_core"),
                    src.get("host_organization_name"),
                    oa.get("is_oa"), oa.get("oa_status"), oa.get("oa_url"),
                    record.get("cited_by_count"), cp.get("value"),
                    cp.get("is_in_top_1_percent"), cp.get("is_in_top_10_percent"), record.get("fwci"),
                    self._extract_id(pt.get("id")), pt.get("display_name"), pt.get("score"),
                    fd.get("display_name"), sf.get("display_name"), dm.get("display_name"),
                    record.get("institutions_distinct_count"), cc,
                    cc > 1 if cc else False,
                    self._reconstruct_abstract(record.get("abstract_inverted_index")),
                    record.get("updated_date"),
                ],
            )
            self.stats["papers"] += 1

            for authorship in record.get("authorships", []):
                author_data = authorship.get("author") or {}
                author_id = self._extract_id(author_data.get("id"))
                if not author_id:
                    continue
                self._insert_author(author_data)

                insts = authorship.get("institutions", [])
                raw_affs = authorship.get("raw_affiliation_strings", [])

                if insts:
                    for idx, inst_data in enumerate(insts):
                        inst_id = self._extract_id(inst_data.get("id"))
                        if not inst_id:
                            continue
                        self._insert_institution(inst_data)
                        ic = normalize_country_code(inst_data.get("country_code"))
                        if ic:
                            self._ensure_country(ic)
                        raw_aff = raw_affs[idx] if idx < len(raw_affs) else None
                        self.con.execute(
                            """INSERT INTO contributions (row_id, paper_id, author_id, institution_id,
                               country_code, author_name, author_position, is_corresponding, raw_affiliation_string)
                               VALUES (nextval('seq_contrib'),?,?,?,?,?,?,?,?)""",
                            [paper_id, author_id, inst_id, ic,
                             authorship.get("raw_author_name"), authorship.get("author_position"),
                             authorship.get("is_corresponding"), raw_aff],
                        )
                        self.stats["contributions"] += 1
                else:
                    countries = authorship.get("countries", [])
                    ac = normalize_country_code(countries[0] if countries else None)
                    raw_aff = self._first_nonempty_affiliation(raw_affs)
                    if ac:
                        self._ensure_country(ac)
                    self.con.execute(
                        """INSERT INTO contributions (row_id, paper_id, author_id, institution_id,
                           country_code, author_name, author_position, is_corresponding, raw_affiliation_string)
                           VALUES (nextval('seq_contrib'),?,?,NULL,?,?,?,?,?)""",
                        [paper_id, author_id, ac,
                         authorship.get("raw_author_name"), authorship.get("author_position"),
                         authorship.get("is_corresponding"), raw_aff],
                    )
                    self.stats["contributions"] += 1

        except Exception as e:
            self.stats["errors"] += 1

    def load_jsonl(self) -> None:
        total_lines = sum(1 for _ in open(self.jsonl_path, encoding="utf-8") if _.strip())

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[green]{task.completed:,}[/green]/[dim]{task.total:,}[/dim]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Loading papers...", total=total_lines)

            with open(self.jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self.process_record(record)
                    except json.JSONDecodeError:
                        self.stats["errors"] += 1
                    progress.advance(task)

    def print_stats(self, elapsed: float) -> None:
        s = self.stats
        console.print(
            f"\n[bold green]✓ Database created:[/bold green] [cyan]{self.db_path}[/cyan]\n"
            f"  Papers loaded:     [green]{s['papers']:>12,}[/green]\n"
            f"  Authors:           [green]{s['authors']:>12,}[/green]\n"
            f"  Institutions:      [green]{s['institutions']:>12,}[/green]\n"
            f"  Countries:         [green]{s['countries_added']:>12,}[/green]\n"
            f"  Contributions:     [green]{s['contributions']:>12,}[/green]\n"
            f"  Skipped:           [dim]{s['skipped']:>12,}[/dim]\n"
            + (f"  [red]Errors:[/red]            [red]{s['errors']:>12,}[/red]\n" if s['errors'] else "")
            + f"  Time:              [dim]{elapsed / 60:.1f} min[/dim]"
        )

    def close(self) -> None:
        self.con.close()
