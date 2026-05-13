"""Command: ``openalex impute pdf`` — recover institution + country from paper PDFs.

Pipeline (per paper): DOI → Unpaywall/arXiv URL → download PDF → first-page
text → langchain structured-output LLM → match authors to ``contributions``
rows → write ``institution_id`` + ``country_code`` per matched row, plus
audit. Per-paper commit makes the run resumable.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp
import click
from rich.console import Console
from rich.table import Table

from openalex.commands.impute_affiliation import (
    _StageProgress,
    _ensure_audit_table,
    _ensure_country,
    _insert_audit,
    _langchain_structured_call_async,
    _load_existing_institutions,
    _resolve_db_path,
)
from openalex.config import load_config
from openalex.imputation import (
    InstitutionMatcher,
    PdfAuthorItem,
    PdfAuthorResponse,
    normalize_country_code,
    synthetic_institution_id,
)
from openalex.pdf import (
    download_pdf,
    extract_first_page_text,
    find_pdf_url,
)

console = Console()

DEFAULT_PDF_CACHE = Path("data/cache/pdfs")
DEFAULT_MODEL = "sorc/qwen3.5-instruct:2b"
DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _parse_sources(raw: str) -> tuple[str, ...]:
    items = [s.strip().lower() for s in raw.split(",") if s.strip()]
    allowed = {"unpaywall", "openalex", "arxiv"}
    bad = [s for s in items if s not in allowed]
    if bad:
        raise click.BadParameter(f"unknown source(s): {bad}; allowed: {sorted(allowed)}")
    return tuple(items) or ("unpaywall", "openalex", "arxiv")


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


def _load_paper_target_rows(con, paper_id: str) -> list[dict[str, Any]]:
    """All contribution rows for a paper, with ORCID and current state."""
    rows = con.execute(
        """
        SELECT c.row_id, c.author_id, c.author_position, a.orcid,
               c.institution_id, c.country_code
        FROM contributions c
        LEFT JOIN authors a ON a.id = c.author_id
        WHERE c.paper_id = ?
        ORDER BY c.row_id
        """,
        [paper_id],
    ).fetchall()
    return [
        {
            "row_id": r[0],
            "author_id": r[1],
            "author_position": r[2],
            "orcid": (r[3] or "").strip() or None,
            "has_institution": r[4] is not None,
            "has_country": r[5] is not None,
        }
        for r in rows
    ]


def _match_pdf_authors(
    llm_authors: list[PdfAuthorItem],
    targets: list[dict[str, Any]],
) -> tuple[list[tuple[dict, PdfAuthorItem, str]], dict[str, int]]:
    """Pair LLM-extracted authors with contribution rows.

    ORCID first; position-index fallback only when both lists have the
    same remaining length. Anything else is reported as
    ``author_count_mismatch`` and dropped — no fuzzy-by-name.
    """
    s = {"matched_by_orcid": 0, "matched_by_position": 0, "author_count_mismatch": 0}
    matched: list[tuple[dict, PdfAuthorItem, str]] = []

    by_orcid = {a.orcid: a for a in llm_authors if a.orcid}
    remaining_targets: list[dict] = []
    remaining_llm = list(llm_authors)
    for t in targets:
        if t["orcid"] and t["orcid"] in by_orcid:
            la = by_orcid[t["orcid"]]
            matched.append((t, la, "pdf orcid"))
            s["matched_by_orcid"] += 1
            if la in remaining_llm:
                remaining_llm.remove(la)
        else:
            remaining_targets.append(t)

    if remaining_targets and remaining_llm:
        if len(remaining_targets) == len(remaining_llm):
            for idx, t in enumerate(remaining_targets):
                la = remaining_llm[idx]
                matched.append((t, la, f"pdf position[{idx}]"))
                s["matched_by_position"] += 1
        else:
            s["author_count_mismatch"] += len(remaining_targets)

    return matched, s


def _build_pdf_prompt(text: str) -> str:
    return (
        "You are given the first page of an academic paper. Extract every "
        "author together with the primary institution they are affiliated "
        "with on this paper and that institution's ISO-3166-1 alpha-2 "
        "country code.\n"
        "Rules:\n"
        "1) `position` is the zero-based order of the author in the byline.\n"
        "2) `institution` should be the most specific identifiable "
        "organisation (university, lab, company); skip department names and "
        "postal addresses.\n"
        "3) Use null when uncertain.\n"
        "Page text (truncated to first page):\n"
        "----- BEGIN -----\n"
        f"{text}\n"
        "----- END -----"
    )


async def _query_pdf_authors_async(
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    max_tokens: int,
) -> list[PdfAuthorItem]:
    response: PdfAuthorResponse = await _langchain_structured_call_async(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        prompt=_build_pdf_prompt(text),
        schema=PdfAuthorResponse,
        max_tokens=max_tokens,
    )
    return list(response.authors)


def _resolve_institution(
    matcher: InstitutionMatcher,
    name: str,
    country_code: str | None,
    threshold: float,
    seen_synthetic: set[str],
) -> tuple[str, str | None, str, dict[str, Any] | None]:
    """Return (institution_id, country_code, label, synthetic_insert_or_None).

    ``label`` is for audit (matched-existing vs synthetic).
    ``synthetic_insert_or_None`` is a row dict to insert into `institutions`
    when a new synthetic ID was minted; ``None`` for existing matches.
    """
    match = matcher.find_match(name, threshold=threshold)
    if match:
        cc = normalize_country_code(match.country_code) or country_code
        return (
            match.institution_id,
            cc,
            f"matched: {match.display_name} (score={match.score:.2f})",
            None,
        )
    syn_id = synthetic_institution_id(name)
    new_inst = None
    if syn_id not in seen_synthetic:
        seen_synthetic.add(syn_id)
        new_inst = {"id": syn_id, "display_name": name, "country_code": country_code}
    return (
        syn_id,
        country_code,
        f"synthetic: {name}",
        new_inst,
    )


def _flush_pdf_paper(
    con,
    paper_id: str,
    pairs: list[tuple[dict, PdfAuthorItem, str]],
    matcher: InstitutionMatcher,
    match_threshold: float,
    min_confidence: float,
    seen_synthetic: set[str],
    stats: dict[str, int],
) -> int:
    """Apply LLM extractions for one paper. Returns rows actually written."""
    audit_rows: list[dict[str, Any]] = []
    new_synthetics: list[dict[str, Any]] = []
    written = 0

    for target, llm_author, method in pairs:
        if llm_author.confidence < min_confidence:
            stats["llm_low_confidence"] += 1
            continue
        name = (llm_author.institution or "").strip()
        cc = normalize_country_code(llm_author.country_code)
        if not name and not cc:
            stats["llm_skipped_empty"] += 1
            continue

        inst_id: str | None = None
        inst_country: str | None = cc
        audit_label = method
        if name:
            (
                inst_id,
                inst_country,
                resolve_label,
                new_inst,
            ) = _resolve_institution(matcher, name, cc, match_threshold, seen_synthetic)
            audit_label = f"{method} | {resolve_label}"
            if new_inst is not None:
                new_synthetics.append(new_inst)
                stats["institution_created_synthetic"] += 1
            else:
                stats["institution_matched_existing"] += 1
        elif cc:
            stats["country_only"] += 1

        if not target["has_institution"] and inst_id is None and not inst_country:
            continue

        normalized_cc = _ensure_country(con, inst_country) if inst_country else None
        con.execute(
            """
            UPDATE contributions
            SET institution_id = COALESCE(institution_id, ?),
                country_code = COALESCE(country_code, ?)
            WHERE row_id = ?
            """,
            [inst_id, normalized_cc, target["row_id"]],
        )
        audit_rows.append({
            "row_id": target["row_id"],
            "paper_id": paper_id,
            "country_code": normalized_cc,
            "matched_terms": audit_label,
            "raw_affiliation_string": None,
            "source": "pdf",
            "confidence": float(llm_author.confidence or 0.0),
        })
        written += 1
        if inst_id is not None and not target["has_institution"]:
            stats["institution_set"] += 1
        if normalized_cc is not None and not target["has_country"]:
            stats["country_set"] += 1

    for inst in new_synthetics:
        cc_norm = _ensure_country(con, inst["country_code"]) if inst["country_code"] else None
        con.execute(
            "INSERT INTO institutions (id, display_name, country_code, type, ror_id, is_synthetic) "
            "VALUES (?, ?, ?, NULL, NULL, TRUE) ON CONFLICT DO NOTHING",
            [inst["id"], inst["display_name"], cc_norm],
        )

    if audit_rows:
        _insert_audit(con, audit_rows, stage="pdf")
        con.commit()
    return written


@click.command("pdf")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
@click.option("--dry-run", is_flag=True, help="Preview without DB writes")
@click.option("--limit", type=int, default=None, help="Cap eligible papers")
@click.option("--source", "source_csv", default="unpaywall,openalex,arxiv", show_default=True,
              help="Comma-separated PDF source order (unpaywall, openalex, arxiv)")
@click.option("--concurrent-downloads", type=int, default=4, show_default=True)
@click.option("--max-pdf-mb", type=int, default=30, show_default=True)
@click.option("--cache-dir", default=str(DEFAULT_PDF_CACHE), show_default=True)
@click.option("--skip-cache", is_flag=True, help="Force re-download even if cached")
@click.option("--request-timeout", type=int, default=60, show_default=True,
              help="HTTP timeout per request (seconds)")
@click.option("--llm-provider", default=None, help="LLM provider (groq or ollama; default from config)")
@click.option("--llm-model", default=None, help="LLM model name (default from config)")
@click.option("--llm-base-url", default=None, help="Base URL for local provider (e.g. Ollama)")
@click.option("--groq-api-key", default=None, help="Groq API key (or GROQ_API_KEY env var)")
@click.option("--llm-max-tokens", type=int, default=1500, show_default=True)
@click.option("--llm-min-confidence", type=float, default=0.0, show_default=True)
@click.option("--match-threshold", type=float, default=0.78, show_default=True,
              help="Cosine similarity threshold for institution dedup")
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
    llm_provider: str | None,
    llm_model: str | None,
    llm_base_url: str | None,
    groq_api_key: str | None,
    llm_max_tokens: int,
    llm_min_confidence: float,
    match_threshold: float,
) -> None:
    """Recover institution + country by LLM-reading the paper's PDF.

    Walks eligible papers (have DOI, at least one contribution row missing
    institution_id), resolves a downloadable URL via Unpaywall/arXiv,
    downloads to a content-addressed cache, extracts page-1 text, calls
    the LLM with a structured-output schema to extract authors +
    institutions + countries, then writes ``institution_id`` and
    ``country_code`` per matched contribution row.
    """
    import os

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

    provider = (llm_provider or cfg.llm_provider or DEFAULT_PROVIDER).strip().lower()
    if provider not in {"groq", "ollama"}:
        raise SystemExit(f"Unsupported --llm-provider: {provider}. Use groq or ollama.")
    model = (llm_model or cfg.llm_model or DEFAULT_MODEL).strip()
    base_url = (llm_base_url or cfg.llm_base_url or DEFAULT_OLLAMA_BASE_URL).strip()
    api_key = ""
    if provider == "groq":
        api_key = (groq_api_key or os.environ.get("GROQ_API_KEY", "") or cfg.groq_api_key or "").strip()
        if not api_key:
            raise SystemExit("Set --groq-api-key, GROQ_API_KEY, or api.groq_key in config.")

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
            f"provider={provider}, model={model})"
        )

        matcher = InstitutionMatcher()
        matcher.index(_load_existing_institutions(con))

        stats = asyncio.run(
            _run_pdf_pass(
                con=con,
                candidates=candidates,
                email=cfg.email,
                sources=sources,
                concurrency=concurrent_downloads,
                cache_dir=cache_path,
                max_bytes=max_pdf_mb * 1024 * 1024,
                skip_cache=skip_cache,
                request_timeout=request_timeout,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_tokens=llm_max_tokens,
                min_confidence=llm_min_confidence,
                match_threshold=match_threshold,
                matcher=matcher,
                dry_run=dry_run,
            )
        )
        _print_summary(stats, dry_run=dry_run)
    finally:
        con.close()


async def _run_pdf_pass(
    con,
    candidates: list[tuple[str, str]],
    email: str,
    sources: tuple[str, ...],
    concurrency: int,
    cache_dir: Path,
    max_bytes: int,
    skip_cache: bool,
    request_timeout: int,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    min_confidence: float,
    match_threshold: float,
    matcher: InstitutionMatcher,
    dry_run: bool,
) -> dict[str, int]:
    stats: dict[str, int] = {
        "papers_eligible": len(candidates),
        "url_found_unpaywall": 0,
        "url_found_openalex": 0,
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
        "llm_called": 0,
        "llm_failed": 0,
        "llm_no_authors": 0,
        "matched_by_orcid": 0,
        "matched_by_position": 0,
        "author_count_mismatch": 0,
        "institution_matched_existing": 0,
        "institution_created_synthetic": 0,
        "institution_set": 0,
        "country_set": 0,
        "country_only": 0,
        "llm_low_confidence": 0,
        "llm_skipped_empty": 0,
        "rows_written": 0,
        "papers_enriched": 0,
    }
    seen_synthetic: set[str] = set()

    headers = {
        "User-Agent": f"openalex-data-collection/0.1.0 (mailto:{email})",
        "Accept": "application/json, application/pdf, */*",
    }
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)

    cache_dir.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(
        headers=headers, timeout=timeout, connector=connector
    ) as session:
        async def _process(paper_id: str, doi: str) -> tuple[str, dict[str, Any]]:
            """Resolve URL, download, extract text, call LLM."""
            outcome: dict[str, Any] = {"paper_id": paper_id, "doi": doi}
            url, src = await find_pdf_url(session, doi, email, sources)
            outcome["url"], outcome["source"] = url, src
            if not url:
                return paper_id, outcome
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
            if not (path and not err):
                return paper_id, outcome

            text = extract_first_page_text(path)
            outcome["text_chars"] = len(text)
            outcome["text_empty"] = not bool(text)
            if not text:
                return paper_id, outcome

            try:
                authors = await _query_pdf_authors_async(
                    provider=provider,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    text=text,
                    max_tokens=max_tokens,
                )
                outcome["llm_authors"] = authors
            except Exception as exc:
                outcome["llm_err"] = str(exc)
            return paper_id, outcome

        # Bounded worker pool (producer/consumer): only `concurrency`
        # coroutines exist at any moment, not one per candidate.
        input_q: asyncio.Queue = asyncio.Queue()
        output_q: asyncio.Queue = asyncio.Queue()
        for pid, doi in candidates:
            input_q.put_nowait((pid, doi))
        for _ in range(max(1, concurrency)):
            input_q.put_nowait(None)

        async def _worker() -> None:
            while True:
                item = await input_q.get()
                if item is None:
                    return
                pid, doi = item
                try:
                    result = await _process(pid, doi)
                except Exception as exc:
                    result = (pid, {"paper_id": pid, "doi": doi, "err": f"crash:{exc}"})
                await output_q.put(result)

        with _StageProgress(
            f"impute pdf — URL+download+text ({len(candidates):,} papers)",
            len(candidates),
            unit="paper",
        ) as bar:
            workers = [
                asyncio.create_task(_worker())
                for _ in range(max(1, concurrency))
            ]
            try:
                for _ in range(len(candidates)):
                    paper_id, outcome = await output_q.get()
                    src = outcome.get("source")
                    if src == "unpaywall":
                        stats["url_found_unpaywall"] += 1
                    elif src == "openalex":
                        stats["url_found_openalex"] += 1
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
                        bar.advance(); continue
                    stats["text_extracted"] += 1
                    stats["chars_total"] += chars

                    llm_err = outcome.get("llm_err")
                    if llm_err:
                        stats["llm_failed"] += 1
                        bar.log(f"{paper_id} → [red]LLM failed: {llm_err}[/red]")
                        bar.advance(); continue
                    llm_authors = outcome.get("llm_authors") or []
                    stats["llm_called"] += 1
                    if not llm_authors:
                        stats["llm_no_authors"] += 1
                        bar.log(f"{paper_id} → [yellow]no authors extracted[/yellow]")
                        bar.advance(); continue

                    targets = _load_paper_target_rows(con, paper_id)
                    pairs, match_stats = _match_pdf_authors(llm_authors, targets)
                    stats["matched_by_orcid"] += match_stats["matched_by_orcid"]
                    stats["matched_by_position"] += match_stats["matched_by_position"]
                    stats["author_count_mismatch"] += match_stats["author_count_mismatch"]

                    writable = [
                        (t, la, m) for (t, la, m) in pairs
                        if not t["has_institution"] or not t["has_country"]
                    ]
                    if not writable:
                        bar.log(f"{paper_id} → already filled / no match")
                        bar.advance(); continue

                    if dry_run:
                        stats["rows_written"] += len(writable)
                        stats["papers_enriched"] += 1
                        bar.log(
                            f"{paper_id} → [green]dry: would write "
                            f"{len(writable)} rows[/green]"
                        )
                    else:
                        written = _flush_pdf_paper(
                            con, paper_id, writable, matcher,
                            match_threshold, min_confidence,
                            seen_synthetic, stats,
                        )
                        stats["rows_written"] += written
                        if written:
                            stats["papers_enriched"] += 1
                            bar.log(f"{paper_id} → [green]+{written} rows[/green]")
                        else:
                            bar.log(f"{paper_id} → 0 rows applied")
                    bar.advance()
            finally:
                await asyncio.gather(*workers, return_exceptions=True)

    return stats


def _print_summary(stats: dict[str, int], dry_run: bool) -> None:
    table = Table(title="impute pdf", show_lines=False)
    table.add_column("Metric", style="white")
    table.add_column("Count", justify="right", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " "), f"{v:,}")
    if stats["text_extracted"]:
        avg = stats["chars_total"] // stats["text_extracted"]
        table.add_row("avg chars per page-1", f"{avg:,}")
    table.add_row("mode", "dry-run" if dry_run else "apply")
    console.print(table)
