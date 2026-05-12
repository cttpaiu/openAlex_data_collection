"""Command: impute-affiliation — fill missing institution_id + country_code.

Three stages:

* **Stage 1** — ``contributions.institution_id IS NULL`` and a non-empty
  ``raw_affiliation_string`` is present. The LLM (Ollama or Groq via
  langchain ``.with_structured_output``) extracts an institution name
  and country from the raw text. The extracted name is matched against
  existing institutions via ``InstitutionMatcher`` (sentence-transformers
  cosine similarity). On a match above ``--match-threshold`` the existing
  institution_id is referenced. Below threshold a synthetic institution
  is created with id ``IMP_<sha1[:10]>`` and ``is_synthetic = TRUE`` so
  re-runs are idempotent and OpenAlex IDs stay distinguishable.

* **Stage 2** — ``institutions.country_code IS NULL``. One LLM call per
  institution (deduped, cheap path). Rule-first inference on the
  institution display_name plus a single sample raw_affiliation_string
  drawn from any contribution that uses the institution; LLM fallback
  only on unresolved cases. Country fix cascades: every
  ``contributions.country_code IS NULL`` row whose ``institution_id``
  matches gets updated.

* **Stage 3** — ``contributions.country_code IS NULL`` after stages 1+2.
  Rule + LLM fallback on the raw affiliation.

The HK→CN normalisation lives in ``convert-to-db`` (load time). A final
``_merge_hong_kong_to_china`` pass remains here as an idempotent safety
net for databases produced before that move.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import deque
from pathlib import Path
from typing import Any

import click
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from openalex.config import load_config
from openalex.imputation import (
    CountryPrediction,
    CountryPredictionResponse,
    InstitutionMatch,
    InstitutionMatcher,
    InstitutionPrediction,
    InstitutionPredictionResponse,
    InstitutionRecord,
    infer_country_from_affiliation,
    normalize_country_code,
    synthetic_institution_id,
)

console = Console()

DEFAULT_MODEL = "sorc/qwen3.5-instruct:2b"
DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────────


@click.command("impute-affiliation")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
@click.option("--dry-run", is_flag=True, help="Show what would be updated without writing changes")
@click.option("--limit", type=int, default=None, help="Limit eligible rows per stage")
@click.option(
    "--stages", default="1,2,3", show_default=True,
    help="Comma-separated stages to run (1=institution, 2=inst-country backfill, 3=country)",
)
@click.option("--llm-fallback", is_flag=True, help="Allow LLM fallback (Stage 1 requires this; 2/3 use it as fallback)")
@click.option("--groq-api-key", default=None, help="Groq API key (or GROQ_API_KEY env var)")
@click.option("--llm-provider", default=None, help="LLM provider: groq or ollama")
@click.option("--llm-model", default=None, help="LLM model name (defaults to config llm.model)")
@click.option("--llm-base-url", default=None, help="Base URL for local provider (e.g. Ollama)")
@click.option("--llm-batch-size", type=int, default=20, show_default=True)
@click.option("--llm-min-confidence", type=float, default=0.8, show_default=True)
@click.option("--llm-max-tokens-stage1", type=int, default=600, show_default=True)
@click.option("--llm-max-tokens-stage2", type=int, default=400, show_default=True)
@click.option("--llm-max-tokens-stage3", type=int, default=300, show_default=True)
@click.option(
    "--match-threshold", type=float, default=0.78, show_default=True,
    help="Cosine similarity threshold for Stage 1 institution dedup",
)
def impute_affiliation_command(
    config_path: str,
    db_path: str | None,
    dry_run: bool,
    limit: int | None,
    stages: str,
    llm_fallback: bool,
    groq_api_key: str | None,
    llm_provider: str | None,
    llm_model: str | None,
    llm_base_url: str | None,
    llm_batch_size: int,
    llm_min_confidence: float,
    llm_max_tokens_stage1: int,
    llm_max_tokens_stage2: int,
    llm_max_tokens_stage3: int,
    match_threshold: float,
) -> None:
    """Impute missing institution_id and country_code via rule+LLM staged pipeline."""
    cfg = load_config(config_path)
    final_db_path = _resolve_db_path(cfg, db_path)
    if not Path(final_db_path).exists():
        console.print(f"[bold red]✗ Database not found:[/bold red] {final_db_path}")
        raise SystemExit(1)

    requested = {s.strip() for s in stages.split(",") if s.strip() in {"1", "2", "3"}}
    if not requested:
        raise SystemExit("--stages must be a comma-separated subset of 1,2,3")

    provider = (llm_provider or cfg.llm_provider or DEFAULT_PROVIDER).strip().lower()
    if provider not in {"groq", "ollama"}:
        raise SystemExit(f"Unsupported --llm-provider: {provider}. Use groq or ollama.")
    model = (llm_model or cfg.llm_model or DEFAULT_MODEL).strip()
    base_url = (llm_base_url or cfg.llm_base_url or DEFAULT_OLLAMA_BASE_URL).strip()

    api_key = ""
    if llm_fallback:
        if provider == "groq":
            api_key = _extract_api_key(
                groq_api_key or os.environ.get("GROQ_API_KEY", "") or cfg.groq_api_key
            )
            if not api_key:
                raise SystemExit("Set --groq-api-key, GROQ_API_KEY, or api.groq_key in config.")
        else:
            api_key = ""

    if "1" in requested and not llm_fallback:
        console.print("[yellow]⚠ Stage 1 requires --llm-fallback. Skipping Stage 1.[/yellow]")
        requested.discard("1")

    import duckdb

    con = duckdb.connect(final_db_path, read_only=False)
    try:
        _ensure_audit_table(con)
        results: dict[str, dict[str, Any]] = {}

        if "1" in requested:
            results["stage_1"] = _run_stage_1(
                con, dry_run, limit, provider, base_url, api_key, model,
                llm_batch_size, llm_min_confidence, match_threshold, llm_max_tokens_stage1,
            )
        if "2" in requested:
            results["stage_2"] = _run_stage_2(
                con, dry_run, limit,
                provider, base_url, api_key if llm_fallback else "",
                model, llm_batch_size, llm_min_confidence, llm_max_tokens_stage2, llm_fallback,
            )
        if "3" in requested:
            results["stage_3"] = _run_stage_3(
                con, dry_run, limit,
                provider, base_url, api_key if llm_fallback else "",
                model, llm_batch_size, llm_min_confidence, llm_max_tokens_stage3, llm_fallback,
            )

        hk_fix = 0
        if not dry_run:
            hk_fix = _merge_hong_kong_to_china(con)
            con.commit()

        _print_overall_summary(results, dry_run=dry_run, hk_fixed=hk_fix)
    finally:
        con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_db_path(cfg, db_path: str | None) -> str:
    if db_path:
        return db_path
    import questionary

    default = str(Path(cfg.db_dir) / "quantum_papers.duckdb")
    return questionary.text("Path to database:", default=default).ask() or default


def _extract_api_key(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        return ""
    match = re.search(r"(gsk_[A-Za-z0-9]+)", value)
    return match.group(1) if match else ""


def _batched(items, batch_size: int):
    step = max(1, batch_size)
    for i in range(0, len(items), step):
        yield items[i:i + step]


class _StageProgress:
    """Live progress bar + rolling tail of recent log lines.

    Wraps a `rich.Progress` (bar + spinner + ETA + elapsed) and a small
    `Panel` showing the last `tail` log lines so a long-running stage
    visibly tells the user "still working, here's what just happened."
    """

    def __init__(self, stage_name: str, total: int, tail: int = 5, unit: str = "batch"):
        self.stage_name = stage_name
        self.unit = unit
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        self.task = self.progress.add_task(stage_name, total=max(total, 1))
        self.logs: deque[str] = deque(maxlen=tail)
        self.live: Live | None = None

    def __enter__(self) -> "_StageProgress":
        self.live = Live(
            self._render(), console=console, refresh_per_second=4, transient=False
        )
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.live is not None:
            self.live.__exit__(exc_type, exc, tb)

    def _render(self) -> Group:
        if self.logs:
            body = "\n".join(self.logs)
        else:
            body = f"[dim]waiting for first {self.unit}...[/dim]"
        return Group(self.progress, Panel(body, title="Recent activity", border_style="dim"))

    def log(self, line: str) -> None:
        self.logs.append(line)
        if self.live is not None:
            self.live.update(self._render())

    def advance(self, n: int = 1) -> None:
        self.progress.advance(self.task, n)
        if self.live is not None:
            self.live.update(self._render())


def _langchain_structured_call(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    prompt: str,
    schema,
    max_tokens: int,
):
    """Invoke an LLM via langchain and force the response into `schema` (pydantic).

    Heavy imports are kept inside the function so non-imputation CLI commands
    start fast.
    """
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
            num_predict=max_tokens,
            format="json",
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=0,
            max_tokens=max_tokens,
        )
    else:
        raise RuntimeError(f"Unsupported LLM provider: {provider}")

    return llm.with_structured_output(schema).invoke(prompt)


def _ensure_audit_table(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS country_imputation_audit (
            run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            row_id INTEGER,
            paper_id VARCHAR,
            inferred_country_code VARCHAR,
            matched_terms TEXT,
            raw_affiliation_string TEXT,
            source VARCHAR,
            confidence DOUBLE,
            stage VARCHAR
        )
        """
    )
    try:
        con.execute("ALTER TABLE country_imputation_audit ADD COLUMN IF NOT EXISTS stage VARCHAR")
    except Exception:
        pass


def _ensure_country(con, code: str | None) -> str | None:
    norm = normalize_country_code(code)
    if not norm:
        return None
    exists = con.execute(
        "SELECT 1 FROM countries WHERE country_code = ? LIMIT 1", [norm]
    ).fetchone()
    if exists:
        return norm
    max_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM countries").fetchone()[0]
    con.execute(
        "INSERT INTO countries (id, country_name, country_code, status) VALUES (?, ?, ?, 1) ON CONFLICT DO NOTHING",
        [max_id + 1, f"[{norm}]", norm],
    )
    return norm


def _insert_audit(con, rows: list[dict[str, Any]], stage: str) -> None:
    if not rows:
        return
    con.executemany(
        """
        INSERT INTO country_imputation_audit (
            row_id, paper_id, inferred_country_code, matched_terms,
            raw_affiliation_string, source, confidence, stage
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.get("row_id"),
                r.get("paper_id"),
                normalize_country_code(r.get("country_code")),
                r.get("matched_terms"),
                r.get("raw_affiliation_string"),
                r.get("source"),
                float(r.get("confidence") or 0.0),
                stage,
            )
            for r in rows
        ],
    )


def _merge_hong_kong_to_china(con) -> int:
    """Idempotent safety net for legacy databases loaded before HK→CN at convert time."""
    _ensure_country(con, "CN")
    c1 = con.execute(
        "UPDATE contributions SET country_code = 'CN' WHERE UPPER(TRIM(country_code)) = 'HK'"
    ).fetchone()[0]
    c2 = con.execute(
        "UPDATE institutions SET country_code = 'CN' WHERE UPPER(TRIM(country_code)) = 'HK'"
    ).fetchone()[0]
    con.execute("DELETE FROM countries WHERE UPPER(TRIM(country_code)) = 'HK'")
    return int(c1 or 0) + int(c2 or 0)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: institution imputation
# ─────────────────────────────────────────────────────────────────────────────


def _load_eligible_inst_rows(con, limit: int | None):
    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""
    return con.execute(
        f"""
        SELECT row_id, paper_id, raw_affiliation_string
        FROM contributions
        WHERE institution_id IS NULL
          AND raw_affiliation_string IS NOT NULL
          AND TRIM(raw_affiliation_string) != ''
        {limit_sql}
        """
    ).fetchall()


def _load_existing_institutions(con) -> list[InstitutionRecord]:
    rows = con.execute(
        "SELECT id, display_name, country_code FROM institutions WHERE display_name IS NOT NULL"
    ).fetchall()
    return [InstitutionRecord(id=r[0], display_name=r[1], country_code=r[2]) for r in rows]


def _query_stage1_batch(
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    batch_rows: list[dict[str, Any]],
    max_tokens: int,
) -> list[InstitutionPrediction]:
    prompt = (
        "Task: from each affiliation string, extract the primary institution name "
        "(university, research institute, lab, or company) and its ISO-3166-1 alpha-2 country code.\n"
        "Rules:\n"
        "1) Keep row_id exactly as input.\n"
        "2) institution_name should be the most specific identifiable organisation, "
        "without departments or addresses (e.g. 'University of Oxford' not 'Dept of Physics, Univ of Oxford').\n"
        "3) Use null when uncertain.\n"
        f"Input:\n{json.dumps(batch_rows, ensure_ascii=False)}"
    )
    response = _langchain_structured_call(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        prompt=prompt,
        schema=InstitutionPredictionResponse,
        max_tokens=max_tokens,
    )
    return [InstitutionPrediction.from_pydantic(item) for item in response.predictions]


def _run_stage_1(
    con,
    dry_run: bool,
    limit: int | None,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    batch_size: int,
    min_confidence: float,
    match_threshold: float,
    max_tokens: int,
) -> dict[str, Any]:
    rows = _load_eligible_inst_rows(con, limit)
    eligible = len(rows)
    stats = {
        "eligible": eligible,
        "matched_existing": 0,
        "created_synthetic": 0,
        "country_only": 0,
        "low_confidence": 0,
        "skipped": 0,
    }
    if eligible == 0:
        console.print("[dim]Stage 1: nothing to impute.[/dim]")
        return {"stats": stats, "updates": [], "synthetic_inserts": []}

    matcher = InstitutionMatcher()
    matcher.index(_load_existing_institutions(con))

    by_id = {r[0]: {"row_id": r[0], "paper_id": r[1], "raw_affiliation_string": r[2]} for r in rows}
    updates: list[dict[str, Any]] = []
    synthetic_inserts: list[dict[str, Any]] = []
    seen_synthetic: set[str] = set()

    total_batches = math.ceil(eligible / max(batch_size, 1))
    with _StageProgress(f"Stage 1 — institution imputation ({eligible:,} rows)", total_batches) as bar:
        for batch_no, chunk in enumerate(_batched(rows, batch_size), start=1):
            request_rows = [
                {"row_id": r[0], "raw_affiliation_string": r[2]} for r in chunk
            ]
            try:
                predictions = _query_stage1_batch(
                    provider=provider,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    batch_rows=request_rows,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                bar.log(f"[red]batch {batch_no}/{total_batches} FAILED — {exc}[/red]")
                bar.advance()
                stats["skipped"] += len(chunk)
                continue

            batch_matched = 0
            batch_synth = 0
            batch_skipped = 0
            batch_lowconf = 0
            for pred in predictions:
                row = by_id.get(pred.row_id)
                if not row:
                    continue
                if pred.confidence < min_confidence:
                    stats["low_confidence"] += 1
                    batch_lowconf += 1
                    continue

                extracted_country = normalize_country_code(pred.country_code)
                inst_id: str | None = None
                inst_country: str | None = None
                source = "llm"
                matched_term = ""

                if pred.institution_name:
                    match = matcher.find_match(pred.institution_name, threshold=match_threshold)
                    if match:
                        inst_id = match.institution_id
                        inst_country = normalize_country_code(match.country_code) or extracted_country
                        matched_term = f"matched: {match.display_name} (score={match.score:.2f})"
                        stats["matched_existing"] += 1
                        batch_matched += 1
                    else:
                        syn_id = synthetic_institution_id(pred.institution_name)
                        inst_id = syn_id
                        inst_country = extracted_country
                        matched_term = f"synthetic: {pred.institution_name}"
                        if syn_id not in seen_synthetic:
                            seen_synthetic.add(syn_id)
                            synthetic_inserts.append({
                                "id": syn_id,
                                "display_name": pred.institution_name,
                                "country_code": extracted_country,
                            })
                        stats["created_synthetic"] += 1
                        batch_synth += 1
                elif extracted_country:
                    stats["country_only"] += 1
                    source = "llm"
                    matched_term = "country-only (no institution extracted)"

                if inst_id is None and not extracted_country:
                    stats["skipped"] += 1
                    batch_skipped += 1
                    continue

                updates.append({
                    "row_id": pred.row_id,
                    "paper_id": row["paper_id"],
                    "raw_affiliation_string": row["raw_affiliation_string"],
                    "institution_id": inst_id,
                    "country_code": inst_country,
                    "matched_terms": matched_term,
                    "source": source,
                    "confidence": pred.confidence,
                })

            bar.log(
                f"batch {batch_no}/{total_batches} → "
                f"matched={batch_matched} synth={batch_synth} "
                f"skip={batch_skipped} lowconf={batch_lowconf}"
            )
            bar.advance()

    if not dry_run:
        for inst in synthetic_inserts:
            cc = _ensure_country(con, inst["country_code"]) if inst["country_code"] else None
            con.execute(
                "INSERT INTO institutions (id, display_name, country_code, type, ror_id, is_synthetic) "
                "VALUES (?, ?, ?, NULL, NULL, TRUE) ON CONFLICT DO NOTHING",
                [inst["id"], inst["display_name"], cc],
            )
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
        _insert_audit(con, updates, stage="1")

    _print_stage_summary("Stage 1 — institution imputation", stats, len(updates), dry_run)
    return {"stats": stats, "updates": updates, "synthetic_inserts": synthetic_inserts}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: institution-country backfill
# ─────────────────────────────────────────────────────────────────────────────


def _load_institutions_missing_country(con, limit: int | None):
    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""
    return con.execute(
        f"""
        SELECT
            i.id,
            i.display_name,
            (SELECT raw_affiliation_string
             FROM contributions
             WHERE institution_id = i.id
               AND raw_affiliation_string IS NOT NULL
               AND TRIM(raw_affiliation_string) != ''
             LIMIT 1) AS sample_aff
        FROM institutions i
        WHERE i.country_code IS NULL
        {limit_sql}
        """
    ).fetchall()


def _query_stage2_batch(
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    batch_rows: list[dict[str, Any]],
    max_tokens: int,
) -> list[CountryPrediction]:
    prompt = (
        "Task: infer ISO-3166-1 alpha-2 country codes for each institution.\n"
        "Rules:\n"
        "1) Use both display_name and sample_affiliation when present.\n"
        "2) Keep row_id exactly as input.\n"
        "3) status is one of: unambiguous, ambiguous, none.\n"
        f"Input:\n{json.dumps(batch_rows, ensure_ascii=False)}"
    )
    response = _langchain_structured_call(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        prompt=prompt,
        schema=CountryPredictionResponse,
        max_tokens=max_tokens,
    )
    return [CountryPrediction.from_pydantic(item) for item in response.predictions]


def _run_stage_2(
    con,
    dry_run: bool,
    limit: int | None,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    batch_size: int,
    min_confidence: float,
    max_tokens: int,
    llm_enabled: bool,
) -> dict[str, Any]:
    rows = _load_institutions_missing_country(con, limit)
    eligible = len(rows)
    stats = {
        "eligible": eligible,
        "rule_applied": 0,
        "llm_applied": 0,
        "llm_low_confidence": 0,
        "unresolved": 0,
        "cascaded_contributions": 0,
    }
    if eligible == 0:
        console.print("[dim]Stage 2: nothing to backfill.[/dim]")
        return {"stats": stats, "updates": []}

    updates: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    with _StageProgress(
        f"Stage 2 rule pass — institution-country ({eligible:,} institutions)",
        eligible,
        unit="institution",
    ) as bar:
        for idx, (inst_id, display_name, sample_aff) in enumerate(rows, start=1):
            text = " | ".join(filter(None, [display_name, sample_aff]))
            inf = infer_country_from_affiliation(text)
            if inf.status == "unambiguous" and inf.country_code:
                updates.append({
                    "row_id": None,
                    "paper_id": None,
                    "institution_id": inst_id,
                    "country_code": normalize_country_code(inf.country_code),
                    "raw_affiliation_string": sample_aff,
                    "matched_terms": ", ".join(inf.matched_terms),
                    "source": "rule",
                    "confidence": 1.0,
                })
                stats["rule_applied"] += 1
            else:
                unresolved.append({
                    "row_id": inst_id,
                    "institution_id": inst_id,
                    "display_name": display_name,
                    "sample_affiliation": sample_aff,
                })
            if idx % 500 == 0 or idx == eligible:
                bar.log(
                    f"institution {idx:,}/{eligible:,} → "
                    f"rule={stats['rule_applied']:,} unresolved={len(unresolved):,}"
                )
            bar.advance()

    if unresolved and llm_enabled:
        total_batches_s2 = math.ceil(len(unresolved) / max(batch_size, 1))
        with _StageProgress(
            f"Stage 2 LLM pass ({len(unresolved):,} unresolved)",
            total_batches_s2,
        ) as bar:
            for batch_no, chunk in enumerate(_batched(unresolved, batch_size), start=1):
                req_rows = [
                    {
                        "row_id": r["row_id"],
                        "display_name": r["display_name"],
                        "sample_affiliation": r["sample_affiliation"],
                    }
                    for r in chunk
                ]
                try:
                    predictions = _query_stage2_batch(
                        provider=provider,
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        batch_rows=req_rows,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:
                    bar.log(f"[red]batch {batch_no}/{total_batches_s2} FAILED — {exc}[/red]")
                    bar.advance()
                    continue
                by_id = {r["institution_id"]: r for r in chunk}
                batch_applied = 0
                batch_lowconf = 0
                for pred in predictions:
                    rec = by_id.get(pred.row_id)
                    if not rec or pred.status != "unambiguous" or not pred.country_code:
                        continue
                    if pred.confidence < min_confidence:
                        stats["llm_low_confidence"] += 1
                        batch_lowconf += 1
                        continue
                    updates.append({
                        "row_id": None,
                        "paper_id": None,
                        "institution_id": rec["institution_id"],
                        "country_code": normalize_country_code(pred.country_code),
                        "raw_affiliation_string": rec["sample_affiliation"],
                        "matched_terms": "llm-inferred from display_name+sample_aff",
                        "source": "llm",
                        "confidence": pred.confidence,
                    })
                    stats["llm_applied"] += 1
                    batch_applied += 1
                bar.log(
                    f"batch {batch_no}/{total_batches_s2} → "
                    f"applied={batch_applied} lowconf={batch_lowconf}"
                )
                bar.advance()

    stats["unresolved"] = eligible - stats["rule_applied"] - stats["llm_applied"]

    if not dry_run and updates:
        for u in updates:
            cc = _ensure_country(con, u["country_code"])
            if not cc:
                continue
            con.execute(
                "UPDATE institutions SET country_code = ? WHERE id = ? AND country_code IS NULL",
                [cc, u["institution_id"]],
            )
            cascade = con.execute(
                """
                UPDATE contributions SET country_code = ?
                WHERE institution_id = ? AND country_code IS NULL
                """,
                [cc, u["institution_id"]],
            ).fetchone()
            if cascade and cascade[0]:
                stats["cascaded_contributions"] += int(cascade[0])
        _insert_audit(con, updates, stage="2")

    _print_stage_summary("Stage 2 — institution-country backfill", stats, len(updates), dry_run)
    return {"stats": stats, "updates": updates}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: contribution-country imputation (existing flow)
# ─────────────────────────────────────────────────────────────────────────────


def _load_eligible_rows(con, limit: int | None):
    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""
    return con.execute(
        f"""
        SELECT row_id, paper_id, raw_affiliation_string
        FROM contributions
        WHERE country_code IS NULL
          AND raw_affiliation_string IS NOT NULL
          AND TRIM(raw_affiliation_string) != ''
        {limit_sql}
        """
    ).fetchall()


def _compute_rule_inference(rows, on_row=None):
    """Rule-only country inference over (row_id, paper_id, raw_aff) tuples.

    Optional ``on_row(idx, total, updates_so_far, unresolved_so_far)`` is
    invoked after each row so callers can drive a progress display.
    """
    updates: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    ambiguous = 0
    none = 0
    total = len(rows)
    for idx, (row_id, paper_id, raw_aff) in enumerate(rows, start=1):
        inf = infer_country_from_affiliation(raw_aff)
        if inf.status == "unambiguous" and inf.country_code:
            updates.append({
                "row_id": row_id,
                "paper_id": paper_id,
                "raw_affiliation_string": raw_aff,
                "country_code": normalize_country_code(inf.country_code),
                "matched_terms": ", ".join(inf.matched_terms),
                "source": "rule",
                "confidence": 1.0,
            })
        else:
            unresolved.append({
                "row_id": row_id,
                "paper_id": paper_id,
                "raw_affiliation_string": raw_aff,
                "rule_status": inf.status,
            })
            if inf.status == "ambiguous":
                ambiguous += 1
            else:
                none += 1
        if on_row is not None:
            on_row(idx, total, len(updates), len(unresolved))
    return {
        "eligible": total,
        "updates": updates,
        "unresolved": unresolved,
        "ambiguous": ambiguous,
        "none": none,
    }


def _query_stage3_batch(
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    batch_rows: list[dict[str, Any]],
    max_tokens: int,
) -> list[CountryPrediction]:
    prompt = (
        "Task: infer ISO-3166-1 alpha-2 country codes from affiliation strings.\n"
        "Rules:\n"
        "1) Keep row_id exactly as input.\n"
        "2) status is one of: unambiguous, ambiguous, none.\n"
        f"Input:\n{json.dumps(batch_rows, ensure_ascii=False)}"
    )
    response = _langchain_structured_call(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        prompt=prompt,
        schema=CountryPredictionResponse,
        max_tokens=max_tokens,
    )
    return [CountryPrediction.from_pydantic(item) for item in response.predictions]


def _llm_infer_batched(
    unresolved_rows: list[dict[str, Any]],
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    batch_size: int,
    min_confidence: float,
    max_tokens: int,
    on_batch=None,
):
    """LLM Stage 3 pass with optional per-batch progress callback.

    ``on_batch(batch_no, total_batches, applied, lowconf, none_or_amb, failed_msg)``
    is invoked after each LLM call. ``failed_msg`` is ``None`` on success.
    """
    updates: list[dict[str, Any]] = []
    stats = {"attempted": 0, "applied": 0, "low_confidence": 0, "none_or_ambiguous": 0}
    by_id = {r["row_id"]: r for r in unresolved_rows}
    total_batches = math.ceil(len(unresolved_rows) / max(batch_size, 1))

    for batch_no, chunk in enumerate(_batched(unresolved_rows, batch_size), start=1):
        request_rows = [
            {"row_id": r["row_id"], "raw_affiliation_string": r["raw_affiliation_string"]}
            for r in chunk
        ]
        try:
            predictions = _query_stage3_batch(
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                model=model,
                batch_rows=request_rows,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if on_batch is not None:
                on_batch(batch_no, total_batches, 0, 0, 0, str(exc))
            continue
        stats["attempted"] += len(chunk)

        batch_applied = 0
        batch_lowconf = 0
        batch_none = 0
        for pred in predictions:
            rec = by_id.get(pred.row_id)
            if not rec:
                continue
            code = normalize_country_code(pred.country_code)
            if not code or pred.status != "unambiguous":
                stats["none_or_ambiguous"] += 1
                batch_none += 1
                continue
            if pred.confidence < min_confidence:
                stats["low_confidence"] += 1
                batch_lowconf += 1
                continue
            updates.append({
                "row_id": pred.row_id,
                "paper_id": rec["paper_id"],
                "raw_affiliation_string": rec["raw_affiliation_string"],
                "country_code": code,
                "matched_terms": "llm-inferred",
                "source": "llm",
                "confidence": pred.confidence,
            })
            stats["applied"] += 1
            batch_applied += 1
        if on_batch is not None:
            on_batch(batch_no, total_batches, batch_applied, batch_lowconf, batch_none, None)
    return updates, stats


def _run_stage_3(
    con,
    dry_run: bool,
    limit: int | None,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    batch_size: int,
    min_confidence: float,
    max_tokens: int,
    llm_enabled: bool,
) -> dict[str, Any]:
    rows = _load_eligible_rows(con, limit)
    if not rows:
        console.print("[dim]Stage 3: nothing to impute.[/dim]")
        return {"stats": {"eligible": 0, "rule_applied": 0, "llm_applied": 0}, "updates": []}

    eligible_n = len(rows)
    with _StageProgress(
        f"Stage 3 rule pass — country imputation ({eligible_n:,} rows)",
        eligible_n,
        unit="row",
    ) as bar:
        def _on_row(idx, total, applied, unresolved):
            if idx % 500 == 0 or idx == total:
                bar.log(f"row {idx:,}/{total:,} → rule={applied:,} unresolved={unresolved:,}")
            bar.advance()
        rule_result = _compute_rule_inference(rows, on_row=_on_row)

    updates = list(rule_result["updates"])
    stats = {
        "eligible": rule_result["eligible"],
        "rule_applied": len(rule_result["updates"]),
        "rule_ambiguous": rule_result["ambiguous"],
        "rule_none": rule_result["none"],
        "llm_attempted": 0,
        "llm_applied": 0,
        "llm_low_confidence": 0,
        "llm_none_or_ambiguous": 0,
    }

    if rule_result["unresolved"] and llm_enabled:
        unresolved_n = len(rule_result["unresolved"])
        total_batches_s3 = math.ceil(unresolved_n / max(batch_size, 1))
        with _StageProgress(
            f"Stage 3 LLM pass ({unresolved_n:,} unresolved)",
            total_batches_s3,
        ) as bar:
            def _on_batch(batch_no, total_batches, applied, lowconf, none_or_amb, failed):
                if failed is not None:
                    bar.log(f"[red]batch {batch_no}/{total_batches} FAILED — {failed}[/red]")
                else:
                    bar.log(
                        f"batch {batch_no}/{total_batches} → "
                        f"applied={applied} lowconf={lowconf} none/amb={none_or_amb}"
                    )
                bar.advance()
            llm_updates, llm_stats = _llm_infer_batched(
                unresolved_rows=rule_result["unresolved"],
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                model=model,
                batch_size=batch_size,
                min_confidence=min_confidence,
                max_tokens=max_tokens,
                on_batch=_on_batch,
            )
        updates.extend(llm_updates)
        stats["llm_attempted"] = llm_stats["attempted"]
        stats["llm_applied"] = llm_stats["applied"]
        stats["llm_low_confidence"] = llm_stats["low_confidence"]
        stats["llm_none_or_ambiguous"] = llm_stats["none_or_ambiguous"]

    if not dry_run and updates:
        for u in updates:
            cc = _ensure_country(con, u["country_code"])
            if not cc:
                continue
            con.execute(
                "UPDATE contributions SET country_code = ? WHERE row_id = ? AND country_code IS NULL",
                [cc, u["row_id"]],
            )
        _insert_audit(con, updates, stage="3")

    _print_stage_summary("Stage 3 — country imputation", stats, len(updates), dry_run)
    return {"stats": stats, "updates": updates}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def _print_stage_summary(title: str, stats: dict[str, Any], total_updates: int, dry_run: bool) -> None:
    table = Table(title=title, show_lines=False)
    table.add_column("Metric", style="white")
    table.add_column("Count", justify="right", style="green")
    for key, value in stats.items():
        table.add_row(key.replace("_", " "), f"{value:,}" if isinstance(value, int) else str(value))
    table.add_row("[bold]total updates[/bold]", f"{total_updates:,}")
    table.add_row("mode", "dry-run" if dry_run else "apply")
    console.print(table)


def _print_overall_summary(results: dict[str, dict[str, Any]], dry_run: bool, hk_fixed: int = 0) -> None:
    if not results:
        return
    table = Table(title="Overall Imputation Summary", show_lines=False)
    table.add_column("Stage", style="white")
    table.add_column("Eligible", justify="right")
    table.add_column("Updates", justify="right", style="green")
    for stage_name in ("stage_1", "stage_2", "stage_3"):
        if stage_name not in results:
            continue
        stage = results[stage_name]
        eligible = stage["stats"].get("eligible", 0)
        table.add_row(stage_name.replace("_", " "), f"{eligible:,}", f"{len(stage['updates']):,}")
    console.print(table)
    if hk_fixed:
        console.print(f"[dim]HK→CN normalization fixed {hk_fixed:,} legacy rows.[/dim]")
    if dry_run:
        console.print("[yellow]Dry-run: no changes written.[/yellow]")
