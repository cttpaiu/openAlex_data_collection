"""Command: impute-country — fill missing country_code from raw affiliations."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from openalex.config import load_config
from openalex.imputation import infer_country_from_affiliation, normalize_country_code

console = Console()


@click.command("impute-country")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
@click.option("--dry-run", is_flag=True, help="Show what would be updated without writing changes")
@click.option("--limit", type=int, default=None, help="Limit eligible rows processed")
@click.option("--llm-fallback", is_flag=True, help="Use Groq in batch mode for rule none/ambiguous rows")
@click.option("--groq-api-key", default=None, help="Groq API key (or set GROQ_API_KEY env var)")
@click.option("--llm-model", default="llama-3.1-8b-instant", show_default=True)
@click.option("--llm-batch-size", type=int, default=20, show_default=True)
@click.option("--llm-min-confidence", type=float, default=0.8, show_default=True)
def impute_country_command(
    config_path: str,
    db_path: str | None,
    dry_run: bool,
    limit: int | None,
    llm_fallback: bool,
    groq_api_key: str | None,
    llm_model: str,
    llm_batch_size: int,
    llm_min_confidence: float,
) -> None:
    """Impute missing contributions.country_code from contributions.raw_affiliation_string."""
    cfg = load_config(config_path)
    final_db_path = _resolve_db_path(cfg, db_path)

    if not Path(final_db_path).exists():
        console.print(f"[bold red]✗ Database not found:[/bold red] {final_db_path}")
        raise SystemExit(1)

    import duckdb

    con = duckdb.connect(final_db_path, read_only=False)
    try:
        rows = _load_eligible_rows(con, limit)
        if not rows:
            if not dry_run:
                _merge_hong_kong_to_china(con)
                con.commit()
            console.print("[green]✓ No eligible rows found (nothing to impute).[/green]")
            return

        result = _compute_rule_inference(rows)
        llm_used = False
        if llm_fallback and result["unresolved"]:
            api_key = _extract_api_key(groq_api_key or os.environ.get("GROQ_API_KEY", ""))
            if not api_key:
                raise SystemExit("Set --groq-api-key or GROQ_API_KEY for --llm-fallback.")
            llm_used = True
            llm_updates, llm_stats = _llm_infer_batched(
                unresolved_rows=result["unresolved"],
                api_key=api_key,
                model=llm_model,
                batch_size=llm_batch_size,
                min_confidence=llm_min_confidence,
            )
            result["updates"].extend(llm_updates)
            result["llm_stats"] = llm_stats
        else:
            result["llm_stats"] = {"attempted": 0, "applied": 0, "low_confidence": 0, "none_or_ambiguous": 0}

        _print_summary(result, dry_run=dry_run, llm_used=llm_used)

        if dry_run:
            return

        _ensure_audit_table(con)
        if result["updates"]:
            _apply_updates(con, result["updates"])
            _insert_audit_rows(con, result["updates"])

        hk_fix = _merge_hong_kong_to_china(con)
        con.commit()

        console.print(
            f"[green]✓ Applied {len(result['updates']):,} imputations.[/green] "
            f"[dim](HK→CN normalized: {hk_fix:,} rows)[/dim]"
        )
    finally:
        con.close()


def _resolve_db_path(cfg, db_path: str | None) -> str:
    if db_path:
        return db_path
    import questionary

    default = str(Path(cfg.db_dir) / "quantum_papers.duckdb")
    return questionary.text("Path to database:", default=default).ask() or default


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


def _compute_rule_inference(rows):
    updates: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    ambiguous = 0
    none = 0
    for row_id, paper_id, raw_aff in rows:
        inf = infer_country_from_affiliation(raw_aff)
        if inf.status == "unambiguous" and inf.country_code:
            updates.append(
                {
                    "row_id": row_id,
                    "paper_id": paper_id,
                    "raw_affiliation_string": raw_aff,
                    "country_code": normalize_country_code(inf.country_code),
                    "matched_terms": ", ".join(inf.matched_terms),
                    "source": "rule",
                    "confidence": 1.0,
                }
            )
        else:
            unresolved.append(
                {
                    "row_id": row_id,
                    "paper_id": paper_id,
                    "raw_affiliation_string": raw_aff,
                    "rule_status": inf.status,
                }
            )
            if inf.status == "ambiguous":
                ambiguous += 1
            else:
                none += 1
    return {
        "eligible": len(rows),
        "updates": updates,
        "unresolved": unresolved,
        "ambiguous": ambiguous,
        "none": none,
    }


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


def _extract_json_payload(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object")
    return parsed


def _query_groq_batch(api_key: str, model: str, batch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = (
        "Task: infer ISO-3166-1 alpha-2 country codes from affiliation strings.\n"
        "Rules:\n"
        "1) Return ONLY valid JSON object (no markdown, no explanation).\n"
        "2) Schema:\n"
        "{\n"
        '  "predictions": [\n'
        "    {\n"
        '      "row_id": <int>,\n'
        '      "country_code": <\"AA\" or null>,\n'
        '      "status": <\"unambiguous\"|\"ambiguous\"|\"none\">,\n'
        '      "confidence": <float 0..1>,\n'
        '      "reason": <short string>\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "3) Keep row_id exactly as input.\n"
        "4) If not sure, use country_code=null and status='none' or 'ambiguous'.\n"
        f"Input rows:\n{json.dumps(batch_rows, ensure_ascii=False)}"
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1800,
    }
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "https://api.groq.com/openai/v1/chat/completions",
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(body),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed: {(proc.stderr or '').strip()}")

    payload = json.loads((proc.stdout or "").strip())
    if "error" in payload:
        err = payload["error"]
        raise RuntimeError(f"Groq API error ({err.get('code','unknown')}): {err.get('message', err)}")
    content = payload["choices"][0]["message"]["content"]
    return _extract_json_payload(content)


def _llm_infer_batched(
    unresolved_rows: list[dict[str, Any]],
    api_key: str,
    model: str,
    batch_size: int,
    min_confidence: float,
):
    updates: list[dict[str, Any]] = []
    stats = {"attempted": 0, "applied": 0, "low_confidence": 0, "none_or_ambiguous": 0}
    by_id = {r["row_id"]: r for r in unresolved_rows}

    for chunk in _batched(unresolved_rows, batch_size):
        request_rows = [{"row_id": r["row_id"], "raw_affiliation_string": r["raw_affiliation_string"]} for r in chunk]
        response_obj = _query_groq_batch(api_key=api_key, model=model, batch_rows=request_rows)
        predictions = response_obj.get("predictions", [])
        stats["attempted"] += len(chunk)

        for pred in predictions:
            row_id = pred.get("row_id")
            if row_id not in by_id:
                continue
            raw = by_id[row_id]["raw_affiliation_string"]
            paper_id = by_id[row_id]["paper_id"]
            code = normalize_country_code((pred.get("country_code") or "").strip().upper() or None)
            status = (pred.get("status") or "").strip().lower()
            confidence = float(pred.get("confidence", 0.0) or 0.0)
            reason = (pred.get("reason") or "").strip()

            if not code or status != "unambiguous":
                stats["none_or_ambiguous"] += 1
                continue
            if confidence < min_confidence:
                stats["low_confidence"] += 1
                continue

            updates.append(
                {
                    "row_id": row_id,
                    "paper_id": paper_id,
                    "raw_affiliation_string": raw,
                    "country_code": code,
                    "matched_terms": reason[:500],
                    "source": "llm",
                    "confidence": confidence,
                }
            )
            stats["applied"] += 1

    return updates, stats


def _print_summary(result, dry_run: bool, llm_used: bool) -> None:
    updates = result["updates"]
    rule_updates = [u for u in updates if u["source"] == "rule"]
    llm_updates = [u for u in updates if u["source"] == "llm"]

    table = Table(title="Country Imputation Summary", show_lines=False)
    table.add_column("Metric", style="white")
    table.add_column("Count", justify="right", style="green")
    table.add_row("Eligible rows", f"{result['eligible']:,}")
    table.add_row("Rule unambiguous", f"{len(rule_updates):,}")
    table.add_row("Rule ambiguous", f"{result['ambiguous']:,}")
    table.add_row("Rule none", f"{result['none']:,}")
    if llm_used:
        table.add_row("LLM attempted", f"{result['llm_stats']['attempted']:,}")
        table.add_row("LLM applied", f"{result['llm_stats']['applied']:,}")
        table.add_row("LLM low confidence", f"{result['llm_stats']['low_confidence']:,}")
        table.add_row("LLM none/ambiguous", f"{result['llm_stats']['none_or_ambiguous']:,}")
    table.add_row("Total updates", f"{len(updates):,}")
    table.add_row("Mode", "dry-run" if dry_run else "apply")
    console.print(table)

    if updates:
        preview = Table(title="Preview (first 12 updates)", show_lines=False)
        preview.add_column("row_id", justify="right")
        preview.add_column("paper_id")
        preview.add_column("country")
        preview.add_column("source")
        preview.add_column("confidence")
        preview.add_column("reason/matched")
        for u in updates[:12]:
            preview.add_row(
                str(u["row_id"]),
                u["paper_id"],
                u["country_code"],
                u["source"],
                f"{u['confidence']:.2f}",
                (u["matched_terms"] or "")[:80],
            )
        console.print(preview)


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
            confidence DOUBLE
        )
        """
    )


def _ensure_country(con, code: str) -> None:
    max_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM countries").fetchone()[0]
    con.execute(
        "INSERT INTO countries (id, country_name, country_code, status) VALUES (?, ?, ?, 1) ON CONFLICT DO NOTHING",
        [max_id + 1, f"[{code}]", code],
    )


def _apply_updates(con, updates: list[dict[str, Any]]) -> None:
    for u in updates:
        code = normalize_country_code(u["country_code"])
        if not code:
            continue
        _ensure_country(con, code)
        con.execute(
            "UPDATE contributions SET country_code = ? WHERE row_id = ? AND country_code IS NULL",
            [code, u["row_id"]],
        )


def _insert_audit_rows(con, updates: list[dict[str, Any]]) -> None:
    con.executemany(
        """
        INSERT INTO country_imputation_audit (
            row_id, paper_id, inferred_country_code, matched_terms, raw_affiliation_string, source, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                u["row_id"],
                u["paper_id"],
                normalize_country_code(u["country_code"]),
                u["matched_terms"],
                u["raw_affiliation_string"],
                u["source"],
                float(u["confidence"]),
            )
            for u in updates
        ],
    )


def _merge_hong_kong_to_china(con) -> int:
    _ensure_country(con, "CN")
    c1 = con.execute(
        "UPDATE contributions SET country_code = 'CN' WHERE UPPER(TRIM(country_code)) = 'HK'"
    ).fetchone()[0]
    c2 = con.execute(
        "UPDATE institutions SET country_code = 'CN' WHERE UPPER(TRIM(country_code)) = 'HK'"
    ).fetchone()[0]
    con.execute("DELETE FROM countries WHERE UPPER(TRIM(country_code)) = 'HK'")
    return int(c1 or 0) + int(c2 or 0)
