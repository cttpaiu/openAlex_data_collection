"""Command: impute-country — fill missing country/institution data.

Three stages:

* **Stage 1** — ``contributions.institution_id IS NULL`` and a non-empty
  ``raw_affiliation_string`` is present. Groq extracts an institution name
  and country from the raw text. The extracted name is matched against
  existing institutions via ``InstitutionMatcher`` (sentence-transformers
  cosine similarity). On a match above ``--match-threshold`` the existing
  institution_id is referenced. Below threshold a synthetic institution
  is created with id ``IMP_<sha1[:10]>`` and ``is_synthetic = TRUE`` so
  re-runs are idempotent and OpenAlex IDs stay distinguishable.

* **Stage 2** — ``institutions.country_code IS NULL``. One LLM call per
  institution (deduped, cheap path). Rule-first inference on the
  institution display_name plus a single sample raw_affiliation_string
  drawn from any contribution that uses the institution; Groq fallback
  only on unresolved cases. Country fix cascades: every
  ``contributions.country_code IS NULL`` row whose ``institution_id``
  matches gets updated.

* **Stage 3** — ``contributions.country_code IS NULL`` after stages 1+2.
  Rule + Groq fallback on the raw affiliation, identical to the original
  flow.

The HK→CN normalisation lives in ``convert-to-db`` (load time). A final
``_merge_hong_kong_to_china`` pass remains here as an idempotent safety
net for databases produced before that move.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import click
from rich.console import Console
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

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_PROVIDER = "groq"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
_GROQ_NEXT_ALLOWED_AT = 0.0
_GROQ_AVG_TOTAL_TOKENS = 900.0


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────────


@click.command("impute-country")
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
def impute_country_command(
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


def _extract_json_payload(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = content.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(content[start:])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object")
    return parsed


def _retry_wait_seconds(error_detail: str, fallback: float = 2.0, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    match = re.search(r"try again in\s*([0-9.]+)s", error_detail, re.IGNORECASE)
    if match:
        return max(float(match.group(1)), 1.0)
    return fallback


def _parse_reset_seconds(raw_reset: str | None) -> float:
    if not raw_reset:
        return 0.0
    value = raw_reset.strip().lower()
    if not value:
        return 0.0
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    total = 0.0
    for amount, unit in re.findall(r"([0-9]*\.?[0-9]+)\s*([hms])", value):
        num = float(amount)
        if unit == "h":
            total += num * 3600.0
        elif unit == "m":
            total += num * 60.0
        else:
            total += num
    return max(total, 0.0)


def _maybe_wait_for_groq_window() -> None:
    now = time.time()
    if _GROQ_NEXT_ALLOWED_AT > now:
        wait_seconds = _GROQ_NEXT_ALLOWED_AT - now
        console.print(f"[yellow]Groq window cooling down. Sleeping {wait_seconds:.1f}s...[/yellow]")
        time.sleep(wait_seconds)


def _apply_groq_rate_headers(headers, payload: dict[str, Any], max_tokens: int) -> None:
    global _GROQ_NEXT_ALLOWED_AT, _GROQ_AVG_TOTAL_TOKENS
    if headers is None:
        return
    remaining_raw = headers.get("x-ratelimit-remaining-tokens")
    reset_raw = headers.get("x-ratelimit-reset-tokens")
    if remaining_raw is None or reset_raw is None:
        return
    try:
        remaining_tokens = int(float(str(remaining_raw).strip()))
    except ValueError:
        return
    reset_seconds = _parse_reset_seconds(str(reset_raw))
    if reset_seconds <= 0:
        return

    usage_total = payload.get("usage", {}).get("total_tokens")
    request_tokens = int(usage_total) if isinstance(usage_total, (int, float)) else max_tokens
    _GROQ_AVG_TOTAL_TOKENS = (0.7 * _GROQ_AVG_TOTAL_TOKENS) + (0.3 * float(request_tokens))
    reserve_needed = int(max(_GROQ_AVG_TOTAL_TOKENS * 1.4, max_tokens * 0.8, 350))
    if remaining_tokens < reserve_needed:
        _GROQ_NEXT_ALLOWED_AT = max(_GROQ_NEXT_ALLOWED_AT, time.time() + reset_seconds)
        console.print(
            f"[yellow]Groq tokens low ({remaining_tokens} left). "
            f"Waiting {reset_seconds:.1f}s for reset...[/yellow]"
        )


def _ollama_chat(base_url: str, model: str, prompt: str, max_tokens: int = 400) -> dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode().strip()
        except Exception:
            detail = ""
        if detail:
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama network error: {exc.reason}") from exc

    if payload.get("error"):
        raise RuntimeError(f"Ollama API error: {payload['error']}")
    content = (
        payload.get("message", {}).get("content")
        or payload.get("response", "")
    )
    return _extract_json_payload(content)


def _groq_chat(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 1800,
    max_retries: int = 4,
) -> dict[str, Any]:
    """Single Groq chat-completion call expecting a JSON-object response."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    body_bytes = json.dumps(body).encode()
    req = urllib.request.Request(
        GROQ_URL,
        data=body_bytes,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )
    for attempt in range(max_retries + 1):
        _maybe_wait_for_groq_window()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
                _apply_groq_rate_headers(getattr(resp, "headers", None), payload, max_tokens=max_tokens)
            break
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode().strip()
            except Exception:
                detail = ""
            if exc.code == 429 and attempt < max_retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait_seconds = _retry_wait_seconds(
                    detail,
                    fallback=min(2 * (attempt + 1), 20),
                    retry_after=retry_after,
                )
                global _GROQ_NEXT_ALLOWED_AT
                _GROQ_NEXT_ALLOWED_AT = max(_GROQ_NEXT_ALLOWED_AT, time.time() + wait_seconds)
                console.print(
                    f"[yellow]Groq rate limit (429). Retrying in {wait_seconds:.1f}s "
                    f"({attempt + 1}/{max_retries})...[/yellow]"
                )
                continue
            if detail:
                raise RuntimeError(f"Groq HTTP {exc.code}: {detail}") from exc
            raise RuntimeError(f"Groq HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_retries:
                wait_seconds = min(2 * (attempt + 1), 10)
                console.print(
                    f"[yellow]Groq network issue. Retrying in {wait_seconds:.1f}s "
                    f"({attempt + 1}/{max_retries})...[/yellow]"
                )
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"Groq network error: {exc.reason}") from exc
    else:
        raise RuntimeError("Groq request failed after retries")

    if "error" in payload:
        err = payload["error"]
        raise RuntimeError(f"Groq API error ({err.get('code', 'unknown')}): {err.get('message', err)}")
    content = payload["choices"][0]["message"]["content"]
    return _extract_json_payload(content)


def _llm_chat(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    if provider == "groq":
        return _groq_chat(api_key=api_key, model=model, prompt=prompt, max_tokens=max_tokens)
    if provider == "ollama":
        return _ollama_chat(base_url=base_url, model=model, prompt=prompt, max_tokens=max_tokens)
    raise RuntimeError(f"Unsupported LLM provider: {provider}")


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

    console.print(f"[bold cyan]Stage 1 — institution imputation[/bold cyan] ({eligible:,} rows)")

    matcher = InstitutionMatcher()
    matcher.index(_load_existing_institutions(con))

    by_id = {r[0]: {"row_id": r[0], "paper_id": r[1], "raw_affiliation_string": r[2]} for r in rows}
    updates: list[dict[str, Any]] = []
    synthetic_inserts: list[dict[str, Any]] = []
    seen_synthetic: set[str] = set()

    for chunk in _batched(rows, batch_size):
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
            console.print(f"[red]Stage 1 batch failed:[/red] {exc}")
            stats["skipped"] += len(chunk)
            continue

        for pred in predictions:
            row = by_id.get(pred.row_id)
            if not row:
                continue
            if pred.confidence < min_confidence:
                stats["low_confidence"] += 1
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
            elif extracted_country:
                stats["country_only"] += 1
                source = "llm"
                matched_term = "country-only (no institution extracted)"

            if inst_id is None and not extracted_country:
                stats["skipped"] += 1
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
        '1) Return ONLY a valid JSON object: {"predictions":[{"row_id":<int>,'
        '"country_code":<"AA"|null>,"status":"unambiguous"|"ambiguous"|"none","confidence":<0..1>}]}\n'
        "2) Use both display_name and sample_affiliation when present.\n"
        "3) Keep row_id exactly as input.\n"
        f"Input:\n{json.dumps(batch_rows, ensure_ascii=False)}"
    )
    response = _llm_chat(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    return [CountryPrediction.from_dict(p) for p in response.get("predictions", [])]


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

    console.print(f"[bold cyan]Stage 2 — institution-country backfill[/bold cyan] ({eligible:,} institutions)")

    updates: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for inst_id, display_name, sample_aff in rows:
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

    if unresolved and llm_enabled:
        for chunk in _batched(unresolved, batch_size):
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
                console.print(f"[red]Stage 2 batch failed:[/red] {exc}")
                continue
            by_id = {r["institution_id"]: r for r in chunk}
            for pred in predictions:
                rec = by_id.get(pred.row_id)
                if not rec or pred.status != "unambiguous" or not pred.country_code:
                    continue
                if pred.confidence < min_confidence:
                    stats["llm_low_confidence"] += 1
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


def _compute_rule_inference(rows):
    updates: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    ambiguous = 0
    none = 0
    for row_id, paper_id, raw_aff in rows:
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
    return {
        "eligible": len(rows),
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
        '1) Return ONLY a valid JSON object: {"predictions":[{"row_id":<int>,'
        '"country_code":<"AA"|null>,"status":"unambiguous"|"ambiguous"|"none","confidence":<0..1>}]}\n'
        "2) Keep row_id exactly as input.\n"
        f"Input:\n{json.dumps(batch_rows, ensure_ascii=False)}"
    )
    response = _llm_chat(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    return [CountryPrediction.from_dict(p) for p in response.get("predictions", [])]


def _llm_infer_batched(
    unresolved_rows: list[dict[str, Any]],
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    batch_size: int,
    min_confidence: float,
    max_tokens: int,
):
    updates: list[dict[str, Any]] = []
    stats = {"attempted": 0, "applied": 0, "low_confidence": 0, "none_or_ambiguous": 0}
    by_id = {r["row_id"]: r for r in unresolved_rows}

    for chunk in _batched(unresolved_rows, batch_size):
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
            console.print(f"[red]Stage 3 batch failed:[/red] {exc}")
            continue
        stats["attempted"] += len(chunk)

        for pred in predictions:
            rec = by_id.get(pred.row_id)
            if not rec:
                continue
            code = normalize_country_code(pred.country_code)
            if not code or pred.status != "unambiguous":
                stats["none_or_ambiguous"] += 1
                continue
            if pred.confidence < min_confidence:
                stats["low_confidence"] += 1
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

    console.print(f"[bold cyan]Stage 3 — country imputation[/bold cyan] ({len(rows):,} rows)")

    rule_result = _compute_rule_inference(rows)
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
        llm_updates, llm_stats = _llm_infer_batched(
            unresolved_rows=rule_result["unresolved"],
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            min_confidence=min_confidence,
            max_tokens=max_tokens,
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
