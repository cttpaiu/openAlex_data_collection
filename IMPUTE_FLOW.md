# `impute llm` — working flow + history

Living doc. Updated 2026-05-18.

Database: `data/db/Quantum_Materials_and_Devices.duckdb`
Source CSV: `data/Quantum_Materials_and_Devices_wos.csv`
LLM provider: Ollama, local
Model: `sorc/qwen3.5-instruct:2b` (`config/collection.yml` → `llm.model`)

---

## Current status

- WoS import complete. 217,142 DOIs in DB.
- Smoke test (200 rows, all 3 stages) passes. Working flags pinned below.
- Full 32k-row run **not yet executed**. Will run on Ollama (~40 h estimated).

---

## Working command (proven on 200-row smoke)

```bash
uv run openalex impute llm \
  --llm-fallback \
  --llm-batch-size 8 \
  --llm-concurrency 8 \
  --llm-provider ollama \
  --llm-max-tokens-stage1 2000 \
  --llm-max-tokens-stage3 2000 \
  --llm-min-confidence 0.0
```

For the full run, prepend `nohup` and redirect logs so it survives the terminal:

```bash
nohup uv run openalex impute llm \
  --llm-fallback \
  --llm-batch-size 8 \
  --llm-concurrency 8 \
  --llm-provider ollama \
  --llm-max-tokens-stage1 2000 \
  --llm-max-tokens-stage3 2000 \
  --llm-min-confidence 0.0 \
  > impute_full.log 2>&1 &

tail -f impute_full.log
```

Prerequisite — Ollama must be running with parallelism enabled:

```bash
pkill -f "ollama serve"
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_KEEP_ALIVE=24h
ollama serve &
```

---

## Smoke-test results (200 rows, 2026-05-18)

| Stage   | Eligible | Updates | Notes |
|---------|---------:|--------:|-------|
| stage 1 |      200 |     194 | matched 118 / synth 76 / skip 6 |
| stage 2 |      112 |     112 | rule 6 / LLM 106 |
| stage 3 |      200 |     112 | LLM 112 applied / 88 none-or-ambiguous (model couldn't infer country) |

Wall time:
- Stage 1: 5:47 (25 batches × ~14 s)
- Stage 2 LLM: 3:26 (14 batches)
- Stage 3 LLM: 5:28 (25 batches)
- Total: ~15 min for 200 rows.

Linear scaling estimate for 32,113 rows: **~40 h**. Plan to run overnight / over a weekend.

---

## What was wrong + how it was fixed

### Initial symptom
Every batch logged `matched=0 synth=0 skip=0 lowconf=32`. 88/1004 batches in ~1h37m. Pipeline produced **zero updates**.

### Three root causes (ranked)

1. **`--llm-max-tokens-stage1` default 600 was too small** for `--llm-batch-size 32`.
   Each output row ≈ 25–35 tokens; 32 rows ≈ ~960 tokens needed. Response truncated mid-array → langchain `with_structured_output` dropped incomplete items → surviving items had no `confidence` field → Pydantic default `0.0` (`openalex/imputation.py:220`) → all rejected at the `<0.8` threshold (`openalex/commands/impute_affiliation.py:730`).
   **Fix:** `--llm-max-tokens-stage1 2000` (and `--llm-max-tokens-stage3 2000`).

2. **Ollama daemon defaulted to `OLLAMA_NUM_PARALLEL=1`.**
   Client-side concurrency was wired correctly (8 workers in `_run_batches_concurrently`, `openalex/commands/impute_affiliation.py:223-267`) but the server queued every request.
   **Fix:** restart `ollama serve` with `OLLAMA_NUM_PARALLEL=8` exported first.

3. **`sorc/qwen3.5-instruct:2b` under-calibrates `confidence`.**
   Even with truncation fixed, the 2B model emits low/zero confidence on most stage-1 and stage-3 outputs (it isn't trained to produce calibrated floats). With threshold = 0.5 we still saw 100 % lowconf on stage 1.
   **Fix:** `--llm-min-confidence 0.0`. The downstream sentence-transformers matcher (`MatchThreshold 0.78`) filters bad institution predictions on its own, so dropping the LLM threshold doesn't pollute the DB.

### Why `--llm-min-confidence 0.0` is safe here
Stage 1 has a second gate: `matcher.find_match(...)` against the existing OpenAlex institution table. Low-confidence LLM extractions that produce a real institution name still match a real `institution_id`; nonsense ones either get a `synthetic_` ID (visible, removable) or fall through to `skipped`. Stage 2's threshold-removal still saw 100 % `applied` with no obvious junk. Re-evaluate after the full run.

---

## Smoke command to re-verify quickly

```bash
uv run openalex impute llm \
  --llm-fallback \
  --llm-batch-size 8 \
  --llm-concurrency 8 \
  --llm-provider ollama \
  --llm-max-tokens-stage1 2000 \
  --llm-max-tokens-stage3 2000 \
  --llm-min-confidence 0.0 \
  --limit 200
```

Success signal: stage 1 `total updates >= ~180`, stage 2 `applied` close to eligible, stage 3 `applied` >= ~half of eligible.

---

## Resume strategy

Pipeline is naturally resumable:

- `eligible` rows are queried fresh at each stage start (rows where `institution_id IS NULL` / `country_code IS NULL`).
- DuckDB autocommits per batch via `_flush_stage1_batch` (`openalex/commands/impute_affiliation.py:564`).
- If the process is killed (Ctrl-C, crash, Ollama down), re-run the **same command** — already-imputed rows are no longer eligible and get skipped automatically.

---

## Known issues / open work

- **Stage 3 ceiling is ~56 %** with the 2B model. 88/200 rows came back `none/ambiguous` — the model can't infer country from the most generic affiliation strings. A bigger model (Groq Llama-3.3 70B) would lift this; deferred.
- **Confidence threshold is a runtime flag, not a config default.** Easy to forget. Consider lowering the default in `openalex/commands/impute_affiliation.py` (`--llm-min-confidence` default = 0.8) once we have more confidence the matcher catches everything.
- **Default model `sorc/qwen3.5-instruct:2b`** is hard-coded in `openalex/config.py:44`. Smaller than ideal. Bump default to `qwen2.5:7b-instruct` (still local) for better stage-1/3 quality.
- **No automatic Ollama health check** at the start of `impute llm`. Currently you find out it's down only when the first batch fails. Could add a `curl localhost:11434/api/tags` probe.

---

## Useful pointers

| What | Where |
|---|---|
| CLI command | `openalex/commands/impute_affiliation.py` |
| Pydantic schemas | `openalex/imputation.py:211-260` |
| Concurrency runner | `openalex/commands/impute_affiliation.py:223-267` |
| LLM client wiring | `openalex/commands/impute_affiliation.py:343-397` |
| Default model | `openalex/config.py:44` |
| Active model | `config/collection.yml` → `llm.model` |
