# OpenAlex CLI

> A command-line pipeline for collecting, validating, and storing quantum computing publications from the [OpenAlex](https://openalex.org) API.

**📖 [Full Documentation →](https://darunesh1.github.io/openAlex_data_collection/)**

---

## Overview

This tool automates a multi-step data collection pipeline that was previously spread across four Jupyter notebooks. It wraps every stage — from query validation to database loading — into a single CLI you run from the terminal.

**Target dataset:** ~96,000 quantum computing papers (2003–2024), filtered by Boolean keywords and 89 OpenAlex topic IDs, stored in a normalised DuckDB database.

## Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.13 |
| [uv](https://docs.astral.sh/uv/) | latest |
| OpenAlex API key | free — [request here](https://openalex.org/account) |
| Disk space | ≥ 2 GB |

## Installation

```bash
git clone https://github.com/Darunesh1/openAlex_data_collection.git
cd openAlex_data_collection
uv sync
```

Verify:

```bash
uv run openalex --help
```

## Quick Start

```bash
# 1. Create config templates
uv run openalex init

# 2. Edit config/collection.yml — add your API key and email
# 3. Edit config/topics.txt — add topic IDs (one per line, format: T10020)
# 4. Edit config/anchor.txt — add must-find DOI/title anchor papers

# 5. Validate before collecting
uv run openalex validate --no-api    # syntax check, no internet needed
uv run openalex validate             # + API existence check for topic IDs

# 6. Explore
uv run openalex search               # keyword count + anchor coverage check
uv run openalex get-topics --details --csv   # discover topic IDs
uv run openalex search-filtered      # keyword+topic count + anchor coverage check
uv run openalex check-anchor         # explicit anchor coverage command

# 7. Sample before full download
uv run openalex sample --size 385    # random validation sample

# 8. Collect
uv run openalex download             # download all → JSONL (~820 MB)
uv run openalex convert-to-db        # JSONL → DuckDB
uv run openalex check-db             # paper-centric coverage health report

# 9. Impute missing institution/country
uv run openalex impute --help                  # list imputation sources
uv run openalex impute crossref                # restore raw_affiliation_string from CrossRef by DOI
uv run openalex impute llm --dry-run           # preview LLM imputation over raw_affiliation_string
uv run openalex impute llm --llm-fallback --llm-batch-size 20   # rule + batched LLM
# optional provider override (otherwise read from config/collection.yml -> llm.*)
uv run openalex impute llm --llm-fallback --llm-provider ollama --llm-model sorc/qwen3.5-instruct:2b
```

## Local LLM (Ollama)

`impute llm` and the LLM passes inside `impute pdf` / `wos-import-impute` need a running Ollama server. Lines like

```
batch 5/9 FAILED — All connection attempts failed
```

mean the daemon is not reachable at `llm.base_url` (default `http://localhost:11434`).

### Start it once

```bash
ollama serve                                     # foreground — Ctrl-C to stop
ollama pull sorc/qwen3.5-instruct:2b             # one-time model download
curl -s localhost:11434/api/tags                 # smoke-test
```

### Keep it alive in the background

```bash
# macOS / Linux — log to a file, survive the terminal
nohup ollama serve > ~/.ollama/serve.log 2>&1 &

# macOS Homebrew install — runs as a launchd service across reboots
brew services start ollama

# Stop it
brew services stop ollama       # service
pkill -f "ollama serve"         # nohup
```

### Tune for parallelism + throughput

Drop these env vars in your shell rc (or `launchctl setenv` on macOS) before `ollama serve` starts. They unlock concurrent decoding and a longer-lived model cache so the LLM passes don't bottleneck the pipeline.

```bash
export OLLAMA_NUM_PARALLEL=8           # parallel requests per model
export OLLAMA_MAX_LOADED_MODELS=2      # keep multiple models hot
export OLLAMA_KEEP_ALIVE=24h           # don't unload between calls
export OLLAMA_FLASH_ATTENTION=1        # faster attention on supported GPUs

ollama serve
```

Match the CLI side to what the server can absorb:

```bash
uv run openalex impute llm \
  --llm-fallback \
  --llm-batch-size 32 \
  --llm-concurrency 8 \
  --llm-provider ollama
```

Same flags work via `wos-import-impute`:

```bash
uv run openalex wos-import-impute \
  --wos-csv data/your_wos.csv \
  --db data/db/your.duckdb \
  --concurrency 20 \
  --llm-provider ollama
```

`--concurrency` controls OpenAlex fetch parallelism (capped by `cfg.concurrent_requests`); `--llm-concurrency` controls Ollama parallelism in the impute step. Tune each independently — OpenAlex tolerates 20–30 concurrent on a polite key, while Ollama parallelism is bounded by `OLLAMA_NUM_PARALLEL` and VRAM/RAM headroom.

## Documentation

The full docs are an [Astro Starlight](https://starlight.astro.build) site at
**https://darunesh1.github.io/openAlex_data_collection/** with two sidebars:

- **User Guide** — overview, installation, quick start, configuration, workflow, and a hand-written page per CLI command (`init`, `validate`, `search`, `get-topics`, `check-anchor`, `sample`, `download`, `convert-to-db`, `check-db`, `import-wos`, `impute crossref`, `impute llm`, `impute pdf`).
- **Developer Guide** — architecture, pipeline data flow, DuckDB schema, dependencies, testing, contributing, plus one reference page per `openalex/*.py` module and per command implementation.

To run the docs locally:

```bash
cd docs
npm install
npm run dev      # http://localhost:4321
npm run build    # static site → docs/dist
```

## Technology Stack

| Package | Role |
|---|---|
| [click](https://click.palletsprojects.com/) | CLI command framework |
| [rich](https://rich.readthedocs.io/) | Coloured terminal output, tables, progress bars |
| [aiohttp](https://docs.aiohttp.org/) | Async HTTP for OpenAlex / CrossRef / Unpaywall / arXiv |
| [duckdb](https://duckdb.org/) | Embedded analytical database |
| [polars](https://pola.rs/) | Fast JSONL + Excel processing |
| [pydantic](https://docs.pydantic.dev/) | Structured-output LLM response schemas |
| [langchain-ollama / -groq](https://python.langchain.com/) | LLM imputation via `.with_structured_output(...)` |
| [pymupdf](https://pymupdf.readthedocs.io/) | PDF text extraction for `impute pdf` |
| [rapidfuzz](https://rapidfuzz.github.io/RapidFuzz/) | Fuzzy title match in `import-wos` |
| [sentence-transformers](https://www.sbert.net/) | Institution-name cosine similarity matcher |

## License

[MIT](LICENSE)
