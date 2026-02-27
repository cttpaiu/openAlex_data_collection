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

# 4. Validate before collecting
uv run openalex validate --no-api    # syntax check, no internet needed
uv run openalex validate             # + API existence check for topic IDs

# 5. Explore
uv run openalex search               # how many papers match?
uv run openalex get-topics --details --csv   # discover topic IDs

# 6. Sample before full download
uv run openalex sample --size 385    # random validation sample

# 7. Collect
uv run openalex download             # download all → JSONL (~820 MB)
uv run openalex convert-to-db        # JSONL → DuckDB
uv run openalex check-db             # completeness health report
```

## Commands

| Command | Description |
|---|---|
| `openalex init` | Create config template files |
| `openalex validate` | Check `keywords.txt` and `topics.txt` before collecting |
| `openalex search` | Count papers matching keyword query (no topics filter) |
| `openalex get-topics` | List topics appearing in keyword results (`--details --csv` for full list) |
| `openalex search-filtered` | Count papers with both keyword + topic filters |
| `openalex sample --size N` | Random reservoir sample to validate query quality |
| `openalex download` | Download all matching papers to JSONL |
| `openalex convert-to-db` | Load JSONL into normalised DuckDB (5-table schema) |
| `openalex check-db` | Completeness health report (Tier 1/2/3 classification) |
| `openalex export-format` | Export to analysis CSVs *(coming soon)* |

## Config Files

All config lives in the `config/` directory, created by `openalex init`:

| File | Purpose |
|---|---|
| `config/collection.yml` | API key, date range, batch sizes, output paths |
| `config/keywords.txt` | Boolean search query (full-text search on titles + abstracts) |
| `config/topics.txt` | OpenAlex topic IDs, one per line (format: `T10020`) |

## Database Schema

The `convert-to-db` command produces a DuckDB with 5 tables:

| Table | Contents | ~Rows (96k papers) |
|---|---|---|
| `papers` | Core metadata per publication | 96,665 |
| `authors` | Researcher names + ORCIDs | 100,767 |
| `institutions` | Organisations + country codes | 7,686 |
| `countries` | Country reference table | 217 |
| `contributions` | Paper ↔ author ↔ institution links | 405,562 |

## Validation

Run `openalex validate` before any collection. It checks:

**Keywords (`keywords.txt`):** non-empty · balanced parentheses · even double quotes · uppercase operators (OR/AND/NOT) · no adjacent operators · no empty groups `()`

**Topics (`topics.txt`):** format `T` + 5 digits · optionally verifies each ID exists on OpenAlex (use `--no-api` to skip)

```bash
uv run openalex validate --no-api   # fast, offline
uv run openalex validate             # includes API check
```

## Technology Stack

| Package | Role |
|---|---|
| [click](https://click.palletsprojects.com/) | CLI command framework |
| [rich](https://rich.readthedocs.io/) | Coloured terminal output, tables, progress bars |
| [aiohttp](https://docs.aiohttp.org/) | Async HTTP for OpenAlex API |
| [duckdb](https://duckdb.org/) | Embedded analytical database |
| [polars](https://pola.rs/) | Fast JSONL processing |
| [pyyaml](https://pyyaml.org/) | Config file parsing |
| [questionary](https://questionary.readthedocs.io/) | Interactive terminal prompts |

## Documentation

Full documentation is available on GitHub Pages:

| Page | Description |
|---|---|
| [Overview](https://darunesh1.github.io/openAlex_data_collection/) | Project overview and quick start |
| [Installation](https://darunesh1.github.io/openAlex_data_collection/installation.html) | Detailed setup guide |
| [Workflow](https://darunesh1.github.io/openAlex_data_collection/workflow.html) | End-to-end pipeline walkthrough |
| [Commands](https://darunesh1.github.io/openAlex_data_collection/commands.html) | Full command reference |
| [Configuration](https://darunesh1.github.io/openAlex_data_collection/config.html) | All config fields explained |
| [Validation Rules](https://darunesh1.github.io/openAlex_data_collection/validation.html) | Keyword and topic ID rules |

## License

[MIT](LICENSE)
